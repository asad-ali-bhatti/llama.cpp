// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration

#include "server-context-ssd-cache.h"

#include "server-context.h"
#include "server-task.h"
#include "llama.h"

#include <cstring>
#include <vector>

namespace llama {

// On-disk blob format for checkpoints that include a draft (MTP) context.
// The blob stored by kv_ssd_store is prefixed with this header when ctx_dft
// was present at save time. Legacy blobs (no header) are detected by the
// absence of the magic value and treated as tgt-only.
static constexpr uint32_t SSD_BLOB_MAGIC   = 0x53534454u; // "TDSS"
static constexpr uint32_t SSD_BLOB_VERSION = 1u;

struct ssd_blob_hdr {
    uint32_t magic;
    uint32_t version;
    uint64_t tgt_size;
    uint64_t dft_size; // 0 when no draft state was saved
};

uint64_t server_ssd_cache::store(uint32_t slot_id,
                                 struct llama_context* ctx,
                                 struct llama_context* ctx_dft,
                                 const common_prompt_checkpoint& ckpt,
                                 const llama_token* tokens,
                                 size_t tokens_size,
                                 uint32_t turn_id)
{
    if (!cache_ || !ctx || !ckpt.data_tgt.data()) return 0;

    // Serialize full tgt state (recurrent + KV cache) for cold-start recovery.
    const size_t tgt_size = llama_state_seq_get_size_ext(ctx, slot_id, 0);
    std::vector<uint8_t> tgt_data(tgt_size);
    if (llama_state_seq_get_data_ext(ctx, tgt_data.data(), tgt_size, slot_id, 0) != tgt_size) {
        LOG_WRN("SSD cache: tgt state serialization size mismatch (slot=%u)\n", slot_id);
        return 0;
    }

    // Serialize dft state when ctx_dft has independent memory (non-shared MTP).
    std::vector<uint8_t> dft_data;
    if (ctx_dft) {
        const size_t dft_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, 0);
        if (dft_size > 0) {
            dft_data.resize(dft_size);
            if (llama_state_seq_get_data_ext(ctx_dft, dft_data.data(), dft_size, slot_id, 0) != dft_size) {
                LOG_WRN("SSD cache: dft state serialization size mismatch (slot=%u) - skipping\n", slot_id);
                dft_data.clear();
            }
        }
    }

    // Build combined blob: [ssd_blob_hdr][tgt_data][dft_data]
    ssd_blob_hdr hdr;
    hdr.magic    = SSD_BLOB_MAGIC;
    hdr.version  = SSD_BLOB_VERSION;
    hdr.tgt_size = (uint64_t)tgt_data.size();
    hdr.dft_size = (uint64_t)dft_data.size();

    const size_t blob_size = sizeof(hdr) + tgt_data.size() + dft_data.size();
    std::vector<uint8_t> blob(blob_size);
    uint8_t* p = blob.data();
    std::memcpy(p, &hdr, sizeof(hdr));           p += sizeof(hdr);
    std::memcpy(p, tgt_data.data(), tgt_data.size()); p += tgt_data.size();
    if (!dft_data.empty()) {
        std::memcpy(p, dft_data.data(), dft_data.size());
    }

    // Use live KV-cache query for pos_min/max rather than ckpt.pos_min/max.
    // The in-memory checkpoint is created with PARTIAL_ONLY (recurrent state
    // only), so its pos fields reflect the last processed position, not the
    // actual KV coverage in the blob. Storing the real values here makes logs
    // accurate and enables future range-validity guards at restore time.
    const llama_pos real_pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), (llama_seq_id)slot_id);
    const llama_pos real_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), (llama_seq_id)slot_id);

    return kv_ssd_store(cache_, slot_id,
                        blob.data(), blob.size(),
                        real_pos_min, real_pos_max,
                        ckpt.n_tokens, turn_id,
                        (const uint32_t*)tokens, tokens_size,
                        cache_->compat_hash);
}

bool server_ssd_cache::load(uint64_t checkpoint_id,
                            struct llama_context* ctx,
                            int32_t& out_pos_min,
                            int32_t& out_pos_max,
                            uint64_t& out_n_tokens,
                            uint32_t dest_seq_id,
                            struct llama_context* ctx_dft)
{
    if (!cache_ || !ctx || checkpoint_id == 0) return false;

    const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache_, checkpoint_id);
    if (!meta) return false;

    std::vector<uint8_t> blob;
    if (!kv_ssd_load(cache_, checkpoint_id, blob)) return false;

    const uint32_t seq_id = (dest_seq_id != UINT32_MAX) ? dest_seq_id : meta->slot_id;

    // Detect new format (magic header) vs legacy (raw tgt blob).
    const uint8_t* tgt_ptr  = blob.data();
    size_t         tgt_bytes = blob.size();
    const uint8_t* dft_ptr  = nullptr;
    size_t         dft_bytes = 0;

    if (blob.size() >= sizeof(ssd_blob_hdr)) {
        ssd_blob_hdr hdr;
        std::memcpy(&hdr, blob.data(), sizeof(hdr));
        if (hdr.magic == SSD_BLOB_MAGIC && hdr.version == SSD_BLOB_VERSION) {
            tgt_ptr  = blob.data() + sizeof(hdr);
            tgt_bytes = (size_t)hdr.tgt_size;
            dft_ptr  = tgt_ptr + tgt_bytes;
            dft_bytes = (size_t)hdr.dft_size;

            if (sizeof(hdr) + tgt_bytes + dft_bytes > blob.size()) {
                LOG_WRN("SSD cache: blob size mismatch for checkpoint %lu - treating as legacy\n",
                        (unsigned long)checkpoint_id);
                tgt_ptr  = blob.data();
                tgt_bytes = blob.size();
                dft_ptr  = nullptr;
                dft_bytes = 0;
            }
        }
    }

    // Restore tgt state (KV cache + recurrent) under the current slot's seq_id.
    if (llama_state_seq_set_data_ext(ctx, tgt_ptr, tgt_bytes, (int32_t)seq_id, 0) == 0) {
        LOG_WRN("SSD cache: failed to restore tgt state for checkpoint %lu\n",
                (unsigned long)checkpoint_id);
        return false;
    }

    // Restore dft state (MTP KV cache) when caller provided ctx_dft.
    if (ctx_dft && dft_ptr && dft_bytes > 0) {
        if (llama_state_seq_set_data_ext(ctx_dft, dft_ptr, dft_bytes, (int32_t)seq_id, 0) == 0) {
            LOG_WRN("SSD cache: failed to restore dft state for checkpoint %lu - MTP will catch up\n",
                    (unsigned long)checkpoint_id);
            // Non-fatal: tgt is restored, MTP will degrade gracefully.
        }
    }

    out_pos_min  = meta->pos_min;
    out_pos_max  = meta->pos_max;
    out_n_tokens = meta->n_tokens;
    return true;
}

uint64_t server_ssd_cache::find_match(const llama_token* tokens, size_t tokens_size, uint32_t current_turn,
                                        uint64_t max_n_tokens, int32_t n_past,
                                        int32_t* out_lcp) {
    if (!cache_) return 0;
    return kv_ssd_find_match(cache_, (const uint32_t*)tokens, tokens_size, current_turn,
                             max_n_tokens, n_past, out_lcp);
}

uint64_t server_ssd_cache::find_by_slot(uint32_t slot_id, uint64_t min_tokens, uint32_t current_turn) {
    if (!cache_) return 0;
    return kv_ssd_find_by_slot(cache_, slot_id, min_tokens, current_turn);
}

void server_ssd_cache::on_turn_complete(uint32_t turn_id) {
    if (cache_) {
        kv_ssd_on_turn_complete(cache_, turn_id);
    }
}

} // namespace llama
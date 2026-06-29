// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration

#ifndef SERVER_CONTEXT_SSD_CACHE_H
#define SERVER_CONTEXT_SSD_CACHE_H

#include "common/kv-ssd-cache.h"
#include "llama.h"
#include "server-task.h"
#include "server-common.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>

namespace llama {

/**
 * Integrates the SSD-backed KV cache with the server checkpoint system.
 * Checkpoints flow through tiers: Hot (active) -> Warm (idle) -> Cold (SSD only).
 */
class server_ssd_cache {
public:
    server_ssd_cache(kv_ssd_cache* cache) : cache_(cache) {}
    ~server_ssd_cache() = default;

    // Store a checkpoint to SSD cache. Returns checkpoint ID (>0) on success.
    // ctx_dft is the MTP draft context; pass nullptr to skip saving draft state.
    uint64_t store(uint32_t slot_id,
                   struct llama_context* ctx,
                   struct llama_context* ctx_dft,
                   const common_prompt_checkpoint& ckpt,
                   const llama_token* tokens,
                   size_t tokens_size,
                   uint32_t turn_id);

    // Load a checkpoint by ID. Restores via llama_state_seq_set_data_ext.
    // dest_seq_id: the sequence ID to restore KV cells under. Pass the CURRENT
    // slot's id on cold-start cross-slot restore. Defaults to UINT32_MAX which
    // falls back to meta->slot_id (correct for same-slot warm loads).
    // ctx_dft: MTP draft context to restore alongside ctx_tgt. Pass nullptr
    // when no draft context exists or for same-slot warm loads.
    bool load(uint64_t checkpoint_id,
              struct llama_context* ctx,
              int32_t& out_pos_min,
              int32_t& out_pos_max,
              uint64_t& out_n_tokens,
              uint32_t dest_seq_id = UINT32_MAX,
              struct llama_context* ctx_dft = nullptr);

    // Find best matching checkpoint for a token sequence.
    // Searches within this conversation's cache only.
    uint64_t find_match(const llama_token* tokens, size_t tokens_size, uint32_t current_turn,
                        uint64_t max_n_tokens = UINT64_MAX,
                        int32_t n_past = -1,
                        int32_t* out_lcp = nullptr);

    // Find best checkpoint for a slot.
    uint64_t find_by_slot(uint32_t slot_id, uint64_t min_tokens, uint32_t current_turn);

    // Hint that a checkpoint will be needed soon.
    // Triggers kernel page cache prefetch to overlap SSD I/O with CPU work.
    void prefetch(uint64_t checkpoint_id) {
        kv_ssd_prefetch(cache_, checkpoint_id);
    }

    // Prefetch all cold checkpoints for a slot.
    void prefetch_slot(uint32_t slot_id) {
        kv_ssd_prefetch_slot(cache_, slot_id);
    }

    // Notify turn completion (triggers tier demotion).
    void on_turn_complete(uint32_t turn_id);

    // Get the underlying cache.
    kv_ssd_cache* get_cache() { return cache_; }

    // Set model compatibility hash (arch, dims, cache types).
    // Called once after model init. Subsequent stores/loads use this for validation.
    void set_compat_hash(uint64_t compat_hash) {
        kv_ssd_set_compat_hash(cache_, compat_hash);
    }

private:
    kv_ssd_cache* cache_; // Not owned
};

} // namespace llama

#endif // SERVER_CONTEXT_SSD_CACHE_H
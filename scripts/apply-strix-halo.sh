#!/usr/bin/env bash
# Local mirror of the apply-strix-halo CI workflow.
# Fetches the latest Strix Halo patches, verifies they apply cleanly,
# and creates/resets the local strix-halo branch from master.
# Does NOT push — leave that to you or the CI.
set -euo pipefail

PATCHES_REPO="gaetan-puleo/llama-cpp-strix-halo-patches"
COMBINED_PATCH="strix-halo-rdna35-combined.patch"
PATCH_FILE="/tmp/strix-halo-rdna35-combined.patch"

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
BLD='\033[1m'
RST='\033[0m'

info()  { echo -e "${BLD}[info]${RST}  $*"; }
ok()    { echo -e "${GRN}[ok]${RST}    $*"; }
warn()  { echo -e "${YLW}[warn]${RST}  $*"; }
fail()  { echo -e "${RED}[fail]${RST}  $*" >&2; }

# ── Prerequisites ──────────────────────────────────────────────────────────────
for cmd in git gh curl; do
  if ! command -v "$cmd" &>/dev/null; then
    fail "Required command not found: $cmd"
    exit 1
  fi
done

HAVE_CMAKE=false
if command -v cmake &>/dev/null; then
  HAVE_CMAKE=true
else
  warn "cmake not found — Gate 2 (CMake configure check) will be skipped locally."
  warn "Gate 2 always runs in CI (ubuntu-latest has cmake)."
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  fail "Not inside a git repository."
  exit 1
}

CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null) || {
  fail "HEAD is detached. Check out master before running this script."
  exit 1
}

if [ "$CURRENT_BRANCH" != "master" ]; then
  warn "Currently on '${CURRENT_BRANCH}', not 'master'."
  warn "The strix-halo branch will be created from HEAD, not master."
  read -r -p "Continue anyway? [y/N] " REPLY
  [[ "${REPLY,,}" == "y" ]] || { info "Aborted."; exit 0; }
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  fail "Working tree or index has uncommitted changes. Stash or commit first."
  exit 1
fi

# ── Download latest patch ──────────────────────────────────────────────────────
info "Fetching latest patch from ${PATCHES_REPO} ..."

PATCHES_SHA=$(gh api "repos/${PATCHES_REPO}/commits/HEAD" --jq '.sha')
info "Patches repo HEAD: ${PATCHES_SHA:0:8}"

PATCH_URL=$(gh api "repos/${PATCHES_REPO}/contents/${COMBINED_PATCH}" --jq '.download_url')
curl -fsSL -o "$PATCH_FILE" "$PATCH_URL"
ok "Patch downloaded: $(wc -l < "$PATCH_FILE") lines"

# ── Gate 1: git apply --check ──────────────────────────────────────────────────
info "Gate 1: verifying patch applies cleanly (dry-run) ..."

VERIFY_DIR=$(mktemp -d)
cleanup() {
  git worktree remove --force "$VERIFY_DIR" 2>/dev/null || true
  rm -rf "$VERIFY_DIR"
}
trap cleanup EXIT

git worktree add --detach "$VERIFY_DIR" HEAD -q

if ! git -C "$VERIFY_DIR" apply --check --3way "$PATCH_FILE" 2>&1; then
  fail "Gate 1 FAILED: patch does not apply cleanly against current master."
  fail "strix-halo branch has NOT been modified."
  exit 1
fi
ok "Gate 1 passed."

# ── Gate 2: CMake configure (CPU-only) ────────────────────────────────────────
if [ "$HAVE_CMAKE" = true ]; then
  info "Gate 2: applying patch in worktree and running CMake configure (CPU-only) ..."

  git -C "$VERIFY_DIR" apply --3way "$PATCH_FILE"

  CMAKE_BUILD_DIR="$VERIFY_DIR/build"
  if ! cmake \
    -B "$CMAKE_BUILD_DIR" \
    -S "$VERIFY_DIR" \
    -DGGML_HIP=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    --no-warn-unused-cli \
    2>&1; then
    fail "Gate 2 FAILED: CMake configure failed after applying the patch."
    fail "strix-halo branch has NOT been modified."
    exit 1
  fi
  ok "Gate 2 passed."
else
  warn "Gate 2 skipped (cmake not available)."
fi

cleanup
trap - EXIT

# ── Apply for real ─────────────────────────────────────────────────────────────
info "Both gates passed. Creating strix-halo branch from HEAD ..."

git switch -C strix-halo HEAD -q
git apply --3way --index "$PATCH_FILE"
git commit \
  -m "strix-halo: apply RDNA3.5 patches [patches@${PATCHES_SHA:0:8}]" \
  -m "Source: ${PATCHES_REPO}@${PATCHES_SHA}" \
  -m "Applied on top of: $(git rev-parse --short HEAD~1) (master)" \
  -q

echo
ok "strix-halo branch created at $(git rev-parse --short HEAD)"
info "To push:  git push --force origin strix-halo"

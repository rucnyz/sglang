#!/usr/bin/env bash
# Source this from every launch script. Splits responsibilities with .env:
#   .env     — pure key=value config (commit-friendly, dotenv-style)
#   env.sh   — shell-only bits: conda activate, PATH/LD_LIBRARY_PATH appending,
#              directory creation, helper functions.

set -euo pipefail

# Conda env (preinstalled: torch 2.11.0+cu130, sglang dev, mooncake, sgl_kernel, flash_mla).
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

# Load .env. set -a auto-exports every assignment.
_ENV_DIR="$(dirname "${BASH_SOURCE[0]}")/.."
set -a
# shellcheck disable=SC1091
source "${_ENV_DIR}/.env"
set +a
unset _ENV_DIR

# Shell-only path manipulations (need to read existing PATH / LD_LIBRARY_PATH).
# ~/.local/bin first so our patched mooncake_master (with PR #2174 TCP UAF
# fix) wins over the system /usr/local/bin/mooncake_master.
export PATH="${HOME}/.local/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$AGINFER_LOGS" "$AGINFER_RESULTS" "$AGINFER_MOONCAKE_SSD"

# rotate_log <path> — rename existing log with .prev so each run starts fresh.
rotate_log() {
    local p="$1"
    if [[ -s "$p" ]]; then
        mv "$p" "${p}.prev"
    fi
}

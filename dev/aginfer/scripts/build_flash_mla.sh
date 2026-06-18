#!/usr/bin/env bash
# Build & install DeepSeek FlashMLA from source for B300 (sm_100a) with CUDA 13.2.
#
# Required by sglang's `dsv4` attention backend (used for DeepSeek-V4-Flash).
# - `import flash_mla` in srt/layers/attention/deepseek_v4_backend.py is NOT yet
#   vendored into sgl_kernel (commit 621dfb888 migrated only flashmla_backend.py
#   and nsa_backend.py).
#
# CUDA 13 quirk: libcudacxx headers moved into `cccl/`. nvcc auto-adds it, but
# plain g++ (which compiles `csrc/pybind.cpp` host-side) doesn't, so cutlass
# fails on `#include <cuda/std/utility>`. We extend CPATH below.
set -euo pipefail

source "$(dirname "$0")/env.sh"

SRC="${FLASH_MLA_SRC:-/scratch/yuzhou/projects/FlashMLA}"
# Use the official deepseek-ai/FlashMLA repo, NOT the vllm-project fork at
# /scratch/yuzhou/projects/eb-vllm/.deps/flashmla-src (whose setup.py at commit
# 46d64a8 references a nonexistent csrc/sm90/decode/dense_fp8/ directory).
if [[ ! -d "$SRC" ]]; then
    echo "Cloning DeepSeek FlashMLA into $SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth=1 --recurse-submodules \
        https://github.com/deepseek-ai/FlashMLA.git "$SRC"
fi

export FLASH_MLA_DISABLE_SM90=1                   # we only have B300, save time
export TORCH_CUDA_ARCH_LIST="10.0"
export CPATH="/usr/local/cuda-13.2/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}"

LOG="$AGINFER_LOGS/flash_mla_build.log"
rotate_log "$LOG"
echo "[build_flash_mla] src=$SRC log=$LOG"

cd "$SRC"
rm -rf build *.egg-info     # ensure a clean build
nvcc --version | head -4 | tee -a "$LOG"
pip install --no-build-isolation -v . 2>&1 | tee -a "$LOG"

#!/usr/bin/env bash
# pristine_saturation — boot pristine sglang per (model × kv_dtype ×
# ssm_dtype) cell; capture server.log; emit cell descriptor JSON for
# validate_pristine.py.
#
# Default matrix: Qwen3.5-9B × 4 dtype combos. Override via CELLS env.
#
# Each cell:
#   1. Kill any stragglers
#   2. Boot pristine sglang (no SGLANG_HIMA, --radix-eviction-policy lru)
#      with the cell's kv_cache_dtype and SGLANG_MAMBA_SSM_DTYPE
#   3. Wait for /health=200
#   4. Tear down
# Then run validate_pristine.py to compare actual pool sizes to the
# HAND_VERIFIED constants in ../dtype_unit_sizes/.
#
# Usage:
#   GPU=3 PORT=30055 OUT_DIR=/tmp/pristine_run \
#       bash dev/interlayer/0_page_state_machine/pristine_saturation/run_pristine.sh

set -uo pipefail
# Script lives at dev/interlayer/0_page_state_machine/pristine_saturation/ —
# cd to sglang repo root.
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
OUT_DIR=${OUT_DIR:-/tmp/pristine_run}
GPU=${GPU:-3}
PORT=${PORT:-30055}
mkdir -p "$OUT_DIR"

model_path() {
    local m="$1"
    local base="$HUB/models--Qwen--$m"
    local snap
    snap=$(ls "$base/snapshots/" 2>/dev/null | head -1)
    [ -n "$snap" ] && echo "$base/snapshots/$snap"
}

run_cell() {
    local label="$1" model="$2" tp="$3" kv_dtype="$4" ssm_dtype="$5"
    local log="$OUT_DIR/$label.server.log"
    local desc="$OUT_DIR/$label.cell.json"
    echo "[$label] booting $model tp=$tp kv=$kv_dtype ssm=$ssm_dtype"

    # Write cell descriptor BEFORE any failure point so the validator
    # always sees this cell. If boot fails, validator parses an empty
    # log and reports a clear FAIL, rather than silently omitting.
    cat > "$desc" <<EOF
{"model": "$model", "tp": $tp, "kv_dtype": "$kv_dtype", "ssm_dtype": "$ssm_dtype"}
EOF

    local mp; mp=$(model_path "$model")
    if [ -z "$mp" ]; then
        echo "[$label] FAIL — $model not found in HUB" > "$log"
        echo "[$label] FAIL (model not in HUB)"
        return 1
    fi

    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    local kv_flag=""
    [ "$kv_dtype" != "auto" ] && kv_flag="--kv-cache-dtype $kv_dtype"

    CUDA_VISIBLE_DEVICES=$GPU \
        SGLANG_MAMBA_SSM_DTYPE="$ssm_dtype" \
    nohup $VENV -m sglang.launch_server \
        --model-path "$mp" --host 127.0.0.1 --port $PORT \
        --tp $tp --mem-fraction-static 0.85 \
        --max-running-requests 256 \
        --reasoning-parser qwen3 \
        --radix-eviction-policy lru \
        $kv_flag \
        --log-level info > "$log" 2>&1 &
    local pid=$!

    # Wait up to 10 min for ready
    local waited=0
    while [ $waited -lt 600 ]; do
        sleep 10; waited=$((waited + 10))
        if curl -s --max-time 2 -o /dev/null -w '%{http_code}' \
                "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
            echo "[$label] ready after ${waited}s"
            break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$label] FAILED — server died"
            tail -25 "$log" >&2
            return 1
        fi
    done

    if [ $waited -ge 600 ]; then
        echo "[$label] TIMEOUT"
        kill -9 $pid 2>/dev/null
        return 1
    fi

    # Teardown
    kill -9 $pid 2>/dev/null
    sleep 3
}

# === Matrix ===
# Default: Qwen3.5-9B × {auto, fp8_e4m3} kv × {float32, bfloat16} ssm = 4 cells
# Each cell ~30-60s boot on 9B. Total ~3 min.
CELLS_DEFAULT=(
    "9B_auto_fp32   Qwen3.5-9B       1 auto      float32"
    "9B_auto_bf16   Qwen3.5-9B       1 auto      bfloat16"
    "9B_fp8_fp32    Qwen3.5-9B       1 fp8_e4m3  float32"
    "9B_fp8_bf16    Qwen3.5-9B       1 fp8_e4m3  bfloat16"
)

CELLS=${CELLS:-default}
if [ "$CELLS" = "default" ]; then
    cells_arr=("${CELLS_DEFAULT[@]}")
else
    IFS=$'\n' read -r -d '' -a cells_arr < <(printf '%s\n' "$CELLS")
fi

any_cell_failed=0
for c in "${cells_arr[@]}"; do
    # shellcheck disable=SC2086
    if ! run_cell $c; then
        any_cell_failed=1
        # keep going so other cells still run, but remember the failure
    fi
done
[ $any_cell_failed -eq 1 ] && echo "WARNING: one or more cells failed to boot"

echo
echo "=== Validation ==="
$VENV dev/interlayer/0_page_state_machine/pristine_saturation/validate_pristine.py --out-dir "$OUT_DIR"

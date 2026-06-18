#!/bin/bash
#
# vllm_baseline.sh — vLLM cross-engine baseline for tab:main-cross-model.
#
# Runs each (model, regime) cell N_TRIALS times under vLLM v0.20.0 and emits
# bench.json files matching the SGLang-side schema, so aggregate.py can union
# them with the stock / static-best / Fulcrum runs.
#
# Strategy: three sequential phases, one per model. Each phase keeps all 8
# GPUs busy at the model's native TP — Kimi/Qwen3-Next at TP=2 (4 parallel
# workers on GPU pairs), Qwen3.5 at TP=1 (8 parallel workers on single GPUs).
# This keeps every cell apples-to-apples with its SGLang counterpart while
# still using the whole node.
#
# Why per-model TP and not a uniform TP:
#   - Kimi-48B OOMs in vLLM v0.20 at TP=1 with --gpu-memory-utilization 0.85
#     (CUDA-graph capture wants ~30 GiB on top of 96 GB weights), so TP=2.
#   - Qwen3-Next 80B does not fit on a single H200, so TP=2.
#   - Qwen3.5-35B fits at TP=1; matching SGLang's TP=1 baseline is required
#     for a fair vLLM-vs-SGLang comparison.
#
# Why three sequential phases and not one big static round-robin:
#   - TP=1 cells take 1 GPU and TP=2 cells take 2 GPUs, so a single static
#     split into N workers cannot keep all 8 GPUs saturated.
#   - Sequencing by model lets each phase pick the worker count that exactly
#     fills 8 GPUs at that phase's TP.
#
# Logging guarantees:
#   - Every cell logs a "start" and "DONE" line with timestamps.
#   - Every "DONE" is followed by a validation line: "  ok reqs=<N>" if
#     bench.json contains a valid run, otherwise "  FAIL <reason>" plus the
#     last 12 lines of runner.log dumped to the main log.
#   - Each worker keeps a local progress counter ("[W0] 3/4 cells done").
#   - End-of-phase prints a model-level OK / FAIL count.
#   - End-of-script prints a global summary table.
#
# Usage:
#   bash dev/eval/main/vllm_baseline.sh [N_TRIALS]
#   N_TRIALS defaults to 5 (matches the stock-baseline n=5 protocol).
#
# Output:
#   dev/eval/runs/vllm-baseline-<timestamp>/<model>/<regime>/<trial>/bench.json
#   plus runner.log / server.log / client.log per cell.
#   Top-level main log written to stdout (redirect when launching with nohup).
#
# Aggregate after:
#   python3 dev/eval/main/aggregate.py dev/eval/runs/vllm-baseline-<ts>
#
# Prereqs:
#   - vLLM v0.20.0 venv at /data/yuzhou/projects/vllm/.venv (used by run_vllm.sh)
#   - All 8 GPUs idle on this host (no other CUDA workloads).
#   - genai-bench installed in the SGLang fork's venv (used for m1/m2 client).

set -u

ROOT=/scratch/yuzhou/projects/sglang
cd "$ROOT"

N_TRIALS=${1:-5}
# Fixed run dir — each invocation overwrites the previous results in place.
RUN="dev/eval/runs/vllm-baseline"
rm -rf "$RUN"
mkdir -p "$RUN"

# Tee stdout/stderr to $RUN/main.log so progress is persistent regardless
# of how the script was launched (terminal / nohup / cron).
exec > >(tee -a "$RUN/main.log") 2>&1
echo "[vllm_baseline] main log: $RUN/main.log"

REGIMES=(m1 m2 m3)

# Phase counters, written by each worker, read at end of phase.
PHASE_OK_FILE=""
PHASE_FAIL_FILE=""

# Global summary file, one line per finished cell: "<status> <model> <regime> <trial>".
GLOBAL_SUMMARY="$RUN/_summary.tsv"
: > "$GLOBAL_SUMMARY"

echo "[vllm_baseline] run dir : $RUN"
echo "[vllm_baseline] n_trials: $N_TRIALS"
echo "[vllm_baseline] start   : $(date)"


# -----------------------------------------------------------------------------
# validate_bench: read bench.json and decide whether the cell produced a
# usable result. Returns 0 (ok) or 1 (fail), and prints a one-line reason.
#
# Considered valid if:
#   - bench.json exists, parses as JSON, and has a non-zero request count
#     under either schema:
#       * genai-bench (m1/m2): num_requests_valid > 0
#       * sglang.bench_serving (m3):  completed > 0
# -----------------------------------------------------------------------------
validate_bench() {
    local bench_file=$1
    if [ ! -f "$bench_file" ]; then
        echo "no bench.json"
        return 1
    fi
    python3 - "$bench_file" <<'PY' 2>/dev/null
import json, sys
try:
    b = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"bench.json unparseable: {e}")
    sys.exit(1)
reqs = b.get("num_requests_valid") or b.get("completed", 0)
errors = b.get("num_errors", 0)
if reqs > 0:
    print(f"reqs={reqs} errors={errors}")
    sys.exit(0)
print(f"reqs=0 (errors={errors})")
sys.exit(1)
PY
    return $?
}


# -----------------------------------------------------------------------------
# kill_processes_on_gpus: actively SIGKILL every CUDA-using process on the
# given GPUs. Used at the end of each cell to clean up any vLLM workers /
# engine-core / API-server children that the bench's own teardown
# (`kill -9 $SV_PID` inside run_vllm.sh) didn't catch — vLLM TP=2 spawns
# multiple worker subprocesses that become orphans of init when the parent
# is killed and can hold GPU memory long after `run_vllm.sh` returns.
#
# Querying nvidia-smi for process IDs on a specific GPU is the most reliable
# way to find them: vLLM workers run under generic process names
# (VLLM::Worker_TP0, VLLM::EngineCore, etc.) so a process-name match would
# miss them.
# -----------------------------------------------------------------------------
kill_processes_on_gpus() {
    local gpus=$1
    local gpu pid pids
    for gpu in ${gpus//,/ }; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' \r\n,')
        for pid in $pids; do
            [ -z "$pid" ] && continue
            kill -9 "$pid" 2>/dev/null || true
        done
    done
}


# -----------------------------------------------------------------------------
# wait_for_gpu_free: block until every listed GPU has at least `free_pct_min`
# percent of its total memory available, or `timeout_s` elapses. Polls
# nvidia-smi every 3 s. Acts as a safety net after kill_processes_on_gpus —
# even with all PIDs killed, the CUDA driver releases mapped pages
# asynchronously and that takes a few seconds.
#
# Why a percentage rather than an absolute MiB threshold: the gate that
# matters for vLLM is "is there enough free memory to satisfy the next
# server's --gpu-memory-utilization request". vLLM at 0.85 needs 85% free;
# we wait for >=90% to leave a small headroom for driver footprint.
#
# Args:
#   $1 gpus         : "0" or "0,1"
#   $2 free_pct_min : require memory.free / memory.total >= this percent
#                     (default 90 — vLLM at 0.85 mem-util plus headroom)
#   $3 timeout_s    : give up and return 1 after this many seconds (default 120)
# -----------------------------------------------------------------------------
wait_for_gpu_free() {
    local gpus=$1
    local free_pct_min=${2:-90}
    local timeout_s=${3:-120}

    local elapsed=0
    local min_free_pct=0
    while [ "$elapsed" -lt "$timeout_s" ]; do
        min_free_pct=100
        local gpu free total pct
        for gpu in ${gpus//,/ }; do
            read -r free total < <(nvidia-smi --query-gpu=memory.free,memory.total \
                                              --format=csv,noheader,nounits \
                                              -i "$gpu" 2>/dev/null \
                                   | tr -d ' \r' | tr ',' ' ')
            [ -z "$free" ] && free=0
            [ -z "$total" ] && total=1
            pct=$(( 100 * free / total ))
            [ "$pct" -lt "$min_free_pct" ] && min_free_pct=$pct
        done
        if [ "$min_free_pct" -ge "$free_pct_min" ]; then
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    # Last-ditch: kill any straggler PIDs again, give CUDA driver one more
    # tick, then return WARN (caller may still try to boot).
    echo "[vllm_baseline] WARN: GPUs $gpus only ${min_free_pct}% free after ${timeout_s}s; force-killing any stragglers"
    kill_processes_on_gpus "$gpus"
    sleep 5
    return 1
}


# -----------------------------------------------------------------------------
# run_one_cell: launch a single (model, regime, trial) cell. Validates the
# resulting bench.json and prints a status line. On failure, dumps a tail of
# runner.log so the operator can see the cause without leaving the main log.
# -----------------------------------------------------------------------------
run_one_cell() {
    local model_key=$1
    local hf_path=$2
    local tp=$3
    local regime=$4
    local trial=$5
    local gpus=$6        # e.g. "0" for TP=1, "0,1" for TP=2
    local port=$7

    local out_dir="$RUN/$model_key/$regime/$trial"
    mkdir -p "$out_dir"

    # Block until the previous cell's vLLM server has released GPU memory on
    # this slot's GPUs. Without this every cell-to-cell transition risks an
    # OOM-at-boot ("Free memory ... is less than desired GPU memory utilization").
    # By the time we get here the previous cell's teardown should already
    # have called kill_processes_on_gpus, so this is normally a fast no-op.
    wait_for_gpu_free "$gpus" 90 120

    echo "[vllm_baseline $(date +%H:%M:%S)] start  $model_key/$regime $trial  tp=$tp gpus=$gpus port=$port"

    MODEL="$hf_path" TP="$tp" GPU_LIST="$gpus" REGIME="$regime" \
        PORT="$port" OUT_DIR="$out_dir" \
        BOOT_TIMEOUT_S=1500 MAX_TIME_MIN=10 PHASE_DURATION_S=200 \
        bash dev/eval/main/run_vllm.sh > "$out_dir/runner.log" 2>&1
    local rc=$?

    local validation
    validation=$(validate_bench "$out_dir/bench.json")
    local valid_rc=$?

    if [ "$rc" -eq 0 ] && [ "$valid_rc" -eq 0 ]; then
        echo "[vllm_baseline $(date +%H:%M:%S)] DONE   $model_key/$regime $trial  ok  $validation"
        echo "ok	$model_key	$regime	$trial	$validation" >> "$GLOBAL_SUMMARY"
        [ -n "$PHASE_OK_FILE" ] && echo 1 >> "$PHASE_OK_FILE"
    else
        echo "[vllm_baseline $(date +%H:%M:%S)] DONE   $model_key/$regime $trial  FAIL  rc=$rc  $validation"
        echo "------ tail of $out_dir/runner.log ------"
        tail -12 "$out_dir/runner.log" 2>/dev/null | sed 's/^/    | /'
        echo "------ end tail ------"
        echo "fail	$model_key	$regime	$trial	rc=$rc $validation" >> "$GLOBAL_SUMMARY"
        [ -n "$PHASE_FAIL_FILE" ] && echo 1 >> "$PHASE_FAIL_FILE"
    fi

    # Active cleanup: SIGKILL any process still holding memory on this slot's
    # GPUs. run_vllm.sh already kills the API-server parent, but vLLM TP=2
    # workers run as separate orphan-able subprocesses, so we kill by GPU.
    # The next cell's wait_for_gpu_free is then just a short driver-release
    # sync, not a wait-for-process-death.
    kill_processes_on_gpus "$gpus"
}


# -----------------------------------------------------------------------------
# run_phase: run one model across all regimes × all trials, distributed across
# `${#WORKER_GPUS[@]}` parallel workers. Each worker is pinned to a fixed
# GPU subset (one element of WORKER_GPUS) and a unique port (one of
# WORKER_PORTS). Cells are split across workers round-robin (NR%n==slot).
#
# Args:
#   $1 model_key   : "qwen3.5" / "qwen3next" / "kimi"
#   $2 hf_path     : HF model path
#   $3 tp          : tensor-parallel size for this model
#
# Reads from caller scope:
#   WORKER_GPUS[]  : array of GPU specs, one per worker
#   WORKER_PORTS[] : array of ports, one per worker
# -----------------------------------------------------------------------------
run_phase() {
    local model_key=$1
    local hf_path=$2
    local tp=$3

    local n_workers=${#WORKER_GPUS[@]}
    local jobs_file
    jobs_file=$(mktemp)

    # Build the job list: regimes × trials, one cell per line.
    for ((trial=1; trial<=N_TRIALS; trial++)); do
        for regime in "${REGIMES[@]}"; do
            echo "$regime|trial${trial}" >> "$jobs_file"
        done
    done
    local n_jobs
    n_jobs=$(wc -l < "$jobs_file")

    # Phase-scoped ok/fail tally files. Each worker appends one line per
    # ok/fail cell; we count the lines after wait.
    PHASE_OK_FILE=$(mktemp)
    PHASE_FAIL_FILE=$(mktemp)
    : > "$PHASE_OK_FILE"
    : > "$PHASE_FAIL_FILE"

    echo
    echo "[vllm_baseline] ============================================================"
    echo "[vllm_baseline] === phase $model_key (tp=$tp)  $n_jobs cells / $n_workers workers"
    echo "[vllm_baseline] ============================================================"

    local slot
    for ((slot=0; slot<n_workers; slot++)); do
        local gpus=${WORKER_GPUS[$slot]}
        local port=${WORKER_PORTS[$slot]}
        local worker_jobs="$jobs_file.slot$slot"
        awk -v slot="$slot" -v n="$n_workers" 'NR%n==slot' "$jobs_file" > "$worker_jobs"
        local worker_n
        worker_n=$(wc -l < "$worker_jobs")

        (
            echo "[vllm_baseline/$model_key W$slot] start, $worker_n cells, gpus=$gpus, port=$port"
            local i=0
            while IFS='|' read -r regime trial; do
                [ -z "$regime" ] && continue
                i=$((i + 1))
                echo "[vllm_baseline/$model_key W$slot] $i/$worker_n: $regime $trial"
                run_one_cell "$model_key" "$hf_path" "$tp" "$regime" "$trial" "$gpus" "$port"
            done < "$worker_jobs"
            echo "[vllm_baseline/$model_key W$slot] DONE, $i/$worker_n cells, $(date)"
        ) &
    done

    wait

    local n_ok n_fail
    n_ok=$(wc -l < "$PHASE_OK_FILE")
    n_fail=$(wc -l < "$PHASE_FAIL_FILE")
    rm -f "$PHASE_OK_FILE" "$PHASE_FAIL_FILE"
    PHASE_OK_FILE=""
    PHASE_FAIL_FILE=""
    echo "[vllm_baseline] === phase $model_key done at $(date): ok=$n_ok / fail=$n_fail / total=$n_jobs ==="
}


# -----------------------------------------------------------------------------
# Phase 1 — Kimi-Linear 48B at TP=2.
# 4 workers, each on a GPU pair, ports 34000..34030. Total 8 GPUs.
# -----------------------------------------------------------------------------
WORKER_GPUS=("0,1" "2,3" "4,5" "6,7")
WORKER_PORTS=(34000 34010 34020 34030)
run_phase "kimi" "moonshotai/Kimi-Linear-48B-A3B-Instruct" 2


# -----------------------------------------------------------------------------
# Phase 2 — Qwen3-Next 80B-A3B at TP=2.
# Same 4-worker / GPU-pair layout as Kimi.
# -----------------------------------------------------------------------------
WORKER_GPUS=("0,1" "2,3" "4,5" "6,7")
WORKER_PORTS=(34000 34010 34020 34030)
run_phase "qwen3next" "Qwen/Qwen3-Next-80B-A3B-Instruct" 2


# -----------------------------------------------------------------------------
# Phase 3 — Qwen3.5-35B-A3B at TP=1.
# 8 workers, each on a single GPU, ports 34000..34070. All 8 GPUs.
# -----------------------------------------------------------------------------
WORKER_GPUS=("0" "1" "2" "3" "4" "5" "6" "7")
WORKER_PORTS=(34000 34010 34020 34030 34040 34050 34060 34070)
run_phase "qwen3.5" "Qwen/Qwen3.5-35B-A3B" 1


# -----------------------------------------------------------------------------
# Final summary: aggregate per-cell statuses across all phases.
# -----------------------------------------------------------------------------
echo
echo "[vllm_baseline] ============================================================"
echo "[vllm_baseline] ALL PHASES DONE at $(date)"
echo "[vllm_baseline] run dir : $RUN"
echo "[vllm_baseline] ------- per-cell status (from $GLOBAL_SUMMARY) -------"
total_ok=$(awk '$1=="ok"' "$GLOBAL_SUMMARY" | wc -l)
total_fail=$(awk '$1=="fail"' "$GLOBAL_SUMMARY" | wc -l)
echo "[vllm_baseline] total : ok=$total_ok / fail=$total_fail"
if [ "$total_fail" -gt 0 ]; then
    echo "[vllm_baseline] failed cells:"
    awk '$1=="fail"{printf "    %s/%s %s  (%s)\n", $2, $3, $4, $5}' "$GLOBAL_SUMMARY"
fi
echo "[vllm_baseline] aggregate via: python3 dev/eval/main/aggregate.py $RUN"

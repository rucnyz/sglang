#!/usr/bin/env bash
# parallel_gpu.sh — fork N tasks across N idle GPUs, in parallel.
#
# Usage:
#   parallel_gpu.sh <task_script> [arg1] [arg2] [arg3] ...
#
# For each `argi`, picks one idle GPU + a unique port, then runs:
#     GPU=<gpu> PORT=<port> ARG=<argi> bash <task_script>
# in parallel. Waits for all to complete; exit code = max of children.
#
# Env vars:
#   PARALLEL_GPUS  override auto-detect (e.g. PARALLEL_GPUS="1,3,5")
#                  comma-separated GPU IDs to use
#   BASE_PORT      starting port (default 31000); each task gets BASE_PORT+i
#   GPU_HEADROOM_MIB  treat a GPU as idle only if memory.used <= this
#                  (default 1024 — tolerates small kernels using tens of MB)
#   STRICT_GPUS    if set, abort if fewer idle GPUs than tasks
#                  default: if not enough GPUs, queue tasks (still parallel,
#                  but limited to available idle count)
#
# Task script contract:
#   - reads $GPU $PORT $ARG from environment
#   - launches its server, runs its workload, captures its outputs, exits
#   - DOES NOT read /tmp/sg-budget.log or any shared file (collisions!)
#   - any artifacts should go under a per-arg subdirectory

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<USAGE
Usage: parallel_gpu.sh <task_script> <arg1> [arg2] [...]

Each arg becomes one task. Auto-selects one idle GPU per task.

Env:
  PARALLEL_GPUS=0,3,5  override GPU auto-select
  BASE_PORT=31000      base port (each task uses BASE_PORT + i)
  GPU_HEADROOM_MIB=1024  GPU is idle if memory.used <= this MiB
  STRICT_GPUS=1        abort if not enough idle GPUs (default: queue)

Examples:
  parallel_gpu.sh run_arm.sh off on
  parallel_gpu.sh run_trial.sh trial1 trial2 trial3 trial4
USAGE
  exit 1
fi

TASK_SCRIPT="$1"; shift
ARGS=("$@")
NTASKS=${#ARGS[@]}
BASE_PORT="${BASE_PORT:-31000}"
GPU_HEADROOM_MIB="${GPU_HEADROOM_MIB:-1024}"

# 1. Discover idle GPUs.
declare -a GPUS
if [[ -n "${PARALLEL_GPUS:-}" ]]; then
  IFS=',' read -ra GPUS <<< "$PARALLEL_GPUS"
  echo "[parallel_gpu] using GPUs from PARALLEL_GPUS: ${GPUS[*]}"
else
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader,nounits \
      | awk -v hr="$GPU_HEADROOM_MIB" -F', ' '
          $2 <= hr && $3 == 0 { print $1 }
        '
  )
  echo "[parallel_gpu] auto-detected idle GPUs (memory.used<=${GPU_HEADROOM_MIB}MiB AND util==0): ${GPUS[*]:-(none)}"
fi

NGPUS=${#GPUS[@]}
if (( NGPUS == 0 )); then
  echo "[parallel_gpu] ERROR: no idle GPUs found" >&2
  exit 1
fi

# 2. Decide concurrency.
if (( NGPUS >= NTASKS )); then
  CONCURRENCY=$NTASKS
elif [[ -n "${STRICT_GPUS:-}" ]]; then
  echo "[parallel_gpu] ERROR: STRICT_GPUS set, need $NTASKS GPUs but only $NGPUS idle" >&2
  exit 1
else
  CONCURRENCY=$NGPUS
  echo "[parallel_gpu] WARN: $NTASKS tasks but only $NGPUS idle GPUs; will queue (concurrency=$CONCURRENCY)"
fi

# 3. Dispatch.
echo "[parallel_gpu] $NTASKS tasks → concurrency=$CONCURRENCY, base_port=$BASE_PORT"
START=$(date +%s)

declare -A PIDS
declare -A SLOT_GPU
declare -A SLOT_PORT

# Initialize slots.
for i in $(seq 0 $((CONCURRENCY - 1))); do
  SLOT_GPU[$i]="${GPUS[$i]}"
  SLOT_PORT[$i]=$((BASE_PORT + i))
done

next_arg=0
slot_finish() {
  local slot="$1"
  local pid="${PIDS[$slot]}"
  wait "$pid" 2>/dev/null || true
  local rc=$?
  if (( rc != 0 )); then
    echo "[parallel_gpu] slot=$slot pid=$pid exited rc=$rc"
  fi
  unset "PIDS[$slot]"
}

# Initial fan-out.
for slot in $(seq 0 $((CONCURRENCY - 1))); do
  if (( next_arg >= NTASKS )); then break; fi
  arg="${ARGS[$next_arg]}"
  gpu="${SLOT_GPU[$slot]}"
  port="${SLOT_PORT[$slot]}"
  echo "[parallel_gpu] launch arg=$arg → GPU=$gpu port=$port (slot=$slot)"
  GPU="$gpu" PORT="$port" ARG="$arg" bash "$TASK_SCRIPT" &
  PIDS[$slot]=$!
  next_arg=$((next_arg + 1))
done

# Drain & re-fill.
exit_max=0
while (( ${#PIDS[@]} > 0 )); do
  # Find ANY child that exits.
  for slot in "${!PIDS[@]}"; do
    pid="${PIDS[$slot]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      (( rc > exit_max )) && exit_max=$rc
      echo "[parallel_gpu] slot=$slot pid=$pid finished rc=$rc"
      unset "PIDS[$slot]"
      # Refill if more args left
      if (( next_arg < NTASKS )); then
        arg="${ARGS[$next_arg]}"
        gpu="${SLOT_GPU[$slot]}"
        port="${SLOT_PORT[$slot]}"
        echo "[parallel_gpu] refill arg=$arg → GPU=$gpu port=$port (slot=$slot)"
        GPU="$gpu" PORT="$port" ARG="$arg" bash "$TASK_SCRIPT" &
        PIDS[$slot]=$!
        next_arg=$((next_arg + 1))
      fi
    fi
  done
  sleep 1
done

DUR=$(( $(date +%s) - START ))
echo "[parallel_gpu] done; wall=${DUR}s; max_rc=$exit_max"
exit "$exit_max"

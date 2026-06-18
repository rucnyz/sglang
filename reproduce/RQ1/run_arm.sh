#!/bin/bash
# Boot one server ARM, run N agentreplay reps on the SAME server (--flush
# between reps), tear down. Token-exact forcing is overlap-compatible (the GPU
# next_token_ids override, task #334), so overlap stays ON for both arms -- the
# realistic config. GPU / PORT are env-overridable (defaults 7 / 30097);
# MEMFRAC is the only conditional override (small-pool experiments); everything
# else uses sglang's own defaults. The static mamba/KV split is sglang's default
# (mamba_full_memory_ratio); we never override it (no --max-mamba-cache-size).
# Usage:
#   run_arm.sh <base|sys> <trace> <stagger> <maxconc> <limit|-> <nreps> <outdir>
set -u
ARM=$1; TRACE=$2; STAGGER=$3; MAXCONC=$4; LIMIT=$5; NREPS=$6; OUTDIR=$7
PORT=${PORT:-30097}; GPU=${GPU:-7}; MODEL=Qwen/Qwen3.5-9B
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay
mkdir -p "$OUTDIR"

# Only non-default sglang flags we keep: --reasoning-parser (the model's own
# parser), --enable-cache-report (agentreplay needs cached_tokens), and
# --mamba-scheduler-strategy extra_buffer (the ONLY way to keep overlap ON on a
# hybrid-mamba model + radix cache; default no_buffer auto-disables overlap).
# Everything else (mem-fraction, context-length, tp, mamba ratio) is left to
# sglang's own defaults.
COMMON="--model-path $MODEL --host 127.0.0.1 --port $PORT \
  --reasoning-parser qwen3 --mamba-scheduler-strategy extra_buffer \
  --enable-cache-report --log-level info"
# MEMFRAC + RATIO are conditional (small-pool / static-best experiments).
[ -n "${MEMFRAC:-}" ] && COMMON="$COMMON --mem-fraction-static $MEMFRAC"
# Optional boot split knob (the design's own --mamba-full-memory-ratio, NOT the
# forbidden --max-mamba-cache-size). A LOW ratio shrinks the mamba pool so
# max_running binds first (mamba-bound regime, for the k2m case2 win); unset =
# sglang default 0.9. Both arms get it (fair).
[ -n "${RATIO:-}" ] && COMMON="$COMMON --mamba-full-memory-ratio $RATIO"

if [ "$ARM" = "base" ]; then
  ENVP="(none)"
  FLAGS="--radix-eviction-policy lru"
else
  # export (can't pass env via a $var in command position: bash parses inline
  # assignments before expansion, so they'd be read as a command name)
  # FULL cross-pool win config (matches reproduce/RQ1/table2 inter arm): the
  # Admitter + cross-fire is the win MECHANISM (per-arrival KV grow from idle
  # mamba); Budgeter alone (no Admitter) is overhead without the win. PF=64 is
  # the fire magnitude that delivered the documented +10.8% out_tps / -23.8%
  # p99_ttft; tick=1.0s; queue-wait + cooldown + idle-strict-check off as the
  # winning config used. Override PF/COOL via env for sweeps.
  export SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 \
         SGLANG_XPOOL_QUEUE_WAIT_US=125000 SGLANG_XPOOL_COOLDOWN_S="${COOL:-1.0}" \
         SGLANG_HIMA_LOG="$OUTDIR/budgeter.jsonl"
  # Calibrated single-curve cost model (dev/eval/cost_model/kappa_fit.json,
  # Qwen3.5-9B/H200): c_M=0 (mamba recompute folded into c_KV). Without this the
  # builtin 35B default has non-zero c_M, which drives wrong-direction k2m fires
  # that shrink the floor-less KV pool to underflow -> CUDA illegal-access crash.
  export SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 \
         SGLANG_CSIGMA_KV_BETA=0.024570739655696554 \
         SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 \
         SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  # Physical fire: cuMemUnmap source chunks + cuMemMap recycled handles to
  # destination. Zero extra GPU memory (handles are conserved). Works at any
  # MEMFRAC.
  # Eviction policy is the SECOND ablation axis (PLAN: prove inter-layer with
  # naive LRU first, then LPB for the further intra-pool gain). Default lpb;
  # EVICT=lru runs the inter-only arm (cross-pool on, naive eviction).
  ENVP="SGLANG_HIMA=1 evict=${EVICT:-lpb} (+tick + calibrated csigma c_M=0, exported)"
  FLAGS="--radix-eviction-policy ${EVICT:-lpb}"
fi

echo "[$ARM] boot: flags=$FLAGS env=$ENVP"
# SIGKILL any prior server by PORT (robust: cmdline-pattern pkill misses GPU
# procs in D-state), then wait for the port to actually release before binding.
PRIORPID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$PRIORPID" ] && kill -9 "$PRIORPID" 2>/dev/null
pkill -9 -f "sglang.launch_server.*$PORT" 2>/dev/null
for i in $(seq 1 40); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " || break
  sleep 1
done
sleep 2

CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV -m sglang.launch_server $COMMON $FLAGS > "$OUTDIR/server_${ARM}.log" 2>&1 &
SVPID=$!

ready=0
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; echo "[$ARM] ready after ~$((i*5))s"; break; fi
  if ! kill -0 $SVPID 2>/dev/null; then echo "[$ARM] SERVER DIED"; tail -25 "$OUTDIR/server_${ARM}.log"; exit 1; fi
  sleep 5
done
[ "$ready" = "1" ] || { echo "[$ARM] BOOT TIMEOUT"; tail -25 "$OUTDIR/server_${ARM}.log"; kill -9 $SVPID 2>/dev/null; exit 2; }

LIMARG=""; [ "$LIMIT" != "-" ] && LIMARG="--limit $LIMIT"
# STAGGER=- omits --stagger so root arrival uses the trace's own absolute t
# (start_t - min_s), i.e. the recorded timeline drives ordering. case3's
# temporal A->B flip needs this; a fixed uniform stagger would erase it.
STAGARG=""; [ "$STAGGER" != "-" ] && STAGARG="--stagger $STAGGER"
for rep in $(seq 1 "$NREPS"); do
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay \
    --trace "$TRACE" $STAGARG \
    --max-concurrency "$MAXCONC" $LIMARG --flush \
    --url "http://127.0.0.1:$PORT/generate" --label "${ARM}_r${rep}" \
    --out "$OUTDIR/${ARM}_r${rep}.json" > "$OUTDIR/${ARM}_r${rep}.log" 2>&1
  echo "[$ARM] rep $rep: $(grep -oE '\"cache_hit\": [0-9.]+|\"throughput_tok_s\": [0-9.]+' "$OUTDIR/${ARM}_r${rep}.json" | tr '\n' ' ')"
done

kill -9 $SVPID 2>/dev/null
echo "[$ARM] DONE -> $OUTDIR"

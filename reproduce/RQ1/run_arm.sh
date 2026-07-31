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
PORT=${PORT:-30097}; GPU=${GPU:-7}; MODEL=${MODEL:-Qwen/Qwen3.5-9B}
# Multi-GPU: GPUS is the CUDA_VISIBLE_DEVICES list (default = single $GPU); TP is
# tensor-parallel size (default 1). For a big model, e.g. GPUS=0,1,2,3 TP=4.
GPUS=${GPUS:-$GPU}; TP=${TP:-1}; GPU0=${GPUS%%,*}
VENV=${VENV:-/scratch/yuzhou/projects/sglang/.venv/bin/python}
AR=/scratch/yuzhou/projects/agentreplay
mkdir -p "$OUTDIR"

# Only non-default sglang flags we keep: --reasoning-parser (the model's own
# parser), --enable-cache-report (agentreplay needs cached_tokens), and
# --mamba-scheduler-strategy extra_buffer (the ONLY way to keep overlap ON on a
# hybrid-mamba model + radix cache; default no_buffer auto-disables overlap).
# Everything else (mem-fraction, context-length, tp, mamba ratio) is left to
# sglang's own defaults.
COMMON="--model-path $MODEL --host 127.0.0.1 --port $PORT \
  --enable-cache-report --log-level info --trust-remote-code"
[ "$TP" -gt 1 ] 2>/dev/null && COMMON="$COMMON --tp $TP"
# mamba-scheduler-strategy extra_buffer keeps overlap ON on hybrid-MAMBA models
# (it builds a conv-state track buffer). Kimi-Linear (linear-attn, no mamba conv
# state) has conv_states_shape=None -> extra_buffer crashes; MAMBA_STRAT=no_buffer
# skips it (overlap auto-off, correctness intact). Default extra_buffer for mamba.
[ "${MAMBA_STRAT:-extra_buffer}" != "none" ] && COMMON="$COMMON --mamba-scheduler-strategy ${MAMBA_STRAT:-extra_buffer}"
# reasoning-parser is model-specific (qwen3's </think> token); non-Qwen models
# (e.g. Kimi-Linear) tokenize it differently and the reasoner grammar backend
# fails-fast. It only affects output PARSING, which token-exact forced replay
# does not use, so REASONING=none skips it. Default qwen3 for the Qwen models.
[ "${REASONING:-qwen3}" != "none" ] && COMMON="$COMMON --reasoning-parser ${REASONING:-qwen3}"
# MEMFRAC + RATIO are conditional (small-pool / static-best experiments).
[ -n "${MEMFRAC:-}" ] && COMMON="$COMMON --mem-fraction-static $MEMFRAC"
# Optional boot split knob (the design's own --mamba-full-memory-ratio, NOT the
# forbidden --max-mamba-cache-size). A LOW ratio shrinks the mamba pool so
# max_running binds first (mamba-bound regime, for the k2m case2 win); unset =
# sglang default 0.9. Both arms get it (fair).
[ -n "${RATIO:-}" ] && COMMON="$COMMON --mamba-full-memory-ratio $RATIO"
[ -n "${MAMBA_CAP:-}" ] && COMMON="$COMMON --max-mamba-cache-size $MAMBA_CAP"
# MAX_RUNNING pins the server admission ceiling (both arms, fair). The m2k
# KV-bound win needs it set at the GPU compute knee so base is KV-limited BELOW
# it (headroom) and sys grows KV up to it; unset = sglang default (mamba//ratio).
[ -n "${MAX_RUNNING:-}" ] && COMMON="$COMMON --max-running-requests $MAX_RUNNING"
# CUDA graph backend passthrough (both arms, fair). Nemotron-3 defaults to the
# torch.compile piecewise decode graph, which hits a torch 2.9 meta_mm() signature
# bug; CUDA_GRAPH_DECODE=full uses the standard captured graph (fast, no compile
# path). Prefill graph is disabled by default upstream; override if needed.
[ -n "${CUDA_GRAPH_DECODE:-}" ] && COMMON="$COMMON --cuda-graph-backend-decode $CUDA_GRAPH_DECODE"
[ -n "${CUDA_GRAPH_PREFILL:-}" ] && COMMON="$COMMON --cuda-graph-backend-prefill $CUDA_GRAPH_PREFILL"

if [ "$ARM" = "base" ]; then
  ENVP="(none)"
  FLAGS="--radix-eviction-policy lru"
else
  # export (can't pass env via a $var in command position: bash parses inline
  # assignments before expansion, so they'd be read as a command name)
  # FULL cross-pool win config (matches reproduce/RQ1/table2 inter arm): the
  # Admitter + cross-fire is the win MECHANISM (per-arrival KV grow from idle
  # mamba); Budgeter alone (no Admitter) is overhead without the win. PF=64 is
  # tick=1.0s; queue-wait=default (100us, not 125000 which causes premature
  # fires and oscillation on case1 workloads); cooldown overridable via COOL.
  # QW override: set QW=<value> env to override QUEUE_WAIT_US for sweeps.
  export SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 \
         SGLANG_XPOOL_QUEUE_WAIT_US="${QW:-100}" SGLANG_XPOOL_COOLDOWN_S="${COOL:-1.0}" \
         SGLANG_HIMA_LOG="$OUTDIR/budgeter.jsonl"
  # Calibrated single-curve cost model (dev/eval/cost_model/kappa_fit.json,
  # Qwen3.5-9B/H200): c_M=0 (mamba recompute folded into c_KV). Without this the
  # builtin 35B default has non-zero c_M, which drives wrong-direction k2m fires
  # that shrink the floor-less KV pool to underflow -> CUDA illegal-access crash.
  # Per-model calibrated single-curve cost model (dev/eval/cost_model/calibrate.sh;
  # c_M=0 for both: mamba recompute folds into c_KV). The c_KV curve is
  # model-specific: the 9B curve mis-prices 35B so the PaybackPlanner flips fire
  # direction every tick (k2m<->m2k oscillation -> OOM), so 35B uses its OWN
  # calibration (kappa_fit qwen3.5-35b).
  if [[ "$MODEL" == *Ling* ]]; then
    export SGLANG_CSIGMA_KV_ALPHA=9.997656e-08 \
           SGLANG_CSIGMA_KV_BETA=5.375527e-03 \
           SGLANG_CSIGMA_KV_GAMMA=2.216512e+02 \
           SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  elif [[ "$MODEL" == *35B* ]]; then
    export SGLANG_CSIGMA_KV_ALPHA=1.306635e-07 \
           SGLANG_CSIGMA_KV_BETA=1.601545e-02 \
           SGLANG_CSIGMA_KV_GAMMA=2.420041e+01 \
           SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  elif [[ "$MODEL" == *Nemotron* ]]; then
    # Nemotron-3-Super-120B-A12B / H200 (calibrate.sh, --disable-cuda-graph
    # + --tp-size 4 --max-mamba-cache-size 16; c_M=0 hybrid single-curve).
    export SGLANG_CSIGMA_KV_ALPHA=4.869673e-08 \
           SGLANG_CSIGMA_KV_BETA=4.003409e-02 \
           SGLANG_CSIGMA_KV_GAMMA=7.036574e+01 \
           SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  else
    export SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 \
           SGLANG_CSIGMA_KV_BETA=0.024570739655696554 \
           SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 \
           SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  fi
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
# Port release lags the 100+ GB KV/mamba pool teardown by seconds: the prior
# server closes its socket but its CUDA context frees device memory async.
# Booting the next arm into a half-freed GPU races that teardown -> boot-time
# contention that drops a request (seen as a 1-error, ~5%-slow outlier rep and
# a widened N=3 std). Wait until the prior server's device memory is actually
# released (free recovers toward the ~140 GB empty-H200 baseline) before boot.
for i in $(seq 1 60); do
  FREE=$(nvidia-smi -i "$GPU0" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "${FREE:-0}" -gt 130000 ] 2>/dev/null && break
  sleep 2
done

CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV -m sglang.launch_server $COMMON $FLAGS > "$OUTDIR/server_${ARM}.log" 2>&1 &
SVPID=$!

ready=0
for i in $(seq 1 ${BOOT_TRIES:-200}); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; echo "[$ARM] ready after ~$((i*5))s"; break; fi
  if ! kill -0 $SVPID 2>/dev/null; then echo "[$ARM] SERVER DIED"; tail -25 "$OUTDIR/server_${ARM}.log"; exit 1; fi
  sleep 5
done
[ "$ready" = "1" ] || { echo "[$ARM] BOOT TIMEOUT"; tail -25 "$OUTDIR/server_${ARM}.log"; kill $SVPID 2>/dev/null; for _ in $(seq 1 30); do kill -0 $SVPID 2>/dev/null || break; sleep 2; done; kill -9 $SVPID 2>/dev/null; exit 2; }

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

# Graceful teardown: SIGTERM drains CUDA contexts cleanly; kill -9 on a live
# CUDA process poisons the driver for the next boot (unkillable R-zombies).
kill $SVPID 2>/dev/null
for _ in $(seq 1 30); do kill -0 $SVPID 2>/dev/null || break; sleep 2; done
kill -9 $SVPID 2>/dev/null
echo "[$ARM] DONE -> $OUTDIR"

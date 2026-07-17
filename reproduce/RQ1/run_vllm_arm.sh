#!/bin/bash
# RQ1 vLLM baseline arm: boot stock vLLM, replay the SAME token-exact agentreplay
# trace the sglang arms use, tear down. Mirrors run_arm.sh so the vLLM column of
# the main cross-model table is reproducible with one command.
#
# NOTE on token-forcing: vLLM has no `custom_params.forced_output_ids` (that is
# sglang's GPU-override extension). replay-vllm falls back to free generation
# capped at max_tokens=len(forced) + ignore_eos, so len_match is ~0.3 (not
# token-exact). The vLLM column is therefore a best-effort length-target
# comparison; the sglang base/sys arms are token-exact (len_match 1.0).
#
# Usage:
#   run_vllm_arm.sh <trace> <stagger> <maxconc> <limit|-> <outdir>
# Env: GPU (default 7), PORT (default 30098), MODEL (default Qwen/Qwen3.5-9B),
#      MAXLEN (default 40000), MEMFRAC (default 0.90), MAXSEQS (default 256).
set -u
TRACE=$1; STAGGER=$2; MAXCONC=$3; LIMIT=$4; OUTDIR=$5
GPU=${GPU:-7}; PORT=${PORT:-30098}; MODEL=${MODEL:-Qwen/Qwen3.5-9B}
GPUS=${GPUS:-$GPU}; TP=${TP:-1}
MAXLEN=${MAXLEN:-40000}; MEMFRAC=${MEMFRAC:-0.90}; MAXSEQS=${MAXSEQS:-256}
VP=${VLLM_BIN:-/scratch/yuzhou/projects/vllm-baseline/.venv/bin}
AR=/scratch/yuzhou/projects/agentreplay
mkdir -p "$OUTDIR"

# SIGKILL any prior binder on PORT (kill vLLM's EngineCore child too: it is a
# separate process that outlives a parent kill and holds all the GPU memory).
PRIORPID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$PRIORPID" ] && kill -9 "$PRIORPID" 2>/dev/null
pkill -9 -f "vllm serve.*$PORT" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
sleep 2

# --served-model-name pins the id replay-vllm requests (else vLLM serves under the
# resolved snapshot path and every request 404s).
TPARG=""; [ "$TP" -gt 1 ] 2>/dev/null && TPARG="--tensor-parallel-size $TP"
CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VP/vllm serve "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port $PORT --max-num-seqs $MAXSEQS $TPARG \
  --max-model-len $MAXLEN --gpu-memory-utilization $MEMFRAC \
  --trust-remote-code > "$OUTDIR/server_vllm.log" 2>&1 &
SVPID=$!

ready=0
for i in $(seq 1 240); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  if [ "$code" = "200" ]; then ready=1; echo "[vllm] ready after ~$((i*5))s"; break; fi
  if ! kill -0 $SVPID 2>/dev/null; then echo "[vllm] SERVER DIED"; tail -30 "$OUTDIR/server_vllm.log"; exit 1; fi
  sleep 5
done
[ "$ready" = "1" ] || { echo "[vllm] BOOT TIMEOUT"; tail -30 "$OUTDIR/server_vllm.log"; kill -9 $SVPID 2>/dev/null; exit 2; }

LIMARG=""; [ "$LIMIT" != "-" ] && LIMARG="--limit $LIMIT"
STAGARG=""; [ "$STAGGER" != "-" ] && STAGARG="--stagger $STAGGER"
TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VP/python -m agentreplay replay-vllm \
  --trace "$TRACE" --model "$MODEL" $STAGARG \
  --max-concurrency "$MAXCONC" $LIMARG \
  --url "http://127.0.0.1:$PORT/v1/completions" --label vllm_r1 \
  --out "$OUTDIR/vllm_r1.json" > "$OUTDIR/vllm_r1.log" 2>&1
echo "[vllm] $(grep -oE '\"throughput_tok_s\": [0-9.]+|\"len_match_rate\": [0-9.]+|\"n_ok\": [0-9]+' "$OUTDIR/vllm_r1.json" 2>/dev/null | tr '\n' ' ')"

kill -9 $SVPID 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "[vllm] DONE -> $OUTDIR"

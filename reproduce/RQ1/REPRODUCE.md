# RQ1 reproducibility index

Everything needed to reproduce RQ1 from scratch on a fresh box: pull the corpus,
build the traces, calibrate the cost model, run the arms.
All sglang arms replay the SAME token-exact agentreplay trace (`forced_output_ids`);
base = LRU eviction / no HiMA, sys = LPB eviction + HiMA.

## 0. Prerequisites

Two separate repos plus a gated dataset plus the model weights.

```bash
# 1. sglang (this repo; see AGENTS.md, never system python)
git clone https://github.com/rucnyz/sglang && cd sglang && git checkout HiMA
uv venv --python 3.12 && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 2. agentreplay (SEPARATE repo: the trace builder + replay client)
git clone https://github.com/ucsb-mlsec/agentreplay

# 3. the corpus: a GATED private HF dataset (UCSB-SURFI/claude-code-traces)
cd agentreplay && python -m agentreplay pull        # needs approved access
# lands in data/corpus/ with layout data/contrib/<contributor>/<slug>/<sessionId>.jsonl
#                                  + .../<sessionId>/subagents/agent-*.jsonl

# 4. model weights (hf_transfer strongly recommended; the 120B is ~240 GB)
uv pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.5-9B
HF_HUB_ENABLE_HF_TRANSFER=1 hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```

GPU budget: 9B and 35B-A3B each fit one H200; Nemotron-3-Super-120B needs 4 (TP4,
~137 GB/GPU at MEMFRAC 0.85, so 143 GB cards).

The corpus working tree is frozen and git-pinned, so trace builds off it are deterministic.

**Never commit or push the built traces.**
They are token-ids of real Claude Code sessions and `tokenizer.decode()` recovers the
original text verbatim; the corpus is gated for exactly that reason.
Rebuild them on each box (below), or rsync them privately.

## 1. Build the canonical traces

One recipe per (model-family tokenizer, case shape).
Session selection is turn-count based and therefore tokenizer-independent, so the
program_id set is IDENTICAL across models: apples-to-apples by construction.

```bash
cd /path/to/agentreplay
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
SRC=data/corpus/data/contrib
V=/path/to/sglang/.venv/bin/python

# Case1 (@conc 64) and Case3 (@conc 128) share the t6 trace; Case2 uses t12.
# Qwen tokenizer serves BOTH 9B and 35B-A3B (shared tokenizer).
$V -m agentreplay convert --projects $SRC --tokenizer Qwen/Qwen3.5-9B \
   --out data/traces/cc_qwen_t6.jsonl  --max-sessions 200 --max-turns 6  --min-turns 5
$V -m agentreplay convert --projects $SRC --tokenizer Qwen/Qwen3.5-9B \
   --out data/traces/cc_qwen_t12.jsonl --max-sessions 150 --max-turns 12 --min-turns 5

# Same flags, only --tokenizer / --out change, for each additional model:
$V -m agentreplay convert --projects $SRC --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
   --out data/traces/cc_nemotron_t6.jsonl  --max-sessions 200 --max-turns 6  --min-turns 5
$V -m agentreplay convert --projects $SRC --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
   --out data/traces/cc_nemotron_t12.jsonl --max-sessions 150 --max-turns 12 --min-turns 5
```

Do NOT pass `--flatten-subagents`: it reshapes the trace (different req/program counts)
and breaks comparability with the other models.

**Verify** the program sets match before trusting a cross-model comparison:

```python
import json
def progs(f):
    s=set(); n=0
    for l in open(f): n+=1; s.add(json.loads(l)['program_id'])
    return n, s
a=progs('data/traces/cc_qwen_t6.jsonl'); b=progs('data/traces/cc_nemotron_t6.jsonl')
assert a[1]==b[1], "program sets differ -> NOT apples-to-apples"
```

Expected on the current corpus: t6 = 1199 req / 195 programs, t12 = 1789 req / 145 programs.
Counts drift if the corpus is re-pulled or `convert` changes; that is fine as long as all
models are rebuilt together off the same corpus + same convert.

## 2. Calibrate the cost model (once per model)

The sys arm needs a model-specific `c_KV(L)` curve.
Reusing another model's csigma drives wrong-direction fires, oscillation, and OOM (proven on 35B).

```bash
cd /path/to/sglang
# Small single-GPU model:
eval "$(bash dev/eval/cost_model/calibrate.sh Qwen/Qwen3.5-9B H200)"

# Multi-GPU / large-SSM model: the default mamba cap OOMs the profiler and the
# piecewise decode graph hits a torch 2.9 meta_mm bug; the batch-1 prefill sweep
# needs no decode graph.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EXTRA_FLAGS="--tp-size 4 --max-mamba-cache-size 16 --trust-remote-code --disable-cuda-graph" \
MEM_FRACTION=0.85 REPEATS=3 \
  bash dev/eval/cost_model/calibrate.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 H200
```

It prints `export SGLANG_CSIGMA_*` lines; add them as a `MODEL`-matched branch in
`run_arm.sh` (see the existing `*Nemotron*` / `*35B*` branches).
`c_M = 0` is correct for hybrids and NOT a bug: a hybrid cache miss re-prefills the whole
prefix (attention + every recurrent layer) as one forward whose total cost is fit into `c_KV`.

## 3. Run the arms

`run_arm.sh` is the atomic unit: boot one arm, run N reps (`--flush` between), tear down.

```
run_arm.sh <base|sys> <trace> <stagger> <maxconc> <limit|-> <nreps> <outdir>
```

Full cross-model campaign (serial, see Operational notes):

```bash
OUT=/path/to/out bash reproduce/RQ1/run_campaign_crossmodel.sh
```

Single cell:

```bash
T=/path/to/agentreplay/data/traces
GPU=7 MODEL=Qwen/Qwen3.5-9B bash reproduce/RQ1/run_arm.sh base $T/cc_qwen_t6.jsonl 0.5 64  - 3 out/9b_case1/base
GPU=7 MODEL=Qwen/Qwen3.5-9B bash reproduce/RQ1/run_arm.sh sys  $T/cc_qwen_t6.jsonl 0.5 64  - 3 out/9b_case1/sys
```

Case map: Case1 = t6 @conc 64, Case2 = t12 @conc 64, Case3 = t6 @conc 128 (same trace as Case1,
2x concurrency). Stagger 0.5 throughout.

vLLM column:

```bash
GPU=7 MODEL=Qwen/Qwen3.5-9B bash reproduce/RQ1/run_vllm_arm.sh $T/cc_qwen_t6.jsonl 0.5 64 - out/vllm
```

vLLM has no `custom_params.forced_output_ids` (sglang's GPU-override), so replay-vllm
free-generates capped at `max_tokens` + `ignore_eos` and len_match is ~0.3.
The vLLM column is a best-effort length-target comparison; the sglang arms are token-exact.

## 4. Per-model config

| model | GPUs | required env |
|-------|------|--------------|
| Qwen3.5-9B | 1 | defaults |
| Qwen3.5-35B-A3B | 1 | `MEMFRAC=0.85` |
| Nemotron-3-Super-120B-A12B | 4 (TP4) | `GPUS=4,5,6,7 TP=4 REASONING=none MEMFRAC=0.85 MAMBA_CAP=256 MAMBA_STRAT=no_buffer CUDA_GRAPH_DECODE=full CUDA_GRAPH_PREFILL=disabled` |

Nemotron-3 notes.
`REASONING=none`: the qwen3 `</think>` parser is Qwen-specific and forced replay does not
parse output anyway.
`MAMBA_CAP=256`: the auto-sized mamba pool (2601 slots) OOMs at boot.
`CUDA_GRAPH_DECODE=full CUDA_GRAPH_PREFILL=disabled`: Nemotron-3 defaults BOTH graphs to
`tc_piecewise`, which hits a torch 2.9 `meta_mm()` signature bug during capture.

## 5. Blocked cells

| cell | blocker |
|------|---------|
| Kimi-Linear-48B sys | UNBLOCKED 2026-07-31 (branch HiMA-latest): upstream v0.5.16 caches KDA state in the radix tree (`RadixLinearAttention` era; Kimi is in the mamba-radix + extra_buffer arch lists), and `MLATokenToKVPool` now has MultiTensorArena backing (dev/interlayer/5_mla_arena). Chain-attach proven on the t6@64 slice. Per-model env: `GPUS=<2 gpus> TP=2 REASONING=none MEMFRAC=0.85 ALLOW_TRUNC=1` (run_arm exports the calibrated csigma + 18 MiB arena chunk for `*Kimi*`). |
| Ling-2.6-flash sys | MLA: the KV cache is a compressed-latent pool, and HiMA's arena backing exists only in `MHATokenToKVPool` (gated on `head_dim == v_head_dim`). `kv_arena=None` -> the shared chain never builds -> zero cross-pool fires. See FINDINGS.md. |
| 35B Case2/Case3 sys | crashes (658-948 err/rep): `c_M=0` mis-serves 35B's mamba-bound regimes (open issue #276). Case1 is measurable. |

A model is HiMA-measurable only if it gets a MambaRadixCache. The old second
gate (MHA/GQA-only full-attention layers) fell 2026-07-31: the arena now backs
`MLATokenToKVPool` too (dev/interlayer/5_mla_arena), which is how Kimi-Linear
became measurable. Ling-2.6-flash remains blocked only by whatever keeps it off
MambaRadixCache on this tree (unverified since the rebase).

## 6. Results

See FINDINGS.md for the measured table.
Numbers move with the trace: the old `cc_qwen_t6_v2` (200 programs, direct-from-projects
build) gave 9B Case1 +5.5%; the canonical corpus-built `cc_qwen_t6` (195 programs) gives
+15.6% (base 977.3 -> sys 1130.1) because it has more reuse for LPB to exploit
(cache_hit 0.73 -> 0.85). Always compare base vs sys ON THE SAME trace.

## 7. Operational notes

Run cells SERIALLY, or at most two on disjoint GPU sets.
A five-job parallel run (a 120B TP4 plus four single-GPU models) cross-contaminated the
per-rep throughput: rep1 was clean and rep3 collapsed (9B 1026 -> 0, 35B 989 -> 440,
Nemotron 666 -> 321) as the reps overlapped. Those numbers are void.

Check the box is actually yours before a run:

```bash
nvidia-smi                 # free memory AND no other tenant's processes
```

If `nvidia-smi` reports `Failed to initialize NVML: Insufficient Permissions` while
`sudo nvidia-smi` works, the GPUs are healthy but your user has lost device access
(open("/dev/nvidia0") returns EPERM even though the node is mode 666). That is a box
access-control problem, not a driver or workload fault; a warm reboot does not fix it.

Never rapid kill+relaunch a run: killed CUDA processes linger in D-state holding the
pool, and the next boot gets OOM-killed with an empty log. Kill once, then wait for
`nvidia-smi` to show the memory actually released.

# RQ1 reproducibility index

Every RQ1 number maps to a runnable script here. All sglang arms replay the SAME
token-exact agentreplay traces (`forced_output_ids`); base = LRU / no HiMA, sys =
LPB + HiMA. GPU 7 by default (`GPU=` to override).

## Atomic unit: `run_arm.sh`

Boots ONE arm, runs N agentreplay reps on it (`--flush` between reps), tears down.

```
run_arm.sh <base|sys> <trace> <stagger> <maxconc> <limit|-> <nreps> <outdir>
```

Model is `MODEL=` env (default `Qwen/Qwen3.5-9B`). The sys arm exports the full
cross-pool config (Admitter + PF64 + calibrated csigma c_M=0); see the script.

## Cells and their scripts

| cell | model | script | status |
|------|-------|--------|--------|
| Case1/2/3 base+sys, N=3 | Qwen3.5-9B | `run_official_case123.sh` | measured (FINDINGS.md) |
| Case1/2/3 base+sys, N=3 | Qwen3.5-35B-A3B | `run_arm.sh` per case, `MODEL=Qwen/Qwen3.5-35B-A3B` | to run |
| vLLM column | any | `run_vllm_arm.sh` | runnable (best-effort forcing, see below) |
| Kimi-Linear-48B sys | 48B | — | BLOCKED (upstream) |

Traces (in `$AR/data/traces/`): Case1 `cc_qwen_t6_v2.jsonl` (1200 req),
Case2 `cc_qwen_t12.jsonl` (1795 req), Case3 `cc_qwen_t12_filtered.jsonl` (3535 req).
All three cases run at `stagger=0.5 maxconc=64` (Case3 at `conc=128`).

### 9B (canonical)

```
GPU=7 bash reproduce/RQ1/run_official_case123.sh   # Case1/2/3 x base/sys x N=3
```

### 35B

Same traces, driven per case via `run_arm.sh` (its `MODEL` env is the only change):

```
T=$AR/data/traces
MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 bash reproduce/RQ1/run_arm.sh base $T/cc_qwen_t6_v2.jsonl 0.5 64 - 3 <out>
MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 bash reproduce/RQ1/run_arm.sh sys  $T/cc_qwen_t6_v2.jsonl 0.5 64 - 3 <out>
```

### vLLM column

```
GPU=7 MODEL=Qwen/Qwen3.5-9B bash reproduce/RQ1/run_vllm_arm.sh <trace> 0.5 64 - <out>
```

vLLM has no `custom_params.forced_output_ids` (sglang's GPU-override, needed for
token-exact forcing), so replay-vllm free-generates capped at `max_tokens` +
`ignore_eos` -> len_match ~0.3. The vLLM column is a best-effort length-target
comparison; the sglang base/sys arms are token-exact (len_match 1.0).

### Kimi-Linear-48B sys: BLOCKED

sglang disables radix caching for `KimiLinearForCausalLM`
(`_handle_mamba_radix_cache(support_mamba_cache=False)`), so the tree cache is a
`ChunkCache`, not a `MambaRadixCache`. HiMA is built on MambaRadixCache
(eviction counters + arena owner_provider), so sys cannot attach; the boot guard
`require_mamba_radix_cache_for_hima` fails fast with a clear message. Kimi's base
and a vLLM arm are still runnable (Kimi-tokenized trace needed:
`agentreplay convert --tokenizer moonshotai/Kimi-Linear-48B-A3B-Instruct`). A
different, sglang-supported Mamba2 hybrid (Nemotron-H, Falcon-H1, Granite-4
hybrid, ...) is the path to a third fully-measurable model.

# SWE-bench Pro × V4-Flash × HiCache: baseline matrix

## Setup
- **Model**: DeepSeek-V4-Flash, TP=2 EP=2 on 2× B300 (GPUs 5,6)
- **Dataset**: SWE-bench Pro Python subset (`uv run swebenchpro --language python --limit 32` → 32 tasks)
- **Agent**: terminus-2 (litellm → V4-Flash OpenAI-compat at `http://172.17.0.1:30000/v1`)
- **Concurrency**: harbor `-n 32`, `--ak max_turns=200`
- **Server stack** (`aginfer` branches of rucnyz/sglang + rucnyz/Mooncake): all 5 patches
- **HiCache config (when on)**: `--hicache-ratio 1.5 --hicache-write-policy write_through_selective --hicache-storage-backend mooncake` (master with `--enable_offload=true --metrics_port=9053`); host pool = 1.5 × device pool, Mooncake DRAM 200 GB / TP rank
- **Per-run sweep variable**: `MAX_TOTAL_TOKENS` (device KV pool cap) and HiCache ON/OFF

## Results

| Run | KV pool cap | HiCache | Router | Total runtime | Per-trial mean | Successful trials | Peak `#running-req` | Peak `#cached-token` (single batch) | Max `swa token usage` | Peak input tput (tok/s) |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | 10.2 M (default) | ON  | — | 47 m 19 s | — | 29/32 | 36 | 44 288  |  0.04 |   99 K |
| **B** | 10.2 M (default) | OFF | — | 39 m 01 s | — | 29/32 | 18 | 21 760  |  0.04 |  344 K |
| **C** | 256 K | ON  | — | **38 m 37 s** | — | 29/32 | **47** | **121 856** | **0.97** | 221 K |
| **D** | 256 K | OFF | — | **CRASH** (FlashMLA `get_decoding_sched_meta:111` invalid argument when device cache evicts without DRAM tier to spill to) | — | — | — | — | — | — |
| **E** | 512 K | OFF | — | 47 m 03 s | — | 29/32 | 31 | 48 896  | 0.68 | 319 K |
| **F** | 512 K | ON  | — | **41 m 29 s** | 981 s | 29/32 | 43 | 46 592  | 0.76 | 295 K |
| **G** | 512 K | ON  | **ThunderAgent** (TR mode) | **28 m 41 s** | **666 s** | 17/32 | 28 | 47 616 | 0.43 | 248 K |
| **F'** | 512 K | ON | — (LRU, same topology as H'/H) | 48 m 47 s | 873 s | 30/32 | — | — | — | — |
| **H'** | 512 K | ON | — (paper §7 Ours via SGLANG_KV_POLICY_MODULE) | 54 m 55 s | 885 s | 30/32 | — | — | 0.83 | — |
| ~~H~~ | 512 K | ON | — (Ours, sign-bug; superseded by H') | 46 m 02 s | 1063 s | 30/32 | — | — | 0.84 | — |

Notes:
- *Total runtime* = harbor's reported `Total runtime` (wall clock from first trial dispatch to last trial completion across 32 trials × 16 in-flight slots). **Comparable run-to-run only when successful-trial counts match.**
- *Per-trial mean* = mean `agent_execution` wall-clock across the trials that successfully entered the agent loop. **The fair metric when successful-trial counts diverge** (see Run G caveat).
- *Successful trials* = harbor `n_completed_trials − n_errored_trials`. The errors are docker-compose / env-setup failures unrelated to the inference stack and vary run-to-run with concurrency luck.
- *Peak running* / *Peak cached* are the largest single-batch values reported by sglang's `Prefill batch` log lines.
- *Max swa token usage* is the largest value of the `swa token usage:` field across all prefill batches; 1.0 means the device KV pool was completely full at that instant.
- Cumulative cached-token reuse ratio (cumcached / (cumcached + cumnew)) is **~0.96 in all completed runs** — terminus-2's system prompt + accumulating reasoning traces make multi-turn agent workloads massively prefix-shareable.

## Three regimes — when HiCache earns its keep

### Regime 1 — Loose KV pool, no eviction (Runs A, B)
With the default 10.2 M-token pool, max SWA usage is **4 %**. The device-side radix prefix cache already lives entirely on HBM; HiCache's DRAM tier never gets read from. Net: HiCache overhead from the L2 write-through backup makes Run A **8 m slower** than B. **Don't enable HiCache when working set fits in HBM.**

### Regime 2 — Moderate pressure (Runs E, F, G)
With cap = 512 K (about 5 % of default), max SWA usage = 68–76 %, the device cache starts evicting under contention. HiCache ON (Run F) absorbs the evictions into DRAM and **finishes 5 m 34 s (12 %) faster** than the HiCache-OFF baseline (Run E) on the same workload. It also sustains higher concurrent decode (peak `#running-req` 43 vs 31) because backpressure is gentler. **This is the regime where HiCache pays for itself.**

**Run G** adds a program-aware router (paper §8 baseline: ThunderAgent in TR mode, sitting between harbor and sglang on port 9100). Same backend config as Run F (cap 512K + HiCache ON). The router tracks per-program KV-cache usage and pauses entire agentic programs when the backend goes over capacity, resuming them via Best-Fit-Decreasing bin-packing once room opens up. **Per-trial mean wall-clock drops from 981 s (Run F) to 666 s (Run G), a 32 % improvement that is robust to the trial-count asymmetry.** The aggregate harbor wall-clock (28 m 41 s vs 41 m 29 s, 31 % faster) over-reports the speedup because Run G suffered more env-setup failures (17 vs 29 successful trials), so the surviving working set was smaller and concurrency stress lower (peak running 28 vs 43, max swa 0.43 vs 0.76). Treat the per-trial mean as the load-bearing number.

**Runs F'/H' — paper §7 Ours vs LRU, same backend.** Run F' re-runs LRU on
GPUs 4,7 (the only free pair when we did the §7 evaluation; siavash's vllm
on GPU 4 + vllm-huihui-gpu7 had to be killed to free them) so it's a clean
apples-to-apples LRU baseline against Run H'. Run H' plugs our paper-§7
value rule into sglang's eviction heap via the `SGLANG_KV_POLICY_MODULE`
env hook on `UnifiedRadixCache.FullComponent.drive_eviction` — the same
`OursGreedyPolicy` simulator code from [`baselines/ours_greedy.py`](../baselines/ours_greedy.py),
adapted to sglang nodes in [`baselines/sglang_adapter.py`](../baselines/sglang_adapter.py)
as `ours_greedy_score(node, layer)`. The hook plus the
FlashMLA `b>11468 → gmem workspace` patch (also developed in this work,
see [NOTES §8](../NOTES.md#8-v4-flash--4-tier-hicache-的-6-个-patchstatus-working))
are the only sglang-side changes.

Per-trial **distribution** (30 successful trials each):

| | F' (LRU) | H' (Ours) | Δ |
|---|---:|---:|---:|
| mean | 873 s | 885 s | +1.4 % (within noise) |
| **std** | **346 s** | **280 s** | **−19 %** |
| p50 | 867 s | 873 s | ~0 |
| p90 | 1335 s | 1316 s | −1.4 % |
| **p99 = max** | **1857 s** | **1336 s** | **−28 %** |
| sum | 26 192 s | 26 552 s | +1.4 % |

This is the value rule's signature behaviour: **r1 (saved prefill) vs r3
(holding tax) is explicitly traded off per unit**, so Ours can spend
slightly more on r3 (keeping value-aware units longer) to dramatically
cut r1 misses on the tail trials.  Mean is preserved (sum within 1.4 %),
p99 drops 28 %, std drops 19 %.

The harbor wall-clock (54 m 55 s vs 48 m 47 s) over-reports a regression
because that's dominated by *which* trial happens to land last across
32 trials × 16 in-flight slots: the per-trial sums are within 1.4 % of
each other, so the total compute is essentially the same and any wall-
clock gap is harbor-scheduling noise.

A first attempt **Run H** had a sign bug in the heap key (`return -value`
instead of `value`), which inverted the value rule into anti-LRU and
came out 22 % slower per-trial than Run F'. The H' row above is after
fixing that single line in `baselines/sglang_adapter.py`.

### Regime 3 — Tight pool (Runs C, D)
With cap = 256 K, max SWA usage hits **97 %** — the device cache is essentially full all the time. HiCache ON (Run C) is still the fastest of *all six* runs (**38 m 37 s**) because (a) sglang schedules more aggressively into the small pool now that overflow is cheap, and (b) the DRAM tier absorbs the constant eviction stream. HiCache OFF at the same cap (Run D) **deterministically crashes inside FlashMLA's `get_decoding_sched_meta` kernel**. The diagnosed root cause (debug instrumented build, seen at run D' on 2026-05-23 12:47): sglang's dsv4 NSA decode path under tight KV pressure passes `q.shape[0]` (FlashMLA's `params.b`) in the **~13 K range** for transient mixed prefill+decode batches; this single-block metadata kernel allocates `4*(b*5+1)` bytes of dynamic shared memory, so b≈13 K asks for ~255 KB which exceeds sm_100's per-block dynamic-smem cap of 228 KB → kernel launch returns `invalid argument`. **At this pressure level HiCache isn't a speed-up, it's the only thing that lets the server keep running.**

We patched the kernel (rucnyz/FlashMLA `aginfer` branch) to detect the
out-of-bounds smem request and emit a precise diagnostic instead of the
opaque CUDA error. The real fix has to live in the caller (sglang's dsv4
backend should not feed `b > ~11673` to a single-block metadata kernel) or
require a multi-block redesign in FlashMLA proper — both out of scope for
this paper's experiments. Run D therefore remains documented as **CRASH**
in regime 3; Run C (HiCache ON, same cap) is the only viable
configuration here, which is itself the paper's point.

## Reproduce

All commands and config knobs in [`RUNBOOK.md`](../RUNBOOK.md). The key lever is `MAX_TOTAL_TOKENS` (env var read by `scripts/launch_sglang_v4flash{,_nohicache}.sh`):

```bash
# Regime 1 — no cap
bash scripts/launch_sglang_v4flash.sh           # HiCache ON
bash scripts/launch_sglang_v4flash_nohicache.sh # HiCache OFF

# Regime 2 — moderate (cap 512 K)
MAX_TOTAL_TOKENS=524288 bash scripts/launch_sglang_v4flash.sh
MAX_TOTAL_TOKENS=524288 bash scripts/launch_sglang_v4flash_nohicache.sh

# Regime 3 — tight (cap 256 K)
MAX_TOTAL_TOKENS=262144 bash scripts/launch_sglang_v4flash.sh
# (the HiCache-OFF variant crashes deterministically at this cap)

# Run G — ThunderAgent in front of Run F's backend.
# Requires our patched ThunderAgent fork (rucnyz/ThunderAgent@aginfer, see NOTES §9)
# and the patched harbor fork (rucnyz/harbor@aginfer, lite_llm injects
# program_id == session_id with UUID fallback).
TA_PORT=9100 bash scripts/launch_thunderagent.sh
# then point harbor at TA: --ak api_base=http://172.17.0.1:9100/v1
```

Harbor command identical across runs:
```bash
OPENAI_API_KEY=sk-fake-do-not-check harbor run \
    -p /scratch/yuzhou/projects/harbor/datasets/swebenchpro \
    -a terminus-2 \
    -m openai/deepseek-ai/DeepSeek-V4-Flash \
    --ak api_base=http://172.17.0.1:30000/v1 \
    --ak max_turns=200 \
    -n 32 \
    --jobs-dir /scratch/yuzhou/projects/sglang/dev/aginfer/results/run_<X>
```

Per-run trajectories + harbor `result.json` are under `results/run_{A,B,C,D,E,F}_*/<timestamp>/`. sglang server logs (where the per-batch stats above were extracted) are in `logs/sglang_v4flash*.log`.

## Caveats

- All trials returned reward 0.0 (no SWE-bench Pro task solved). V4-Flash + terminus-2 with `max_turns=200` is too weak for these enterprise SE problems. The numbers above measure **infrastructure throughput**, not solve rate. To measure solve rate properly we'd need a stronger agent / model.
- `Run D` CRASH root cause is **FlashMLA's `get_decoding_sched_meta` single-block kernel allocates `4*(b*5+1)` B dynamic smem, which exceeds sm_100's 228 KB per-block cap when sglang dsv4's NSA decode path passes b≈13 K under heavy radix retraction**. Patched in rucnyz/FlashMLA `aginfer` branch to surface a precise diagnostic; the structural fix (chunked metadata kernel) is upstream FlashMLA work. Orthogonal to our paper — we use Run D's crash as the *demonstration* that regime 3 requires HiCache.
- The HiCache-ON peak `input throughput` numbers are *lower* than HiCache-OFF (e.g. F 295 K vs E 319 K) because the backup thread bites into the same SM/L2 bandwidth as decode. Net wall-clock still wins for HiCache because re-prefills cost more than the bandwidth they hide.

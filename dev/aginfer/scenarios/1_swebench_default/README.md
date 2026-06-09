# swebench_default — original workload, no caps

## Workload spec

| param | value |
|---|---|
| harbor profile | `swebenchpro` |
| harbor agent | `terminus-2` |
| trials (`-n`) | 32 |
| concurrent (`-l`) | 32 |
| `--max-turns` | 200 |
| `temperature` | 0.0 |
| `seed` | 42 |
| sglang `--max-total-tokens` | unset (~10 M default) |
| `--max-completion-tokens` | unset (no cap) |
| sglang TP / EP | 2 / 2 |
| HiCache | `--enable-hierarchical-cache --hicache-ratio 1.5` |
| HiCache backend | mooncake (200 GB host global segment) |
| GPUs | typically 0,1 or 5,6 (any free pair) |

Why this workload exists:
* This is the "out-of-the-box" setting — what a user would run
  swebench at if they didn't tune anything.
* Used as the **paper §8 fairness baseline** for LRU vs TA vs OURS.
* The runaway-generation tail (~ 1 % requests, 60 k completion
  tokens, 80 % wall time) is preserved here on purpose — it's
  the natural distribution for this benchmark at temperature 0.

What's **not** here:
* HBM pressure (peaks at ~ 0.02 of pool — pool is too big)
* admission_controller never fires
* daemon promote path is structurally inactive

→ For pressure-driven scenarios, see [`../hbm_pressure/`](../hbm_pressure/).

## Arms

| arm | sglang policy | daemon | result |
|---|---|---|---|
| [`lru/`](arms/lru/) | default LRU | none | 1515.1 ± 121.7 s (N=3) |
| [`ta/`](arms/ta/) | default LRU + ThunderAgent proxy :9200 | TA proxy | 1429.4 ± 62.4 s (N=3) |
| [`ours_inline/`](arms/ours_inline/) | `ours_greedy_score` inline scorer | none (no daemon) | 1392.8 ± 53.6 s (N=3) |
| [`ours_full/`](arms/ours_full/) | `ours_greedy_score` + aginfer daemon (kv_scheduler + admission) | aginfer | 1391.7 ± 105.3 s (N=4) |

Headline: see [`ANALYSIS.md`](ANALYSIS.md).

## Reproduce

```bash
bash repro.sh                # runs all 4 arms × N=3 (≈ 12 GPU hours)
```

Or one arm at a time via [`../_shared/run_k.sh <variant>`](../_shared/run_k.sh).

## Cycle naming

* `cycle1`, `cycle3`, `cycle5` = baseline arm (kv_off / lru) — B in B/O alternation
* `cycle2`, `cycle4`, `cycle6` = ours / ta arm — O in B/O alternation
* `extend_cycleN` = added later to push N≥3 → p<0.05

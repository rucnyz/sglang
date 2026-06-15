# high_concurrency — TA's own strong-gain regime (PLANNED)

## Why this workload

ThunderAgent's own benchmark (their `examples/datagen/harbor/`)
documents that at **batch size ≥ 80** with a single SGLang backend,
the radix cache saturates and TA delivers **2.08-2.48× throughput**
over default routing.  At our current bs = 32 TA provides "no
meaningful benefit" — by their own admission.

This scenario reproduces TA's strong-gain regime and adds OURS on
top.  Expected outcome: **LRU << TA << OURS**, all p < 0.05.

## Planned workload spec

| param | value | delta vs swebench_default |
|---|---|---|
| **trials (`-n`)** | **96** | **3 ×** (TA threshold is ≥ 80) |
| **concurrent (`-l`)** | **96** | **3 ×** |
| **`--max-completion-tokens`** | **8192** | **2 ×** vs hbm_pressure (more runaway → more pressure) |
| `--max-total-tokens` | 262144 (256 K) | same as hbm_pressure |
| `--max-turns` | 200 | same |
| `temperature` / `seed` | 0.0 / 42 | same |
| TP / EP | 2 / 2 | same |
| HiCache | ON | same |
| **TA `--scheduler-interval`** | **0.5 s** | **4 ×** faster reaction |
| GPUs | 5,6 | same |
| run window | 120 min (TA paper convention) | adjust if too long |

## Expected behavior

* **LRU**: single backend's radix cache thrashes → hit rate
  drops from 95 % → 30-40 % → wall explodes
* **TA**: BFD pause/resume → keeps hit rate at ~ 95 % → wall ≈
  half of LRU (TA's claimed 2.08-2.48 ×)
* **OURS**: TA-like pause + V_u-based eviction + multi-tier
  promote → wall < TA

## Open work

* Implement repro.sh (~ 1 h GPU per arm × 4 arms = 4 h GPU)
* Decide whether to also do multi-backend variant (2 sglang +
  2 workers, TA's 1.91 × scaling claim)
* Run all 4 arms × N=3

This is the **top-priority outstanding workload scenario** for
paper §8.  Blocker: none — just GPU time.

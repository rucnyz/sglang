# T38 — Default-policy module (PLAN §3, DESIGN §3 superset framing)

When no daemon is attached, sglang still has to make eviction
decisions.  DESIGN §3's "superset framing" insists this default
behavior live ON THE SAME CODE PATH as the daemon-attached case
— "modes differ only in which V_u inputs the in-process scorer
reads from the hint table."

T38 provides that default policy as a callable that plugs into
sglang's `SGLANG_KV_POLICY_MODULE` env var, so the "aginfer
disabled" mode IS just "aginfer with the default policy module
loaded."  One code path, one ablation knob.

## CONTRACT

`baselines/sglang_adapter.py:default_policy_score(node, layer) -> float`

| Inputs | Output |
|---|---|
| `node.last_access_time: int` | `float(last_access_time) + hit_count * 2^-31` |
| `node.hit_count: int` | (lower = evict first) |

Properties:
* `hit_count == 0` → byte-identical to `lru_score` → stock sglang
  behavior in the limit.
* Two nodes with different `last_access_time` → age dominates
  (the bonus is strictly < 1.0, the minimum gap between
  distinct timestamps, so it can NEVER flip age ordering).
* Two nodes with the SAME `last_access_time` → higher
  `hit_count` ranks higher (kept longer).  Mirrors sglang's
  historical `hit_count >= write_through_threshold` signal.

## WHY hit-count tie-break

Pure LRU breaks ties arbitrarily (heap insertion order).  Under
batched prefill the same scheduler step refreshes multiple leaves
to identical `last_access_time`, so ties are common in practice
— and the arbitrary choice has been blamed for unstable eviction
trajectories in past Run K traces.  The 2^-31 bonus makes the
choice deterministic AND prefers hot leaves; safe by construction
(can't flip age-distinct pairs).

## STAGES (8)

```
A. Shape
  A0 returns float
  A1 hit_count=0 byte-identical to lru_score (no hidden offset)
B. Ordering invariants
  B0 age dominates regardless of hit_count (adversarial: older
     node has hit=2^30 bonus, newer has hit=0; older still wins)
  B1 uniform hit_count → ordering matches lru_score over 10
     random nodes
  B2 tied last_access_time → higher hit_count wins
  B3 max bonus across full int32 range < 1.0 (the invariant that
     makes B0 hold for ALL hit_counts)
C. Plugin shape
  C0 `baselines.sglang_adapter:default_policy_score` importable
     via the SGLANG_KV_POLICY_MODULE format
  C1 sglang's own `_load_eviction_scorer` resolves the spec end-
     to-end (catches env-var format / import-path drift)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t38/verify.py
```

Runs in ~0.5 s.  Pure-Python; no sglang launch needed (stage C1
just imports the resolver).

## RESULTS

**PASSED** — all 8 stages.

* date: 2026-06-01
* raw log: `results/20260601_t38_initial_pass.log`
* implementation: 8 LoC in `baselines/sglang_adapter.py`
  (`_DEFAULT_HIT_COUNT_BONUS` + `default_policy_score`)

| Stage | Result |
|---|---|
| A0 returns float | PASS |
| A1 hit=0 ≡ lru_score | PASS |
| B0 age dominates over any hit_count | PASS |
| B1 uniform hit ordering = LRU | PASS |
| B2 tied age → hit wins | PASS |
| B3 max bonus < 1.0 invariant | PASS |
| C0 module:callable importable | PASS |
| C1 sglang resolver loads it | PASS |

## ABLATION USAGE

To launch sglang with this default policy (instead of stock LRU):

```bash
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:default_policy_score \
  python -m sglang.launch_server ...
```

To launch with the ours-greedy V_u (paper §7):

```bash
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score \
  python -m sglang.launch_server ...
```

One env-var flip = one ablation; same code path through the
cache manager.

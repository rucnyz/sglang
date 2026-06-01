# T38 — Default-policy module (PLAN §3, DESIGN §3 superset framing)

When no daemon is attached, sglang still has to make eviction
decisions.  DESIGN §3's "superset framing" insists this default
behavior live ON THE SAME CODE PATH as the daemon-attached case
— "modes differ only in which V_u inputs the in-process scorer
reads from the hint table."

T38 provides that default policy as a CALLABLE that plugs into
sglang's `SGLANG_KV_POLICY_MODULE` env var.  Operators (or
ablation harnesses) flip one env var to select it.

## SCOPE BOUNDARIES (audit #176-round1)

This commit ships the CALLABLE half of T38 only.  Two
DESIGN §3 spec items remain open as follow-on work:

1. **Default-on-no-env-var wiring**: sglang's
   `_load_eviction_scorer` (`unified_radix_cache.py:69`) still
   falls back to `_default_eviction_score` (pure LRU) when
   `SGLANG_KV_POLICY_MODULE` is unset.  To realise the "one code
   path" claim, the fallback needs to be `default_policy_score`.
   That's a real sglang patch + an ablation regression check
   against stock behavior — tracked separately (see PLAN §3 T38
   follow-on note).
2. **`should_write_through(node)` plugin point**: DESIGN §3 names
   write-through policy as the OTHER half of the default-policy
   module (alongside eviction scoring).  Not implemented here.
   Tracked separately.

In the current state of the world, "aginfer disabled" mode (no
env var) runs sglang's internal `_default_eviction_score` (pure
LRU, no hit-count tie-break).  Setting
`SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:default_policy_score`
runs T38's default-policy module instead.  The two differ ONLY
when two leaves share the same `last_access_time` — see
"ABLATION USAGE" below.

## CONTRACT

`baselines/sglang_adapter.py:default_policy_score(node, layer) -> float`

| Inputs | Output |
|---|---|
| `node.last_access_time: int` | `float(last_access_time) + hit_count * 2^-50` |
| `node.hit_count: int` | (lower = evict first) |

Properties:
* `hit_count == 0` → byte-identical to `lru_score` → stock sglang
  behavior in the limit.
* Two nodes with different `last_access_time` → age dominates
  for any practical `hit_count`.  `hit_count` is an unbounded
  Python int (`unified_radix_cache.py:1703`), but the bonus
  exponent `2^-50` keeps max bonus under 1.0 for hit_counts up
  to 2^50 ≈ 10^15 — well beyond any realistic deployment.
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
| B3 max bonus at int32 < 1.0 | PASS |
| B4 age dominates under unbounded hit_count (#176 regression pin) | PASS |
| C0 module:callable importable | PASS |
| C1 sglang resolver loads it | PASS |

## ABLATION USAGE

To launch sglang with T38's default policy:

```bash
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:default_policy_score \
  python -m sglang.launch_server ...
```

To launch with the ours-greedy V_u (paper §7):

```bash
SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score \
  python -m sglang.launch_server ...
```

**Caveat (audit #176)**: stock sglang with NO env var runs
`_default_eviction_score` (pure LRU, no hit-count tie-break), NOT
this module.  So:

| launch | eviction key |
|---|---|
| no env var | `last_access_time` (sglang's `_default_eviction_score`) |
| `SGLANG_KV_POLICY_MODULE=…:default_policy_score` (T38) | `last_access_time + hit_count × 2^-50` |
| `SGLANG_KV_POLICY_MODULE=…:ours_greedy_score` (paper §7) | V_u(node) — saved-prefill − holding tax |

The first two only differ in the **tied-`last_access_time`** case
(which is common under batched prefill, where many leaves
refresh together in the same scheduler step).  For Run K
ablations comparing T38 vs ours-greedy, this is the right
denominator.  For Run K ablations comparing against historical
sglang, use NO env var so the comparison is byte-identical.

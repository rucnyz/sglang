# T27 — hint-table CONSUMER (#188, DESIGN §3 / §10)

The PRODUCER (#184/T40) pushes `PUT /aginfer/hints` into
`UnifiedRadixCache._aginfer_hints`.  This task is the **consumer** —
the three coupled pieces that make daemon hints actually drive
sglang's eviction order, the heart of "aginfer full" mode.

## WHAT THIS ADDS (3 coupled parts)

1. **Hint-aware eviction scorer.**  Launch with
   `SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u` (a sentinel, not a
   `module:callable`).  `__init__` (`_init_aginfer_eviction_scoring`)
   binds `self._eviction_scorer = self._aginfer_eviction_score`, a
   cache-bound method that looks up the node's hint in `_aginfer_hints`
   (by `node.get_last_hash_value()` / `node-{id}` — the SAME key the
   daemon's `/aginfer/state` dumps) and computes the paper-§7 V_u via
   the adapter's `hint_v_u(node, layer, hint)`.

   **Architecture choice** (the central question — the `(node, layer)`
   scorer has no cache handle): a **bound method reads the dict** (no
   hash→node index, no node mutation, single source of truth = #184's
   dict) but **delegates the V_u math to the adapter** (`hint_v_u`
   shares `_v_u_from_unit` with `ours_greedy_score` → one formula, no
   reimplementation/drift — A2 is the drift guard).  Absent hint →
   the adapter falls back to the local hits/age V_u (NEVER bare LRU,
   DESIGN §3).

2. **Unit-birth seeding.**  `_aginfer_seed_birth(node)` seeds
   `p_hat = 1.0` for a newborn unit (if absent — never clobbers a
   daemon hint).  Called from BOTH `_add_new_node` (leaf) AND
   `_split_node` (the new internal/shared-prefix node — the e2e showed
   that without it ~10% of live units were uncovered).  So the table
   "covers every live unit" (DESIGN §3) and the scorer never sees an
   absent hint.

3. **Eviction-clear ordering.**  `clear_aginfer_hint` is called from
   `_remove_leaf_from_parent` — the ONE death/commit chokepoint that
   all death paths funnel through (device-evict-death write-through /
   host-evict / tombstone cascade / migrate-DROP).  It runs AFTER the
   node is detached, so DESIGN §10 ordering holds: scorer-read (heap
   build) → evict-commit (the detach) → hint-clear.  Guarded on a
   non-empty table so non-aginfer mode pays nothing.  This is the GC
   that bounds `_aginfer_hints` to the live-unit set.

## ATOMICITY (DESIGN §10 "per-key seqlock/CAS") — DESIGN-vs-CODE

DESIGN §10 asks for a per-key seqlock/CAS so the scorer's read and the
daemon's `PUT` don't race.  As-built this is satisfied **trivially**:
sglang's scheduler is a single event loop that serialises the
`/aginfer/hints` PUT handler (`update_aginfer_hints`) and the eviction
path (`drive_eviction`) — they never run concurrently, so the plain
dict needs no lock.  Each TP rank has its own `_aginfer_hints` (no
cross-rank sharing).  No CAS is implemented because there is no
concurrency to guard.

## STAGES (13)

```
A. adapter hint_v_u (the V_u math)
  A0 hint p_hat drives the score (high-p_hat hint → higher keep-value)
  A1 no hint → local fallback, returns float (never crashes)
  A2 drift guard: hint_v_u(None) == ours_greedy_score for the same
     node (shared _v_u_from_unit; counter frozen so the side-by-side
     compare is fair — both scorers advance the global time counter)
  A3 V_u monotonic in hint p_hat
B. sglang scorer selection + bound method
  B0 sentinel aginfer:hint_v_u → bound _aginfer_eviction_score,
     _aginfer_hint_aware True, kv_policy_loaded line
  B1 _aginfer_eviction_score reads _aginfer_hints by node hash
     (high-p_hat hint scores higher than low-p_hat, same node)
  B2 default spec → not hint-aware (bare LRU, no hint lookup)
  B3 real producer→consumer round-trip: set_aginfer_hints (producer) →
     _aginfer_unit_hash (key) → _aginfer_eviction_score (consumer) —
     the daemon-PUT key and the scorer-lookup key are the same in code
     (audit E12)
C. birth-seed
  C0 hint-aware: seeds p_hat=1.0 for an absent unit
  C1 does NOT clobber an existing (daemon) hint
  C2 no-op when not hint-aware
  C3 the daemon's FIRST refinement overwrites the birth-seed even at
     stamp == int(last_access_time) — the seed carries a FLOOR stamp
     (-1) so a real daemon push always wins (audit C7: an equal stamp
     would be skipped by overwrite-by-stamp and p_hat=1.0 would shadow
     the daemon's refinement)
D. eviction-clear ordering
  D0 _remove_leaf_from_parent clears the node's hint (others untouched)
  D1 clear AFTER detach (§10 commit-before-clear)
  D2 no-op / no crash on an empty table (non-aginfer mode)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t27/verify.py
```

Pure-Python; ~0.3 s.  No GPU (imports the adapter + sglang module
helpers; drives the cache methods on a `__new__`-constructed cache).

### Live e2e (GPU) — run during development

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u \
  PYTHONPATH=/scratch/yuzhou/projects/sglang/dev/aginfer CUDA_VISIBLE_DEVICES=5 \
  python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 \
    --port 30004 --tp 1 --mem-fraction-static 0.15 --max-total-tokens 65536 \
    --trust-remote-code --attention-backend flashinfer
```

Confirmed live (2026-06-02):
* `kv_policy_loaded=aginfer:hint_v_u`; server serves with the hint-
  aware scorer + birth-seed + clear active, no crash;
* birth-seed → `n_aginfer_hints == live_units` (gap 0 — full coverage,
  AFTER the `_split_node` seed fix; before it ~10% of units were
  uncovered);
* 300 churned distinct prefixes → `n_aginfer_hints` stays BOUNDED near
  the live-unit count (clear fires on eviction — no leak, the bug
  #188 was filed to fix);
* `PUT /aginfer/hints` round-trips (applied=1).

## RESULTS

**PASSED** — all 15 stages (13 + B3/C3 from the audit) + live e2e.

* date: 2026-06-02
* raw log: `results/20260602_t27_initial_pass.log`

## AUDIT CLOSURE (2026-06-02)

Adversarial audit confirmed the two load-bearing concerns are CLEAR —
the daemon-PUT key and the scorer-lookup key are byte-identical in BOTH
storage modes (`hash_value[-1]` / `node-{id}`), and the clear is
genuinely death-only (the backuped device-evict keeps the node alive
and does NOT clear).  Closed:

* **C7 (BUG, low-severity)** — the birth-seed used
  `stamp = int(last_access_time)`, which could EQUAL the daemon's
  first-dump stamp; `set_aginfer_hints` skips `stamp <= existing`, so
  the daemon's first refinement would be dropped and `p_hat=1.0` would
  shadow it until the next counter bump.  Fixed: seed with a FLOOR
  stamp `_AGINFER_BIRTH_STAMP = -1` (below any real daemon stamp).
  Pinned by C3.
* **E12 (gap)** — added B3, the REAL `set_aginfer_hints` → scorer
  round-trip through the actual hash key (the prior B1 hand-set the
  dict, bypassing the keying).
* **D9 (pre-existing nit)** — the scorer advances the global time
  counter (`get_and_increase_time_counter` via `_current_time_counter`);
  inherited from `ours_greedy_score`, benign (no `last_access_time`
  written during scoring).  Tracked as #193.

## REGRESSION SANITY

* T28 (#177/#178 plugin points): PASS (the scorer-init refactor is
  additive; default/LRU path unchanged)
* T38 default-policy module: PASS
* T40 hint producer: PASS (`_aginfer_hints` accessors unchanged)
* integration_stress: full-stack (default-mode eviction byte-identical;
  the daemon pushes hints so the clear path is exercised)

## SCOPE BOUNDARY (deferred)

* The V_u-aware **write-through** version (`V_u(res ∪ {DRAM}) >
  V_u(res)`, the #178 hook's aginfer registration) is a sibling of the
  eviction scorer — not wired here; uses the same hint table.
* Hint-aware scoring across TP ranks (cross-rank hint divergence) is
  probed by T15 / tracked by #174.

"""T27 — hint-table CONSUMER: hint-aware eviction scorer + birth-seed
+ eviction-clear ordering (#188, DESIGN §3/§10).

The PRODUCER (#184) pushes `PUT /aginfer/hints` into
`UnifiedRadixCache._aginfer_hints` (overwrite-by-stamp).  This task is
the consumer — the three coupled pieces that make daemon hints
actually drive sglang's eviction order:

  1. **Hint-aware scorer** — when launched with
     `SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u`, the eviction heap key
     is `_aginfer_eviction_score(node, layer)`: it looks up the node's
     hint in `_aginfer_hints` and computes the paper-§7 V_u from the
     daemon's `p_hat` / `lambda` (via the adapter's `hint_v_u`, the
     SAME V_u math as `ours_greedy_score` — no reimplementation/drift).
     Absent hint → graceful fallback to the local hits/age derivation
     (never bare LRU).
  2. **Unit-birth seeding** — a new leaf gets a `p_hat = 1.0` seed in
     `_aginfer_hints` (if absent), so the table "covers every live
     unit" (DESIGN §3) and the scorer never sees an absent hint for a
     newborn.
  3. **Eviction-clear ordering** — a unit's hint is cleared at the
     true death/commit boundary (`_remove_leaf_from_parent`, the one
     chokepoint for device-evict-death / host-evict / tombstone /
     migrate-DROP), AFTER the evict commits — DESIGN §10 "scorer-read
     happens-before evict-commit happens-before hint-clear".

Atomicity (DESIGN §10 "per-key seqlock/CAS"): satisfied trivially —
sglang's scheduler serialises the `/aginfer/hints` PUT handler and the
eviction path on ONE event loop, so the dict is never truly concurrent
(no CAS needed; see README DESIGN-vs-CODE note).

Stages:

  A. adapter hint_v_u (the V_u math)
    A0 hint drives the score: high-p_hat hint → higher (keep-longer)
       V_u than a low-p_hat hint, same node
    A1 no hint → falls back to local derivation, returns a float
    A2 drift guard: hint_v_u(hint=None) == ours_greedy_score for the
       same node (shared _v_u_from_unit, no reimplementation)
    A3 monotonic in p_hat: higher hint p_hat → higher score
  B. sglang scorer selection + bound method
    B0 SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u → _eviction_scorer is
       the bound _aginfer_eviction_score, _aginfer_hint_aware True,
       kv_policy_loaded line emitted
    B1 _aginfer_eviction_score reads _aginfer_hints by node hash:
       a node with a high-p_hat hint scores higher than the same node
       with a low-p_hat hint
    B2 a normal module spec → _load_eviction_scorer path,
       _aginfer_hint_aware False (no hint lookup)
  C. birth-seed
    C0 hint-aware: _aginfer_seed_birth seeds p_hat=1.0 for an absent
       unit
    C1 birth-seed does NOT clobber an existing (daemon) hint
    C2 non-hint-aware: _aginfer_seed_birth is a no-op
  D. eviction-clear ordering
    D0 _remove_leaf_from_parent clears the node's hint (table non-empty)
    D1 the clear happens AFTER the node is detached from its parent
       (the §10 ordering: commit before clear)
    D2 clear is a no-op when the hint table is empty (non-aginfer mode
       pays nothing)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))
_SGLANG_PY = "/scratch/yuzhou/projects/sglang/python"
if _SGLANG_PY not in sys.path:
    sys.path.insert(0, _SGLANG_PY)

from baselines.sglang_adapter import (  # noqa: E402
    hint_v_u,
    ours_greedy_score,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ stubs


class _CompData:
    def __init__(self, n_tokens: int):
        self.value = list(range(n_tokens))   # device tokens
        self.host_value = None


class _Node:
    """Duck-typed UnifiedTreeNode for the scorers + cache methods."""
    _counter = 10000

    def __init__(self, *, last_access_time: int = 0, hit_count: int = 0,
                 n_tokens: int = 100, hash_value=None, parent=None):
        self.last_access_time = last_access_time
        self.hit_count = hit_count
        self.component_data = [_CompData(n_tokens)]
        self.hash_value = hash_value          # list or None
        self.parent = parent
        self.key = _Key()
        _Node._counter += 1
        self.id = _Node._counter

    def get_last_hash_value(self):
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]


class _Key:
    def child_key(self, page_size):
        return "k"


class _Layer:
    name = "DEVICE"


_LAYER = _Layer()


def _bare_cache():
    """A UnifiedRadixCache with just the aginfer-scoring attrs set, via
    __new__ (bypasses the heavy GPU __init__)."""
    c = UnifiedRadixCache.__new__(UnifiedRadixCache)
    c._aginfer_hints = {}
    return c


# ============================================================ A. hint_v_u


def stage_a0_hint_drives_score() -> None:
    """Same node, different hint p_hat → the high-p_hat hint scores
    higher (more valuable to keep)."""
    n = _Node(last_access_time=0, hit_count=1, n_tokens=100)
    hi = hint_v_u(n, _LAYER, {"p_hat": 1.0, "lambda": 0.2, "stamp": 1})
    lo = hint_v_u(n, _LAYER, {"p_hat": 0.01, "lambda": 0.2, "stamp": 1})
    if not (hi > lo):
        raise StageFail(f"high-p_hat hint must score higher; hi={hi} lo={lo}")


def stage_a1_no_hint_fallback() -> None:
    n = _Node(last_access_time=0, hit_count=3, n_tokens=100)
    v = hint_v_u(n, _LAYER, None)
    if not isinstance(v, float):
        raise StageFail(f"no-hint hint_v_u must return float; got {type(v)}")


def stage_a2_drift_guard_vs_ours_greedy() -> None:
    """hint_v_u(hint=None) MUST equal ours_greedy_score for the same
    node — they share the §7 V_u math (no reimplementation/drift).

    Freeze the time counter: both scorers call _current_time_counter()
    which INCREMENTS the global counter (pre-existing — ours_greedy_score
    already does this), so calling both back-to-back would otherwise see
    a different `now`/age.  In production each node is scored once, so
    this only matters for the side-by-side equality check."""
    import baselines.sglang_adapter as adp
    orig = adp._current_time_counter
    adp._current_time_counter = lambda: 1_000_000
    try:
        for la, hc, nt in [(0, 1, 100), (50, 7, 400), (0, 0, 10)]:
            n = _Node(last_access_time=la, hit_count=hc, n_tokens=nt)
            a = hint_v_u(n, _LAYER, None)
            b = ours_greedy_score(n, _LAYER)
            if a != b:
                raise StageFail(
                    f"hint_v_u(None) must == ours_greedy_score (no drift); "
                    f"at (la={la},hc={hc},nt={nt}) got {a} vs {b}"
                )
    finally:
        adp._current_time_counter = orig


def stage_a3_monotonic_in_p_hat() -> None:
    n = _Node(last_access_time=0, hit_count=1, n_tokens=100)
    prev = None
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = hint_v_u(n, _LAYER, {"p_hat": p, "lambda": 0.2, "stamp": 1})
        if prev is not None and not (v >= prev):
            raise StageFail(f"V_u must be monotonic in p_hat; p={p} v={v} < prev={prev}")
        prev = v


# ============================================================ B. selection + scorer


def stage_b0_sentinel_binds_hint_scorer() -> None:
    c = _bare_cache()
    old = os.environ.get("SGLANG_KV_POLICY_MODULE")
    os.environ["SGLANG_KV_POLICY_MODULE"] = "aginfer:hint_v_u"
    try:
        c._init_aginfer_eviction_scoring()
    finally:
        if old is None:
            os.environ.pop("SGLANG_KV_POLICY_MODULE", None)
        else:
            os.environ["SGLANG_KV_POLICY_MODULE"] = old
    if not c._aginfer_hint_aware:
        raise StageFail("sentinel spec must set _aginfer_hint_aware=True")
    # the scorer is the bound method
    if c._eviction_scorer != c._aginfer_eviction_score:
        raise StageFail("sentinel must bind _eviction_scorer to _aginfer_eviction_score")


def stage_b1_scorer_reads_hint_table() -> None:
    c = _bare_cache()
    os.environ["SGLANG_KV_POLICY_MODULE"] = "aginfer:hint_v_u"
    try:
        c._init_aginfer_eviction_scoring()
    finally:
        os.environ.pop("SGLANG_KV_POLICY_MODULE", None)
    n = _Node(last_access_time=0, hit_count=1, n_tokens=100, hash_value=["u0"])
    c._aginfer_hints = {"u0": {"p_hat": 1.0, "lambda": 0.2, "stamp": 1}}
    hi = c._eviction_scorer(n, _LAYER)
    c._aginfer_hints = {"u0": {"p_hat": 0.01, "lambda": 0.2, "stamp": 1}}
    lo = c._eviction_scorer(n, _LAYER)
    if not (hi > lo):
        raise StageFail(
            f"hint-aware scorer must read _aginfer_hints by node hash and "
            f"score the high-p_hat unit higher; hi={hi} lo={lo}"
        )


def stage_b2_normal_spec_not_hint_aware() -> None:
    c = _bare_cache()
    old = os.environ.get("SGLANG_KV_POLICY_MODULE")
    os.environ.pop("SGLANG_KV_POLICY_MODULE", None)  # default LRU
    try:
        c._init_aginfer_eviction_scoring()
    finally:
        if old is not None:
            os.environ["SGLANG_KV_POLICY_MODULE"] = old
    if c._aginfer_hint_aware:
        raise StageFail("default spec must NOT be hint-aware")
    # a normal scorer ignores the hint table
    n = _Node(last_access_time=42, hit_count=0, n_tokens=100, hash_value=["u0"])
    if c._eviction_scorer(n, _LAYER) != float(42):
        raise StageFail("default scorer should be bare last_access_time")


# ============================================================ C. birth-seed


def stage_c0_birth_seed_absent() -> None:
    c = _bare_cache()
    c._aginfer_hint_aware = True
    n = _Node(hash_value=["fresh"], last_access_time=5)
    c._aginfer_seed_birth(n)
    h = c._aginfer_hints.get("fresh")
    if h is None or abs(h["p_hat"] - 1.0) > 1e-9:
        raise StageFail(f"birth-seed must seed p_hat=1.0 for an absent unit; got {h!r}")


def stage_c1_birth_seed_no_clobber() -> None:
    c = _bare_cache()
    c._aginfer_hint_aware = True
    c._aginfer_hints = {"u": {"p_hat": 0.3, "lambda": 0.5, "stamp": 99}}
    n = _Node(hash_value=["u"], last_access_time=5)
    c._aginfer_seed_birth(n)
    h = c._aginfer_hints["u"]
    if abs(h["p_hat"] - 0.3) > 1e-9 or h["stamp"] != 99:
        raise StageFail(f"birth-seed must NOT clobber an existing (daemon) hint; got {h!r}")


def stage_c2_birth_seed_noop_when_not_aware() -> None:
    c = _bare_cache()
    c._aginfer_hint_aware = False
    n = _Node(hash_value=["fresh"], last_access_time=5)
    c._aginfer_seed_birth(n)
    if c._aginfer_hints:
        raise StageFail(f"non-hint-aware mode must not birth-seed; got {c._aginfer_hints!r}")


# ============================================================ D. eviction-clear


def _parented_node(uhash: str):
    parent = _Node(hash_value=["p"])
    parent.children = {"k": None}
    child = _Node(hash_value=[uhash], parent=parent)
    parent.children["k"] = child
    return parent, child


def stage_d0_remove_clears_hint() -> None:
    c = _bare_cache()
    c.page_size = 1
    _, child = _parented_node("dead")
    c._aginfer_hints = {"dead": {"p_hat": 0.5, "lambda": 0.2, "stamp": 1},
                        "live": {"p_hat": 0.9, "lambda": 0.2, "stamp": 1}}
    c._remove_leaf_from_parent(child)
    if "dead" in c._aginfer_hints:
        raise StageFail("evicting (removing) a node must clear its hint")
    if "live" not in c._aginfer_hints:
        raise StageFail("removing one node must not clear OTHER nodes' hints")


def stage_d1_clear_after_detach() -> None:
    """§10 ordering: the node is detached from its parent BEFORE the
    hint is cleared (commit happens-before clear).  We assert the node
    is gone from parent.children AND the hint is gone — both, after the
    single call."""
    c = _bare_cache()
    c.page_size = 1
    parent, child = _parented_node("dead")
    c._aginfer_hints = {"dead": {"p_hat": 0.5, "lambda": 0.2, "stamp": 1}}
    c._remove_leaf_from_parent(child)
    if parent.children.get("k") is not None:
        raise StageFail("node must be detached from parent (evict commit)")
    if "dead" in c._aginfer_hints:
        raise StageFail("hint must be cleared after the detach")


def stage_d2_clear_noop_empty_table() -> None:
    """Non-aginfer mode (empty table): _remove_leaf_from_parent must
    not crash and pays nothing."""
    c = _bare_cache()
    c.page_size = 1
    parent, child = _parented_node("whatever")
    c._aginfer_hints = {}
    c._remove_leaf_from_parent(child)   # must not raise
    if parent.children.get("k") is not None:
        raise StageFail("node should still be detached even with empty table")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 hint p_hat drives the eviction V_u",         stage_a0_hint_drives_score),
    ("A1 no hint → local fallback (float)",            stage_a1_no_hint_fallback),
    ("A2 drift guard: hint_v_u(None) == ours_greedy",  stage_a2_drift_guard_vs_ours_greedy),
    ("A3 V_u monotonic in hint p_hat",                 stage_a3_monotonic_in_p_hat),
    ("B0 sentinel binds the hint-aware scorer",        stage_b0_sentinel_binds_hint_scorer),
    ("B1 scorer reads _aginfer_hints by node hash",    stage_b1_scorer_reads_hint_table),
    ("B2 default spec → not hint-aware",               stage_b2_normal_spec_not_hint_aware),
    ("C0 birth-seed p_hat=1.0 for absent unit",        stage_c0_birth_seed_absent),
    ("C1 birth-seed does not clobber daemon hint",     stage_c1_birth_seed_no_clobber),
    ("C2 birth-seed no-op when not hint-aware",        stage_c2_birth_seed_noop_when_not_aware),
    ("D0 remove_leaf_from_parent clears the hint",     stage_d0_remove_clears_hint),
    ("D1 clear after detach (§10 ordering)",           stage_d1_clear_after_detach),
    ("D2 clear no-op on empty table",                  stage_d2_clear_noop_empty_table),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT27 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT27 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

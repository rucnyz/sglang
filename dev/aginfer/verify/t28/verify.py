"""T28 + #177 — DESIGN §3 plugin points: default eviction scorer +
should_write_through (#177 + #178).

DESIGN §3 ("Aginfer is sglang's decision pipeline — superset
framing"): sglang's historical heuristics are expressed as aginfer's
DEFAULT policy module, reached through the SAME code path whether or
not the daemon is attached.  Two physical plugin points carry this:

  * **Eviction scorer** (`SGLANG_KV_POLICY_MODULE`) — already wired
    (T29).  #177 makes the in-process DEFAULT the LRU-equivalent V_u
    (`last_access_time` + a `hit_count` tie-break), matching the
    daemon-side `baselines.sglang_adapter:default_policy_score`, so
    "aginfer disabled" and "aginfer default policy" are literally the
    same function — one code path.
  * **Write-through trigger** — #178 adds the NEW
    `should_write_through(node, threshold)` plugin point
    (`SGLANG_WRITE_THROUGH_MODULE`), factoring the hardcoded
    `hit_count >= write_through_threshold` check in `_inc_hit_count`
    into a pluggable hook whose default preserves historical
    behaviour.

Stages:

  A. eviction default (#177)
    A0 _default_eviction_score = last_access_time + hit_count·2^-50
       (the LRU-equivalent V_u, not bare last_access_time)
    A1 ablation / no-tie path: two DISTINCT last_access_time nodes are
       ordered by AGE regardless of hit_count (the bonus < 1.0 never
       flips a distinct-age pair → baseline behaviour unchanged)
    A2 tie-break path: two SAME-age nodes order by hit_count (higher
       hits kept longer) — the DESIGN §3 refinement
    A3 cross-tree drift guard: sglang `_default_eviction_score` ==
       daemon `baselines.sglang_adapter.default_policy_score` for
       sampled nodes (catches formula / 2^-50 constant drift between
       the two trees — the #175 drift-guard pattern)
    A4 plugin override still resolves (T29 contract intact)
  B. write-through plugin (#178)
    B0 _default_should_write_through(node, threshold) == (hit_count
       >= threshold) — preserves historical behaviour
    B1 _load_write_through_policy() with no env → the default
    B2 env SGLANG_WRITE_THROUGH_MODULE=mod:fn → loads it; malformed /
       failed → falls back to default (mirrors the eviction-scorer
       T9 load contract)
    B3 callsite integration: _inc_hit_count calls self._write_through_
       policy — an override deciding True/False overrides the
       hardcoded threshold (write_backup fires per the POLICY)
    B4 default callsite regression: with the default policy,
       write_backup fires IFF hit_count >= threshold (byte-identical
       to pre-#178 behaviour)
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

from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    _default_eviction_score,
    _load_eviction_scorer,
    _default_should_write_through,
    _load_write_through_policy,
    UnifiedRadixCache,
)
from baselines.sglang_adapter import default_policy_score  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ stubs


class _Node:
    """Duck-typed UnifiedTreeNode for the score functions (they read
    only last_access_time + hit_count)."""
    def __init__(self, last_access_time: int, hit_count: int = 0,
                 backuped: bool = False, evicted: bool = False):
        self.last_access_time = last_access_time
        self.hit_count = hit_count
        self.backuped = backuped
        self.evicted = evicted


class _Layer:
    name = "FULL"


_LAYER = _Layer()


# ============================================================ A. eviction default


def stage_a0_default_is_bare_lru() -> None:
    """The in-process eviction default is the LRU-equivalent V_u =
    bare last_access_time (#177, DESIGN §3 'last_access as p_hat
    surrogate').  hit_count must NOT enter the eviction score (that's
    the write-through trigger's job, #178)."""
    n = _Node(last_access_time=1000, hit_count=7)
    score = _default_eviction_score(n, _LAYER)
    if score != float(n.last_access_time):
        raise StageFail(
            f"_default_eviction_score should be bare last_access_time "
            f"({float(n.last_access_time)!r}); got {score!r} — hit_count "
            f"must not affect eviction ordering"
        )


def stage_a1_distinct_age_unchanged() -> None:
    """No-tie path (ablation): two nodes with DISTINCT last_access_time
    are ordered by age regardless of hit_count — the bonus < 1.0 never
    flips a distinct-age pair, so baseline ordering is unchanged."""
    older = _Node(last_access_time=1000, hit_count=10**6)   # huge hits
    newer = _Node(last_access_time=1001, hit_count=0)
    s_old = _default_eviction_score(older, _LAYER)
    s_new = _default_eviction_score(newer, _LAYER)
    # lower score = evict first; older (smaller age) must score lower
    # EVEN with a million hits.
    if not (s_old < s_new):
        raise StageFail(
            f"distinct ages must order by age regardless of hits: "
            f"older={s_old!r} should be < newer={s_new!r}"
        )


def stage_a2_no_hit_count_tiebreak() -> None:
    """The eviction default does NOT tie-break by hit_count: two
    same-age nodes get the SAME score regardless of hits.  (#177 — an
    additive float tie-break was non-functional below the ULP at
    realistic last_access_time, and exact ties never occur since the
    cache spaces every node distinctly; hit_count drives write-through,
    #178, not eviction.)"""
    cold = _Node(last_access_time=1000, hit_count=1)
    hot = _Node(last_access_time=1000, hit_count=50)
    if _default_eviction_score(hot, _LAYER) != _default_eviction_score(cold, _LAYER):
        raise StageFail(
            "same-age nodes must score identically (no hit_count "
            "tie-break in the eviction default)"
        )


def stage_a3_cross_tree_drift_guard() -> None:
    """The sglang in-process default and the daemon-side adapter
    default_policy_score MUST be the same function of (last_access,
    hit_count) — otherwise 'aginfer disabled' and 'aginfer default
    policy' (DESIGN §3 modes 1 & 2) silently diverge.  Guards against
    formula / 2^-50 constant drift across the two trees (#175 pattern)."""
    for la, hc in [(0, 0), (1000, 7), (2**40, 10**6), (123, 1)]:
        n = _Node(last_access_time=la, hit_count=hc)
        a = _default_eviction_score(n, _LAYER)
        b = default_policy_score(n, _LAYER)
        if a != b:
            raise StageFail(
                f"sglang default scorer diverged from adapter "
                f"default_policy_score at (la={la}, hc={hc}): "
                f"sglang={a!r} adapter={b!r}"
            )
        # Spec pin (not just mutual equality): the shared value MUST be
        # bare float(last_access_time) — catches the case where BOTH
        # trees drift together (e.g. both grow a hit_count term).
        if a != float(la):
            raise StageFail(
                f"default eviction score must be bare last_access_time; "
                f"at (la={la}, hc={hc}) both trees gave {a!r} != {float(la)!r} "
                f"— hit_count leaked back into the eviction score"
            )


def stage_a4_plugin_override_resolves() -> None:
    """T29 contract intact: an env override still loads (the #177
    default change must not break the plugin mechanism)."""
    spec = "baselines.sglang_adapter:default_policy_score"
    old = os.environ.get("SGLANG_KV_POLICY_MODULE")
    os.environ["SGLANG_KV_POLICY_MODULE"] = spec
    try:
        fn = _load_eviction_scorer()
    finally:
        if old is None:
            os.environ.pop("SGLANG_KV_POLICY_MODULE", None)
        else:
            os.environ["SGLANG_KV_POLICY_MODULE"] = old
    if fn is not default_policy_score:
        raise StageFail(
            f"override SGLANG_KV_POLICY_MODULE={spec} should resolve to "
            f"default_policy_score; got {fn!r}"
        )


# ============================================================ B. write-through


def stage_b0_default_preserves_threshold() -> None:
    for hc, thr, expect in [(0, 2, False), (1, 2, False), (2, 2, True),
                            (5, 2, True), (1, 1, True)]:
        n = _Node(last_access_time=0, hit_count=hc)
        got = _default_should_write_through(n, thr)
        if got is not expect:
            raise StageFail(
                f"_default_should_write_through(hit={hc}, thr={thr}) "
                f"should be {expect}; got {got}"
            )


def stage_b1_load_default() -> None:
    old = os.environ.get("SGLANG_WRITE_THROUGH_MODULE")
    os.environ.pop("SGLANG_WRITE_THROUGH_MODULE", None)
    try:
        fn = _load_write_through_policy()
    finally:
        if old is not None:
            os.environ["SGLANG_WRITE_THROUGH_MODULE"] = old
    if fn is not _default_should_write_through:
        raise StageFail(
            f"no-env _load_write_through_policy should be the default; "
            f"got {fn!r}"
        )


def stage_b2_load_override_and_failure() -> None:
    # valid override — a REAL (node, threshold) -> bool write-through
    # policy (not a wrong-signature eviction scorer); assert it both
    # RESOLVES and DECIDES correctly when invoked (audit D12: the prior
    # fixture used a (node, layer) -> float scorer, which resolves but
    # returns a float that silently mis-decides at the callsite).
    from baselines.sglang_adapter import default_policy_should_write_through
    spec = "baselines.sglang_adapter:default_policy_should_write_through"
    old = os.environ.get("SGLANG_WRITE_THROUGH_MODULE")
    os.environ["SGLANG_WRITE_THROUGH_MODULE"] = spec
    try:
        fn = _load_write_through_policy()
        if fn is _default_should_write_through:
            raise StageFail("valid override should NOT fall back to default")
        if fn is not default_policy_should_write_through:
            raise StageFail(f"override should resolve to the spec'd callable; got {fn!r}")
        # it must actually behave as a (node, threshold) -> bool policy
        verdict = fn(_Node(last_access_time=0, hit_count=3), 2)
        if verdict is not True:
            raise StageFail(
                f"resolved write-through policy must DECIDE (hit 3 >= thr 2 "
                f"→ True); got {verdict!r} (wrong-signature callable?)"
            )
        # malformed spec → default
        os.environ["SGLANG_WRITE_THROUGH_MODULE"] = "no_colon_here"
        fn2 = _load_write_through_policy()
        if fn2 is not _default_should_write_through:
            raise StageFail("malformed spec should fall back to default")
        # unimportable → default
        os.environ["SGLANG_WRITE_THROUGH_MODULE"] = "nonexistent.mod:fn"
        fn3 = _load_write_through_policy()
        if fn3 is not _default_should_write_through:
            raise StageFail("unimportable spec should fall back to default")
    finally:
        if old is None:
            os.environ.pop("SGLANG_WRITE_THROUGH_MODULE", None)
        else:
            os.environ["SGLANG_WRITE_THROUGH_MODULE"] = old


class _StubController:
    write_policy = "write_through"


def _wt_cache(policy):
    """A UnifiedRadixCache with just enough wired to exercise
    _inc_hit_count's write-through branch."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = _StubController()
    cache.write_through_threshold = 2
    cache._write_through_policy = policy
    cache._backup_calls = []
    cache.write_backup = lambda node: cache._backup_calls.append(node)
    return cache


def stage_b3_callsite_uses_policy() -> None:
    """_inc_hit_count must consult self._write_through_policy, not the
    hardcoded threshold.  An override that says 'write through at hit 1'
    fires write_backup at hit_count 1 (below the threshold of 2);
    an override that says 'never' does not fire even above threshold."""
    # always-True override → fires at hit_count 1 (< threshold 2)
    cache = _wt_cache(lambda node, thr: True)
    n = _Node(last_access_time=0, hit_count=0, backuped=False, evicted=False)
    cache._inc_hit_count(n)  # hit_count → 1
    if len(cache._backup_calls) != 1:
        raise StageFail(
            "always-True write-through policy should fire write_backup "
            f"at hit_count=1; backups={len(cache._backup_calls)}"
        )
    # always-False override → never fires, even at hit_count 5
    cache2 = _wt_cache(lambda node, thr: False)
    n2 = _Node(last_access_time=0, hit_count=4, backuped=False, evicted=False)
    cache2._inc_hit_count(n2)  # hit_count → 5 (>= threshold 2)
    if cache2._backup_calls:
        raise StageFail(
            "always-False write-through policy must suppress write_backup "
            f"even above threshold; backups={len(cache2._backup_calls)}"
        )


def stage_b4_default_callsite_regression() -> None:
    """With the DEFAULT policy wired, _inc_hit_count fires write_backup
    IFF hit_count >= threshold — byte-identical to pre-#178."""
    cache = _wt_cache(_default_should_write_through)
    # threshold 2: hit_count goes 0→1 (no), 1→2 (yes)
    n = _Node(last_access_time=0, hit_count=0, backuped=False, evicted=False)
    cache._inc_hit_count(n)  # → 1, no backup
    if cache._backup_calls:
        raise StageFail("default: no backup at hit_count=1 (< threshold 2)")
    cache._inc_hit_count(n)  # → 2, backup
    if len(cache._backup_calls) != 1:
        raise StageFail(
            f"default: write_backup must fire at hit_count=2 (>= threshold); "
            f"backups={len(cache._backup_calls)}"
        )
    # already-backuped node never re-fires
    cache3 = _wt_cache(_default_should_write_through)
    nb = _Node(last_access_time=0, hit_count=9, backuped=True, evicted=False)
    cache3._inc_hit_count(nb)
    if cache3._backup_calls:
        raise StageFail("a backuped node must not re-trigger write_backup")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 default scorer = bare last_access_time (#177)", stage_a0_default_is_bare_lru),
    ("A1 distinct-age ordering unchanged (ablation)", stage_a1_distinct_age_unchanged),
    ("A2 no hit_count tie-break in eviction default", stage_a2_no_hit_count_tiebreak),
    ("A3 cross-tree drift guard (sglang == adapter)", stage_a3_cross_tree_drift_guard),
    ("A4 SGLANG_KV_POLICY_MODULE override still resolves", stage_a4_plugin_override_resolves),
    ("B0 _default_should_write_through == (hit >= threshold)", stage_b0_default_preserves_threshold),
    ("B1 _load_write_through_policy no-env → default", stage_b1_load_default),
    ("B2 override resolves; malformed/failed → default", stage_b2_load_override_and_failure),
    ("B3 _inc_hit_count consults the write-through policy (#178)", stage_b3_callsite_uses_policy),
    ("B4 default callsite regression (hit >= threshold)", stage_b4_default_callsite_regression),
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
        print(_red(f"\nT28 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT28 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

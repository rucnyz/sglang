"""T38 verify — default-policy module (PLAN §3 / DESIGN §3 superset).

The default-policy module is the inline scorer that runs when no
daemon is attached.  Per DESIGN §3:

    aginfer is the single decision pipeline that sglang invokes
    for every cache-management choice. sglang's historical
    heuristics (LRU eviction, hit_count >= write_through_threshold
    write-through trigger) are expressed as aginfer's default
    policy module: the policy module that runs when no daemon is
    attached.

Concrete contract (post-#177):
  * Same callable signature as the other scorers in
    ``baselines/sglang_adapter`` (``(node, layer) -> float``);
    pluggable via ``SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter
    :default_policy_score``.
  * Eviction value = **bare ``last_access_time``** — the LRU-
    equivalent V_u ("last_access as p_hat surrogate", DESIGN §3).
    Identical to ``lru_score`` for ALL inputs, and byte-identical to
    sglang's in-process ``_default_eviction_score`` (one code path).
  * ``hit_count`` does NOT affect the eviction score.  DESIGN §3 puts
    hit_count in the WRITE-THROUGH trigger (``should_write_through``
    / #178), not eviction ordering.

#177 removed an earlier ``+ hit_count·2^-50`` eviction tie-break: it
was non-functional (the bonus sits below the float64 ULP at any
realistic ``last_access_time``) AND moot — the cache assigns a
DISTINCT ``last_access_time`` to every node (same-batch prefix nodes
are spaced 1e-5 apart), so exact ties never occur.  The hit_count
write-through behaviour it was loosely mirroring lives in T28 (#178).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.sglang_adapter import (  # noqa: E402
    default_policy_score,
    lru_score,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- node + layer stubs ----


@dataclass
class _StubLayer:
    name: str


@dataclass
class _StubNode:
    """Minimum shape default_policy_score / lru_score touch."""
    last_access_time: int
    hit_count: int = 0
    id: int = 0
    component_data: Any = None  # unused for these scorers


_LAYER_HBM = _StubLayer(name="DEVICE")


# ============================================================ A. shape


def stage_a0_returns_float() -> None:
    n = _StubNode(last_access_time=42, hit_count=7)
    score = default_policy_score(n, _LAYER_HBM)
    if not isinstance(score, float):
        raise StageFail(f"score is not float: {type(score).__name__}")


def stage_a1_zero_hits_equals_lru() -> None:
    """With hit_count=0 the default policy must produce EXACTLY the
    same value as lru_score (no hidden offset)."""
    for ts in (0, 1, 42, 1_000_000):
        n = _StubNode(last_access_time=ts, hit_count=0)
        d = default_policy_score(n, _LAYER_HBM)
        l = lru_score(n, _LAYER_HBM)
        if d != l:
            raise StageFail(
                f"hit=0: default({ts})={d} != lru({ts})={l}"
            )


# ============================================================ B. ordering


def stage_b0_hit_count_does_not_affect_score() -> None:
    """Post-#177 contract: the eviction score is hit_count-INDEPENDENT.
    At a fixed last_access_time, ANY hit_count (including the unbounded
    2^32+ that sglang's `node.hit_count += 1` allows by construction)
    yields the SAME score == lru_score.  hit_count's job is the
    write-through trigger (#178), not eviction."""
    for hc in (0, 1, 50, 2 ** 30, 2 ** 32, 10 ** 18):
        n = _StubNode(last_access_time=777, hit_count=hc)
        d = default_policy_score(n, _LAYER_HBM)
        if d != float(777):
            raise StageFail(
                f"eviction score must be bare last_access_time "
                f"regardless of hit_count; hit={hc} gave {d} (expected 777.0)"
            )
        if d != lru_score(n, _LAYER_HBM):
            raise StageFail(
                f"default must equal lru_score for hit={hc}; "
                f"default={d} lru={lru_score(n, _LAYER_HBM)}"
            )


def stage_b1_uniform_hit_count_matches_lru_order() -> None:
    """Random nodes with uniform hit_count → ordering of
    default_policy_score matches lru_score.  10 nodes, uniform
    hit_count=5."""
    import random
    rng = random.Random(20260601)
    nodes = [
        _StubNode(last_access_time=rng.randint(0, 1_000_000),
                  hit_count=5)
        for _ in range(10)
    ]
    by_default = sorted(nodes, key=lambda n: default_policy_score(n, _LAYER_HBM))
    by_lru = sorted(nodes, key=lambda n: lru_score(n, _LAYER_HBM))
    if [id(x) for x in by_default] != [id(x) for x in by_lru]:
        raise StageFail(
            "uniform-hit ordering should match LRU; ties may break "
            "in different orders.  default="
            f"{[n.last_access_time for n in by_default]} "
            f"lru={[n.last_access_time for n in by_lru]}"
        )


def stage_b2_tied_age_identical_score_no_tiebreak() -> None:
    """Post-#177: the eviction default does NOT tie-break by
    hit_count.  Two nodes at the SAME last_access_time get the SAME
    score regardless of hit_count (a deliberate non-feature — the
    cache never produces exact last_access_time ties anyway, spacing
    every node distinctly, and the float tie-break was non-functional).
    A future hit-aware refinement, if ever wanted, belongs in a real
    (tuple-keyed) scorer, not a lossy float sum."""
    a = _StubNode(last_access_time=100, hit_count=1)
    b = _StubNode(last_access_time=100, hit_count=50)
    if default_policy_score(a, _LAYER_HBM) != default_policy_score(b, _LAYER_HBM):
        raise StageFail(
            "same last_access_time must give the SAME eviction score "
            "(no hit_count tie-break in the eviction default — that is "
            "the write-through trigger's job, #178)"
        )


# ============================================================ C. plugin shape


def stage_c0_module_spec_resolvable() -> None:
    """The function is importable via the exact
    ``SGLANG_KV_POLICY_MODULE=pkg.mod:callable`` format that sglang's
    resolver uses, so flipping ``SGLANG_KV_POLICY_MODULE=baselines.
    sglang_adapter:default_policy_score`` works without further
    plumbing."""
    spec = "baselines.sglang_adapter:default_policy_score"
    mod_name, attr = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise StageFail(
            f"{spec} not importable — sglang resolver would fail to "
            f"find the default-policy scorer"
        )
    if not callable(fn):
        raise StageFail(f"{spec} resolved to non-callable: {fn!r}")
    # And it returns a float on a minimal node.
    n = _StubNode(last_access_time=0, hit_count=0)
    if not isinstance(fn(n, _LAYER_HBM), float):
        raise StageFail("resolved callable did not return float")


def stage_c1_sglang_resolver_loads_it() -> None:
    """Exercise sglang's actual policy-module resolver
    (``_resolve_kv_policy_module``) to confirm our spec is honoured
    end-to-end.  This catches a regression where the env-var format
    or import path changes silently."""
    import os
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.mem_cache.unified_radix_cache import (
        _default_eviction_score,
        _load_eviction_scorer,
    )
    spec = "baselines.sglang_adapter:default_policy_score"
    os.environ["SGLANG_KV_POLICY_MODULE"] = spec
    try:
        scorer = _load_eviction_scorer()
    finally:
        os.environ.pop("SGLANG_KV_POLICY_MODULE", None)
    # Resolver falls back to _default_eviction_score on failure;
    # post-fix it MUST return our scorer (a different function).
    if scorer is _default_eviction_score:
        raise StageFail(
            f"sglang resolver fell back to _default_eviction_score "
            f"for {spec!r} — load failed silently"
        )
    n = _StubNode(last_access_time=42, hit_count=0)
    out = scorer(n, _LAYER_HBM)
    if out != float(42):
        raise StageFail(
            f"resolved scorer returned unexpected value: got {out} "
            f"for last_access_time=42 hit=0 (expected 42.0)"
        )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 default_policy_score returns float",       stage_a0_returns_float),
    ("A1 hit_count=0 equals lru_score exactly",     stage_a1_zero_hits_equals_lru),
    ("B0 hit_count does NOT affect eviction score (incl 2^32+)", stage_b0_hit_count_does_not_affect_score),
    ("B1 uniform hit ordering matches LRU",         stage_b1_uniform_hit_count_matches_lru_order),
    ("B2 tied age → identical score (no tie-break)", stage_b2_tied_age_identical_score_no_tiebreak),
    ("C0 module:callable spec resolvable",          stage_c0_module_spec_resolvable),
    ("C1 sglang _resolve_kv_policy_module loads it", stage_c1_sglang_resolver_loads_it),
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
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT38 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT38 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""#253 regression gate — the aginfer eviction-scorer plugin must HONOR
--radix-eviction-policy on the default (no-aginfer-env) path.

Stock sglang keys the FullComponent eviction heap on
``eviction_strategy.get_priority(n)``; the aginfer fork routed it through the
pluggable ``_eviction_scorer`` whose default hardwired LRU (last_access_time),
silently ignoring lfu/slru/priority/fifo/mru. The fix (#253) makes
``FullComponent._evict_keyfn`` branch on ``cache._aginfer_value_aware``:
  * value-aware scorer attached -> the pluggable scorer (layer-keyed)
  * default                     -> stock ``eviction_strategy.get_priority``

This test exercises ``_evict_keyfn`` directly with a mock cache. Do-no-harm:
for the default ``lru`` policy get_priority == last_access_time == the old
default scorer (no-op); only non-LRU defaults change. Run:
    PYTHONPATH=. python verify/radix_eviction_policy/verify.py
"""
import os
import sys

_SGLANG_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
)
if _SGLANG_PY not in sys.path:
    sys.path.insert(0, _SGLANG_PY)

from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache_components.tree_component import EvictLayer
from sglang.srt.mem_cache.aginfer.cache_policy import _default_eviction_score
from sglang.srt.mem_cache.evict_policy import (
    LRUStrategy, LFUStrategy, FIFOStrategy, MRUStrategy, PriorityStrategy, SLRUStrategy,
)

GREEN, RED, RST = "\033[32m", "\033[31m", "\033[0m"
_fails = []


def check(name, cond):
    print(f"  {GREEN}PASS{RST}  {name}" if cond else f"  {RED}FAIL{RST}  {name}")
    if not cond:
        _fails.append(name)


class _Cache:
    pass


class _Self:
    pass


class _Node:
    def __init__(self, hit_count, last_access_time, creation_time=0.0, priority=0):
        self.hit_count = hit_count
        self.last_access_time = last_access_time
        self.creation_time = creation_time
        self.priority = priority


def keyfn_for(strategy, *, value_aware=False, scorer=None, layer=EvictLayer.DEVICE):
    s = _Self()
    s.cache = _Cache()
    s.cache.eviction_strategy = strategy
    s.cache._aginfer_value_aware = value_aware
    s.cache._eviction_scorer = scorer if scorer is not None else _default_eviction_score
    return FullComponent._evict_keyfn(s, layer)


def main():
    print("=== verify/radix_eviction_policy (#253: default path honors "
          "--radix-eviction-policy) ===")
    n = _Node(hit_count=3, last_access_time=7.0, creation_time=2.0, priority=5)

    # A — do-no-harm: default + LRU == last_access_time == the old default scorer
    k = keyfn_for(LRUStrategy())
    check("A0 lru default key == last_access_time (do-no-harm no-op)", k(n) == 7.0)
    check("A1 lru default key == old _default_eviction_score",
          k(n) == _default_eviction_score(n, EvictLayer.DEVICE))

    # B — THE FIX: non-LRU policies are now honored on the default path
    check("B0 lfu -> (hit_count, last_access_time)",
          keyfn_for(LFUStrategy())(n) == (3, 7.0))
    check("B1 fifo -> creation_time", keyfn_for(FIFOStrategy())(n) == 2.0)
    check("B2 mru -> -last_access_time", keyfn_for(MRUStrategy())(n) == -7.0)
    check("B3 priority -> (priority, last_access_time)",
          keyfn_for(PriorityStrategy())(n) == (5, 7.0))
    check("B4 slru -> (segment, last_access_time) tuple",
          isinstance(keyfn_for(SLRUStrategy())(n), tuple))
    # the bug would have returned 7.0 (LRU) for ALL of the above:
    check("B5 lfu key != the buggy LRU fallback", keyfn_for(LFUStrategy())(n) != 7.0)

    # C — value-aware path: the pluggable scorer, layer-keyed, unchanged
    seen = {}
    def scorer(node, layer):
        seen["layer"] = layer
        return 42.0
    kv = keyfn_for(LFUStrategy(), value_aware=True, scorer=scorer)
    check("C0 value-aware -> pluggable scorer (ignores eviction_strategy)", kv(n) == 42.0)
    check("C1 value-aware passes the layer through", seen.get("layer") == EvictLayer.DEVICE)
    kh = keyfn_for(LFUStrategy(), value_aware=True, scorer=scorer, layer=EvictLayer.HOST)
    kh(n)
    check("C2 host layer threaded to the scorer", seen.get("layer") == EvictLayer.HOST)

    # D — missing flag (defensive): treated as default, not value-aware
    s = _Self(); s.cache = _Cache()
    s.cache.eviction_strategy = LFUStrategy()
    s.cache._eviction_scorer = _default_eviction_score
    # no _aginfer_value_aware attribute at all
    check("D0 missing _aginfer_value_aware -> default (stock get_priority)",
          FullComponent._evict_keyfn(s, EvictLayer.DEVICE)(n) == (3, 7.0))

    print()
    if _fails:
        print(f"{RED}radix_eviction_policy FAIL — {len(_fails)} check(s): {_fails}{RST}")
        sys.exit(1)
    print(f"{GREEN}radix_eviction_policy PASS — all stages green (#253 default-path "
          f"policy honored; lru + value paths no-op){RST}")


if __name__ == "__main__":
    main()

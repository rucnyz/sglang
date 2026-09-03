"""disk_tier_migrate — P5 (safe subset) DISK-tier migrate semantics.

Offline (no GPU / no live sglang server) unit tests for the two new
``cache_hooks.apply_aginfer_migrations`` branches added when the P5 scope
was narrowed to a safe additive subset (per user decision: reuse stock
sglang's own async host->storage write-through path, never invent a new
storage-only radix-tree state, and never claim a delete/reclaim capability
sglang's storage backends (HiCacheFile / nixl / mooncake) don't expose):

  * ``add DISK``    -> ``cache.write_backup_storage(node)``, gated on
                       ``enable_storage`` + ``cache_controller`` +
                       ``node.backuped`` (host-backed).  Best-effort,
                       ADDITIVE ONLY -- never the unit's only copy.
  * ``remove DISK`` -> rejected up front with
                       ``disk_remove_unsupported_upstream`` (no delete API
                       upstream); pre-this-task it silently "succeeded"
                       (applied=1, transition="other") while doing NOTHING.

Uses a minimal duck-typed fake cache/tree (same pattern as verify/t27),
NOT a real ``UnifiedRadixCache`` -- these two branches don't touch device
pools, HiCache, or any GPU state, so a full server is unnecessary weight
for pinning this logic.  Methods deliberately NOT defined on the fake
(``_evict_to_host`` / ``_is_device_leaf`` / ``_cascade_evict`` / ...) act as
a tripwire: if a future change makes one of these actions fall through
into the removes-application code path it doesn't belong in, the missing
attribute raises loudly instead of the test silently passing.

Run:
    python dev/aginfer/verify/disk_tier_migrate/verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Tuple

_HERE = Path(__file__).resolve().parent
_SGLANG_PY = _HERE.parent.parent.parent.parent / "python"
if str(_SGLANG_PY) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PY))

from sglang.srt.mem_cache.aginfer import cache_hooks  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components import (  # noqa: E402
    BASE_COMPONENT_TYPE,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ fakes


class _Node:
    """Duck-typed UnifiedTreeNode: only the fields apply_aginfer_migrations'
    add-DISK / remove-DISK branches (+ the shared pre-checks they fall
    through) actually touch."""
    _counter = 0

    def __init__(self, *, hash_value=None, device=True, host=None,
                 backuped: bool = False):
        _Node._counter += 1
        self.id = _Node._counter
        self.hash_value = hash_value
        self.children: dict = {}
        self.component_data = [SimpleNamespace(
            value=[1, 2, 3] if device else None,
            host_value=host,
            lock_ref=0,
        )]
        self.backuped = backuped


class _FakeCache:
    def __init__(self, *, enable_storage: bool = False,
                 cache_controller: Any = None):
        self.root_node = _Node()
        self._aginfer_collision_seen: set = set()
        self._aginfer_migrate_skipped_counters: dict = {}
        self._aginfer_migrate_counters: dict = {}
        self.enable_storage = enable_storage
        self.cache_controller = cache_controller
        self.tree_components: list = []
        self.components = {BASE_COMPONENT_TYPE: object()}
        self._components_tuple: tuple = ()
        self.write_backup_storage_calls: List[Any] = []
        self.write_backup_calls: List[Any] = []
        self._write_backup_storage_raises: Exception | None = None

    def add_leaf(self, node: _Node) -> None:
        key = str(node.hash_value[-1]) if node.hash_value else f"node-{node.id}"
        self.root_node.children[key] = node

    def _aginfer_node_summary(self, node) -> dict:
        return {"id": node.id}

    def _update_evictable_leaf_sets(self, node) -> None:
        pass

    def _is_host_leaf(self, node) -> bool:
        return True

    def write_backup(self, node) -> int:
        self.write_backup_calls.append(node)
        node.component_data[BASE_COMPONENT_TYPE].host_value = [9]
        node.backuped = True
        return 10  # non-zero -> not write_through_declined:zero_tokens

    def write_backup_storage(self, node) -> None:
        self.write_backup_storage_calls.append(node)
        if self._write_backup_storage_raises is not None:
            raise self._write_backup_storage_raises


def _action(hash_: str, add: list[str], remove: list[str],
            action_id: str = "a") -> dict:
    return {"hash": hash_, "add_tiers": add, "remove_tiers": remove,
            "action_id": action_id}


def _run(cache: _FakeCache, actions: list[dict]) -> dict:
    return cache_hooks.apply_aginfer_migrations(cache, actions)


def _skip_reason(resp: dict, action_id: str) -> str:
    for s in resp["skipped"]:
        if s["action_id"] == action_id:
            return s["reason"]
    raise StageFail(f"no skip entry for action_id={action_id!r}; "
                    f"skipped={resp['skipped']!r}")


# ============================================================ stages


def stage_0_baseline_add_dram_still_works() -> None:
    """Sanity: the fake harness itself round-trips the PRE-EXISTING add=DRAM
    path (untouched by this change) before we trust it for the new branches."""
    cache = _FakeCache(cache_controller=object())
    n = _Node(hash_value=["u0"], device=True, host=None)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u0", ["DRAM"], [], "a0")])
    if resp["applied"] != 1:
        raise StageFail(f"baseline add=DRAM should apply; resp={resp!r}")
    if cache.write_backup_calls != [n]:
        raise StageFail("baseline: write_backup should have been called on n")


def stage_1_remove_disk_alone_rejected() -> None:
    """remove=[DISK] alone -> disk_remove_unsupported_upstream, applied=0.
    Note this fires even for a hash NOT in the tree (rejected before hash
    resolution) -- the whole point is "sglang can never honour this", which
    doesn't depend on whether the specific node exists."""
    cache = _FakeCache()
    resp = _run(cache, [_action("nonexistent", [], ["DISK"], "a1")])
    if resp["applied"] != 0:
        raise StageFail(f"remove=[DISK] must not apply; resp={resp!r}")
    reason = _skip_reason(resp, "a1")
    if reason != "disk_remove_unsupported_upstream":
        raise StageFail(f"expected disk_remove_unsupported_upstream; got {reason!r}")


def stage_2_remove_disk_combined_rejects_whole_action() -> None:
    """remove=[DISK, HBM] in ONE action -> the WHOLE action is rejected
    (disk_remove_unsupported_upstream), NOT a partial apply that silently
    drops the DISK part while removing HBM.  We prove "whole action
    rejected" by NOT defining _is_device_leaf/_evict_to_host on the fake:
    if the code fell through into applying the HBM removal, it would raise
    AttributeError instead of returning a clean skip."""
    cache = _FakeCache()
    n = _Node(hash_value=["u2"], device=True, host=[9])
    cache.add_leaf(n)
    resp = _run(cache, [_action("u2", [], ["DISK", "HBM"], "a2")])
    if resp["applied"] != 0:
        raise StageFail(f"combined remove with DISK must not apply; resp={resp!r}")
    reason = _skip_reason(resp, "a2")
    if reason != "disk_remove_unsupported_upstream":
        raise StageFail(f"expected disk_remove_unsupported_upstream; got {reason!r}")
    # HBM must be untouched (still has device value).
    if n.component_data[BASE_COMPONENT_TYPE].value is None:
        raise StageFail("HBM must NOT have been evicted by the rejected action")


def stage_3_add_disk_declined_no_storage_backend() -> None:
    """enable_storage=False -> disk_add_declined:no_storage_backend,
    write_backup_storage must never be called."""
    cache = _FakeCache(enable_storage=False, cache_controller=None)
    n = _Node(hash_value=["u3"], device=True, host=[9], backuped=True)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u3", ["DISK"], [], "a3")])
    if resp["applied"] != 0:
        raise StageFail(f"add=[DISK] w/o storage backend must not apply; resp={resp!r}")
    reason = _skip_reason(resp, "a3")
    if reason != "disk_add_declined:no_storage_backend":
        raise StageFail(f"expected disk_add_declined:no_storage_backend; got {reason!r}")
    if cache.write_backup_storage_calls:
        raise StageFail("write_backup_storage must NOT be called when declined")


def stage_4_add_disk_declined_no_cache_controller() -> None:
    """enable_storage=True but cache_controller=None (inconsistent config,
    defense-in-depth) -> same no_storage_backend decline."""
    cache = _FakeCache(enable_storage=True, cache_controller=None)
    n = _Node(hash_value=["u4"], device=True, host=[9], backuped=True)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u4", ["DISK"], [], "a4")])
    reason = _skip_reason(resp, "a4")
    if reason != "disk_add_declined:no_storage_backend":
        raise StageFail(f"expected disk_add_declined:no_storage_backend; got {reason!r}")


def stage_5_add_disk_declined_not_host_backed() -> None:
    """Storage available but the node is HBM-only (never DRAM-backed) ->
    disk_add_declined:not_host_backed -- write_backup_storage would have
    silently no-op'd here; we catch it BEFORE calling it so the daemon gets
    an honest reason instead of a fake applied=1."""
    cache = _FakeCache(enable_storage=True, cache_controller=object())
    n = _Node(hash_value=["u5"], device=True, host=None, backuped=False)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u5", ["DISK"], [], "a5")])
    if resp["applied"] != 0:
        raise StageFail(f"add=[DISK] on HBM-only node must not apply; resp={resp!r}")
    reason = _skip_reason(resp, "a5")
    if reason != "disk_add_declined:not_host_backed":
        raise StageFail(f"expected disk_add_declined:not_host_backed; got {reason!r}")
    if cache.write_backup_storage_calls:
        raise StageFail("write_backup_storage must NOT be called on a non-host-backed node")


def stage_6_add_disk_success_on_host_backed_node() -> None:
    """The happy path: storage available + node already DRAM-backed ->
    applied=1, write_backup_storage called exactly once with the node,
    and the P1 metrics counter tags it 'disk_backup' (not 'other')."""
    cache = _FakeCache(enable_storage=True, cache_controller=object())
    n = _Node(hash_value=["u6"], device=True, host=[9], backuped=True)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u6", ["DISK"], [], "a6")])
    if resp["applied"] != 1:
        raise StageFail(f"add=[DISK] on host-backed node should apply; resp={resp!r}")
    if resp["applied_hashes"] != ["u6"]:
        raise StageFail(f"applied_hashes should be ['u6']; got {resp['applied_hashes']!r}")
    if cache.write_backup_storage_calls != [n]:
        raise StageFail(
            f"write_backup_storage should be called exactly once with n; "
            f"calls={cache.write_backup_storage_calls!r}")
    if cache._aginfer_migrate_counters.get("disk_backup") != 1:
        raise StageFail(
            f"transition metrics should tag this 'disk_backup'; "
            f"counters={cache._aginfer_migrate_counters!r}")


def stage_7_add_disk_after_add_dram_same_action() -> None:
    """add=[DRAM, DISK] in ONE action on an HBM-only node: DRAM must apply
    FIRST (making the node host-backed) so the DISK branch's `node.backuped`
    gate sees the just-established host copy and actually fires --
    the ordering this whole feature depends on ('adds applied first', and
    within adds, DRAM before DISK in source order)."""
    cache = _FakeCache(enable_storage=True, cache_controller=object())
    n = _Node(hash_value=["u7"], device=True, host=None, backuped=False)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u7", ["DRAM", "DISK"], [], "a7")])
    if resp["applied"] != 1:
        raise StageFail(
            f"combined add=[DRAM,DISK] on HBM-only node should apply "
            f"(DRAM add makes it host-backed before the DISK check); "
            f"resp={resp!r}")
    if cache.write_backup_calls != [n]:
        raise StageFail("write_backup (DRAM) should have been called")
    if cache.write_backup_storage_calls != [n]:
        raise StageFail("write_backup_storage (DISK) should have been called after DRAM")


def stage_8_add_disk_raises_is_caught_and_skipped() -> None:
    """write_backup_storage raising must be caught and reported as a
    disk_backup_raised skip, not propagate and crash the batch."""
    cache = _FakeCache(enable_storage=True, cache_controller=object())
    cache._write_backup_storage_raises = RuntimeError("boom")
    n = _Node(hash_value=["u8"], device=True, host=[9], backuped=True)
    cache.add_leaf(n)
    resp = _run(cache, [_action("u8", ["DISK"], [], "a8")])
    if resp["applied"] != 0:
        raise StageFail(f"a raising write_backup_storage must not count as applied; resp={resp!r}")
    reason = _skip_reason(resp, "a8")
    if not reason.startswith("disk_backup_raised:RuntimeError"):
        raise StageFail(f"expected disk_backup_raised:RuntimeError:...; got {reason!r}")


def stage_9_disk_never_blocks_readd_already_present() -> None:
    """A unit that already got add=[DISK] applied can be re-requested with
    add=[DISK] again in a LATER action without hitting add_already_present
    -- DISK is deliberately never added to `current` residence (no
    synchronous confirmation, no delete API -> no residence guarantee to
    track), so a repeat best-effort backup is allowed (harmless, not
    blocked)."""
    cache = _FakeCache(enable_storage=True, cache_controller=object())
    n = _Node(hash_value=["u9"], device=True, host=[9], backuped=True)
    cache.add_leaf(n)
    resp1 = _run(cache, [_action("u9", ["DISK"], [], "a9-1")])
    if resp1["applied"] != 1:
        raise StageFail(f"first add=[DISK] should apply; resp={resp1!r}")
    resp2 = _run(cache, [_action("u9", ["DISK"], [], "a9-2")])
    if resp2["applied"] != 1:
        raise StageFail(
            f"re-requested add=[DISK] should apply again (not "
            f"add_already_present -- DISK isn't tracked in residence); "
            f"resp={resp2!r}")
    if len(cache.write_backup_storage_calls) != 2:
        raise StageFail(
            f"write_backup_storage should have fired twice; "
            f"calls={cache.write_backup_storage_calls!r}")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("0 baseline add=DRAM (harness sanity)",                     stage_0_baseline_add_dram_still_works),
    ("1 remove=[DISK] alone -> disk_remove_unsupported_upstream", stage_1_remove_disk_alone_rejected),
    ("2 remove=[DISK,HBM] combined -> whole action rejected",     stage_2_remove_disk_combined_rejects_whole_action),
    ("3 add=[DISK] no storage backend -> declined",               stage_3_add_disk_declined_no_storage_backend),
    ("4 add=[DISK] no cache_controller -> declined",              stage_4_add_disk_declined_no_cache_controller),
    ("5 add=[DISK] not host-backed -> declined",                  stage_5_add_disk_declined_not_host_backed),
    ("6 add=[DISK] host-backed -> applied (happy path)",          stage_6_add_disk_success_on_host_backed_node),
    ("7 add=[DRAM,DISK] combined, HBM-only start -> applied",     stage_7_add_disk_after_add_dram_same_action),
    ("8 write_backup_storage raises -> caught + skipped",         stage_8_add_disk_raises_is_caught_and_skipped),
    ("9 repeat add=[DISK] never add_already_present",             stage_9_disk_never_blocks_readd_already_present),
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
        print(_red(f"\ndisk_tier_migrate FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\ndisk_tier_migrate PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

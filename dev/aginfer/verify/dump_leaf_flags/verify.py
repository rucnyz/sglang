"""#212 — sglang dump emission of the three leaf flags, BOTH paths.

The /aginfer/state dump emits ``is_device_leaf`` / ``is_host_leaf`` /
``is_tree_leaf`` per unit (the structural predicates the daemon's
``migrate_candidates`` mirrors, #210).  Two code paths emit them:

  * the DICT path (``_dump_aginfer_state_dict``) — straight Python dict;
  * the BYTES path (``_dump_aginfer_state_bytes``) — a HAND-WRITTEN JSON
    byte buffer (the HTTP hot path), where a stray comma / quote / wrong
    field name would silently corrupt the wire and NOT turn any
    daemon-side stage red (those use synthetic ``_unit()`` dicts).

This round-trips a real ``UnifiedRadixCache`` walk over a small tree with
KNOWN leaf status, runs BOTH paths, ``json.loads`` them, and asserts the
three flags are (a) present, (b) byte-equal across the two paths, and
(c) match the node predicates (``_is_device_leaf`` / ``_is_host_leaf`` /
``len(children) == 0``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    BASE_COMPONENT_TYPE,
    UnifiedRadixCache,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- duck-typed tree node + component data (only what the dump reads) ----


class _CD:
    """One ``component_data[ct]`` slot."""
    def __init__(self, value=None, host_value=None,
                 lock_ref=0, host_lock_ref=0):
        self.value = value            # device (HBM) token ids, or None
        self.host_value = host_value  # host (DRAM) token ids, or None
        self.lock_ref = lock_ref
        self.host_lock_ref = host_lock_ref


class _N:
    """Duck-typed UnifiedTreeNode covering the dump + leaf-predicate reads."""
    _next_id = 0

    def __init__(self, *, value=None, host_value=None, children=None,
                 evicted=False, backuped=True, session_ids=None,
                 last_access_time=10, hit_count=1, lock_ref=0, host_lock_ref=0):
        _N._next_id += 1
        self.id = _N._next_id
        # component_data is a LIST indexed by component type (base only here).
        self.component_data = [_CD(value, host_value, lock_ref, host_lock_ref)]
        self.children = children or {}
        self.evicted = evicted
        self.backuped = backuped
        self.session_ids = session_ids
        self.last_access_time = last_access_time
        self.hit_count = hit_count
        self.hash_value = [f"hash-{self.id}"]


def _bare_cache(root) -> UnifiedRadixCache:
    """A UnifiedRadixCache with just enough wired for the two dump paths:
    the real ``_is_device_leaf`` / ``_is_host_leaf`` (which read node
    attrs only) run unchanged; the pool-usage assembly is stubbed (the
    test targets the per-unit leaf-flag emission, not pool accounting)."""
    c = UnifiedRadixCache.__new__(UnifiedRadixCache)
    c.root_node = root
    # program-state / metrics overlay inputs (empty → no programs to echo;
    # the test targets per-unit leaf flags, not program state).
    c._aginfer_program_states = {}
    c._aginfer_runtime_metrics = {}
    c._aginfer_hints = {}
    # BASE_COMPONENT_TYPE may be an enum/int index into component_data; the
    # node stores a single-element list, so index 0 must be the base.  If
    # BASE_COMPONENT_TYPE isn't 0, remap the node's list so cd[base] works.
    # pool-usage assembly is stubbed — the test targets the per-unit leaf
    # flags, not pool accounting.
    c._aginfer_pool_usage = lambda: {
        "HBM": {"subpools": {}}, "DRAM": {"subpools": {}},
        "DISK": {"subpools": {}}}
    c._aginfer_patch_dram_used = lambda pool_usage, dram_used_by_sp: None
    return c


def _units_by_hash(payload: dict):
    return {u["hash"]: u for u in payload["units"]}


def stage_roundtrip_leaf_flags() -> None:
    bpt = 2048
    base = BASE_COMPONENT_TYPE
    # Build a tree with KNOWN leaf status:
    #   A : HBM leaf, no children            → device_leaf=T, tree_leaf=T
    #   B : HBM, has child C with device val → device_leaf=F, tree_leaf=F
    #   C : HBM leaf (child of B)            → device_leaf=T, tree_leaf=T
    #   D : DRAM-only (evicted), no children → host_leaf=T, tree_leaf=T,
    #                                          device_leaf=F (no device value)
    toks = [0, 1, 2]
    C = _N(value=list(toks))
    B = _N(value=list(toks), children={1: C})
    A = _N(value=list(toks))
    D = _N(value=None, host_value=list(toks), evicted=True, backuped=True)
    root = _N(children={10: A, 20: B, 30: D})
    cache = _bare_cache(root)

    # If the base component type isn't index 0, the single-slot list above
    # would mis-index; assert the contract the fixture relies on.
    if int(base) != 0:
        raise StageFail(f"fixture assumes BASE_COMPONENT_TYPE indexes 0; "
                        f"got {base!r} — extend the node component_data list")

    dict_payload = cache._dump_aginfer_state_dict(bpt, "kv", {})
    bytes_payload = json.loads(
        bytes(cache._dump_aginfer_state_bytes(bpt, "kv", {})))

    du = _units_by_hash(dict_payload)
    bu = _units_by_hash(bytes_payload)

    # same unit set across the two paths
    if set(du) != set(bu):
        raise StageFail(f"unit hashes differ across paths: dict={set(du)} "
                        f"bytes={set(bu)}")
    if not du:
        raise StageFail("no units emitted — fixture produced an empty walk")

    LEAF_KEYS = ("is_device_leaf", "is_host_leaf", "is_tree_leaf")
    # ground truth from the real predicates on the live nodes
    truth = {
        A.hash_value[-1]: (cache._is_device_leaf(A), cache._is_host_leaf(A),
                           len(A.children) == 0),
        B.hash_value[-1]: (cache._is_device_leaf(B), cache._is_host_leaf(B),
                           len(B.children) == 0),
        C.hash_value[-1]: (cache._is_device_leaf(C), cache._is_host_leaf(C),
                           len(C.children) == 0),
        D.hash_value[-1]: (cache._is_device_leaf(D), cache._is_host_leaf(D),
                           len(D.children) == 0),
    }
    # sanity: the fixture really does exercise both True and False on each axis
    dev = {t[0] for t in truth.values()}
    host = {t[1] for t in truth.values()}
    tree = {t[2] for t in truth.values()}
    if not (dev == {True, False} and host == {True, False}
            and tree == {True, False}):
        raise StageFail(f"fixture must vary all three flags; got dev={dev} "
                        f"host={host} tree={tree}")

    for h, u_dict in du.items():
        u_bytes = bu[h]
        for k in LEAF_KEYS:
            if k not in u_dict or k not in u_bytes:
                raise StageFail(f"unit {h}: leaf key {k!r} missing "
                                f"(dict={k in u_dict}, bytes={k in u_bytes})")
            if type(u_dict[k]) is not bool or type(u_bytes[k]) is not bool:
                raise StageFail(f"unit {h}: leaf key {k!r} must be JSON bool; "
                                f"dict={u_dict[k]!r} bytes={u_bytes[k]!r}")
            if u_dict[k] != u_bytes[k]:
                raise StageFail(f"unit {h}: leaf key {k!r} DIFFERS across "
                                f"paths — dict={u_dict[k]} bytes={u_bytes[k]} "
                                f"(hand-written bytes-path typo?)")
        got = (u_dict["is_device_leaf"], u_dict["is_host_leaf"],
               u_dict["is_tree_leaf"])
        if got != truth[h]:
            raise StageFail(f"unit {h}: leaf flags {got} != node predicates "
                            f"{truth[h]} (is_device_leaf/is_host_leaf/tree_leaf)")

    print(_green("  [leaf-flags] dict ↔ bytes round-trip: all three flags "
                 "present, bool, byte-equal across paths, match predicates "
                 "(#212) OK"))


def main() -> int:
    print("=" * 64)
    print("verify/dump_leaf_flags (#212) — leaf-flag emission, both paths")
    print("=" * 64)
    try:
        stage_roundtrip_leaf_flags()
    except StageFail as e:
        print(_red(f"  FAIL: {e}"))
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        print(_red(f"  ERROR: {e}"))
        traceback.print_exc()
        return 1
    print(_green("ALL STAGES PASSED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

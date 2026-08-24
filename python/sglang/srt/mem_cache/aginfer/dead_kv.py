"""Deterministic Dead-KV reclamation for an ended aginfer program.

This module intentionally does not route SESSION_END through the normal
pressure-driven policy.  Once the orchestrator declares a program ended its
exclusive KV is unreachable, so it is safe (and desirable) to release it
immediately.  The implementation is a free function over UnifiedRadixCache so
it remains server-free unit-testable and keeps the core cache class thin.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.unified_cache_components import EvictLayer


def _node_depth(node, root) -> int:
    depth = 0
    while node is not root and getattr(node, "parent", None) is not None:
        depth += 1
        node = node.parent
    return depth


def _node_hash(cache, node) -> str:
    get_hash = getattr(cache, "_aginfer_unit_hash", None)
    if get_hash is not None:
        return get_hash(node)
    return f"node-{getattr(node, 'id', 'unknown')}"


def _busy_reason(cache, node) -> str | None:
    """Return a stable retry reason when asynchronous cache work owns *node*."""

    if getattr(node, "write_through_pending_id", None) is not None:
        return "busy_write_through"
    for pending in getattr(cache, "ongoing_write_through", {}).values():
        if getattr(pending, "node", None) is node or node in getattr(
            pending, "publish_nodes", ()
        ):
            return "busy_write_through"

    load_backs = getattr(cache, "ongoing_load_back", {})
    if getattr(node, "id", None) in load_backs or any(
        getattr(pending, "node", None) is node for pending in load_backs.values()
    ):
        return "busy_load_back"

    for pending in getattr(cache, "ongoing_backup", {}).values():
        if pending and pending[0] is node:
            return "busy_storage_backup"
    for pending in getattr(cache, "ongoing_prefetch", {}).values():
        if getattr(pending, "anchor_node", None) is node:
            return "busy_storage_prefetch"

    if any(
        getattr(component_data, "lock_ref", 0) > 0
        or getattr(component_data, "host_lock_ref", 0) > 0
        for component_data in getattr(node, "component_data", ())
    ):
        return "locked"
    return None


def aginfer_program_busy(cache, program_id: str) -> list[dict[str, Any]]:
    """Return local blockers that make a program unsafe to reclaim now.

    The scheduler reduces this predicate across the TP/CP group before any
    rank mutates its tree.  That two-phase check prevents one shard from
    freeing a node while a sibling shard still has an asynchronous copy or
    lock outstanding.
    """

    root = getattr(cache, "root_node", None)
    if root is None:
        return []

    blocked: list[dict[str, Any]] = []
    stack = list(getattr(root, "children", {}).values())
    while stack:
        node = stack.pop()
        stack.extend(getattr(node, "children", {}).values())
        if program_id not in getattr(node, "session_ids", ()):
            continue
        reason = _busy_reason(cache, node)
        if reason is not None:
            blocked.append(
                {
                    "hash": _node_hash(cache, node),
                    "node_id": getattr(node, "id", None),
                    "reason": reason,
                }
            )
    return blocked


def end_aginfer_program(cache, program_id: str) -> dict[str, Any]:
    """Mark ``program_id`` ENDED and reclaim its exclusive KV immediately.

    Holder removal and physical deletion are deliberately one operation:

    * shared nodes only lose ``program_id`` and remain resident;
    * exclusively-held nodes are peeled leaf-to-root and have all device/host
      component allocations released;
    * locked or asynchronously-busy nodes retain the holder and are returned in
      ``skipped`` so the scheduler (or an idempotent caller) can retry safely.

    Keeping the holder on a skipped exclusive node is important.  It is the
    durable retry marker; removing it before the physical delete would make a
    later call unable to rediscover the dead node.
    """

    if not isinstance(program_id, str) or not program_id:
        return {
            "ok": False,
            "reason": "program_id must be a non-empty string",
            "status": "invalid",
            "state_changed": 0,
            "matched_nodes": 0,
            "holders_removed": 0,
            "released_nodes": 0,
            "released_hbm_tokens": 0,
            "released_dram_tokens": 0,
            "remaining_nodes": 0,
            "skipped": [],
        }

    setter = getattr(cache, "set_aginfer_program_state", None)
    root = getattr(cache, "root_node", None)
    if setter is None or root is None:
        return {
            "ok": False,
            "reason": f"unsupported_tree_cache:{type(cache).__name__}",
            "status": "unsupported",
            "state_changed": 0,
            "matched_nodes": 0,
            "holders_removed": 0,
            "released_nodes": 0,
            "released_hbm_tokens": 0,
            "released_dram_tokens": 0,
            "remaining_nodes": 0,
            "skipped": [],
        }

    ok, reason, state_changed = setter(
        pid=program_id,
        state="ENDED",
        pre_pause_state=None,
    )
    if not ok:
        return {
            "ok": False,
            "reason": reason,
            "status": "invalid",
            "state_changed": int(state_changed),
            "matched_nodes": 0,
            "holders_removed": 0,
            "released_nodes": 0,
            "released_hbm_tokens": 0,
            "released_dram_tokens": 0,
            "remaining_nodes": 0,
            "skipped": [],
        }

    # Snapshot candidates before mutation.  A node is processed after all of
    # its descendants so deleting a leaf can expose its parent in this call.
    candidates = []
    stack = list(getattr(root, "children", {}).values())
    while stack:
        node = stack.pop()
        stack.extend(getattr(node, "children", {}).values())
        if program_id in getattr(node, "session_ids", ()):
            candidates.append(node)
    candidates.sort(key=lambda node: _node_depth(node, root), reverse=True)

    matched_nodes = len(candidates)
    holders_removed = 0
    released_nodes = 0
    released_hbm_tokens = 0
    released_dram_tokens = 0
    released_hashes: list[str] = []
    skipped: list[dict[str, Any]] = []

    def skip(node, why: str) -> None:
        skipped.append(
            {
                "hash": _node_hash(cache, node),
                "node_id": getattr(node, "id", None),
                "reason": why,
            }
        )

    for node in candidates:
        holders = getattr(node, "session_ids", None)
        if holders is None or program_id not in holders:
            continue

        # A shared prefix is still reachable.  Only release the ended holder;
        # the final remaining holder's SESSION_END will reclaim it later.
        if len(holders) > 1:
            holders.discard(program_id)
            holders_removed += 1
            continue

        children = list(getattr(node, "children", {}).values())
        if children:
            # A child that still carries this pid is an exclusive descendant
            # that could not be removed (normally locked/busy).  Preserve this
            # holder too, so a retry can peel the entire chain.  Children that
            # no longer carry the pid are shared or untracked, so this prefix
            # is not provably dead and only its holder metadata is removed.
            if any(
                program_id in getattr(child, "session_ids", ()) for child in children
            ):
                skip(node, "child_pending")
            else:
                holders.discard(program_id)
                holders_removed += 1
            continue

        why = _busy_reason(cache, node)
        if why is not None:
            skip(node, why)
            continue

        node_hash = _node_hash(cache, node)
        base_data = node.component_data[0] if node.component_data else None
        had_device = base_data is not None and base_data.value is not None
        had_host = base_data is not None and base_data.host_value is not None
        record_remove = getattr(cache, "_record_remove_event", None)
        if record_remove is not None:
            if had_device:
                record_remove(node, medium=StorageMedium.GPU)
            if had_host:
                record_remove(node, medium=StorageMedium.CPU)
        for component in getattr(cache, "_components_tuple", ()):
            device_freed, host_freed = cache._evict_component_and_detach_lru(
                node,
                component,
                target=EvictLayer.ALL,
                tracker=None,
            )
            released_hbm_tokens += int(device_freed)
            released_dram_tokens += int(host_freed)

        parent = node.parent
        cache.evictable_device_leaves.discard(node)
        cache.evictable_host_leaves.discard(node)
        cache._remove_leaf_from_parent(node)
        if parent is not None:
            cache._update_evictable_leaf_sets(parent)
        holders.discard(program_id)
        holders_removed += 1
        released_nodes += 1
        released_hashes.append(node_hash)

    remaining_nodes = sum(
        1 for node in candidates if program_id in getattr(node, "session_ids", ())
    )
    if skipped:
        status = "partial"
    elif released_nodes or holders_removed:
        status = "applied"
    else:
        status = "already_absent"

    return {
        "ok": True,
        "reason": "ok",
        "status": status,
        "state_changed": int(state_changed),
        "matched_nodes": matched_nodes,
        "holders_removed": holders_removed,
        "released_nodes": released_nodes,
        "released_hashes": released_hashes,
        "released_hbm_tokens": released_hbm_tokens,
        "released_dram_tokens": released_dram_tokens,
        "remaining_nodes": remaining_nodes,
        "skipped": skipped,
    }

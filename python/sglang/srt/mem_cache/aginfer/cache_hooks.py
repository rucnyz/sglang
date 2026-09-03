"""aginfer cache hooks (refactor #251 Stage A.2): hint-table + value-eviction
scorer logic as free functions over a UnifiedRadixCache. The upstream cache
keeps thin delegators. Default path is byte-for-byte stock LRU (do-no-harm)."""
from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Optional
from sglang.srt.mem_cache.aginfer.cache_policy import (
    _default_eviction_score,
    _load_eviction_scorer,
    _load_write_through_policy,
    _AGINFER_HINT_SCORER_SPEC,
    _AGINFER_WRITE_THROUGH_SPEC,
    _AGINFER_BIRTH_PHAT,
    _AGINFER_BIRTH_LAMBDA,
    _AGINFER_BIRTH_STAMP,
)
from sglang.srt.mem_cache.unified_cache_components import (  # apply_aginfer_migrations deps
    EvictLayer,
    BASE_COMPONENT_TYPE,
)

if TYPE_CHECKING:  # annotation-only; runtime import would be circular (urc imports us)
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

logger = logging.getLogger("sglang.srt.mem_cache.unified_radix_cache")

def set_aginfer_hints(cache, hints: "list") -> tuple:
    """T40 (#184, DESIGN §6 PUT /aginfer/hints + §10 overwrite-by-
    stamp): apply a batch of daemon-pushed V_u hints.

    Each ``hint`` is ``{"hash", "p_hat", "lambda", "stamp"}``
    (already type-validated by the HTTP layer's
    ``_validate_hints_body``; re-checked here so a direct caller /
    a future non-HTTP path cannot poison the table).

    Overwrite-by-stamp: a hash is written only when the incoming
    ``stamp`` is STRICTLY newer than the stored one.  An equal
    stamp is an idempotent no-op (the daemon re-pushed the same
    D_t against the same sglang time_counter — DESIGN §10 R2); an
    older stamp is a stale out-of-order delivery and is dropped
    (it must not clobber a newer value).

    Returns ``(ok, reason, applied)`` where ``applied`` counts the
    hashes whose stamp advanced (so an idempotent re-push of a
    whole batch returns ``applied == 0``).
    """
    if not isinstance(hints, list):
        return (False, f"hints must be a list; got {type(hints).__name__}", 0)
    applied = 0
    for h in hints:
        if not isinstance(h, dict):
            return (False, f"hint must be an object; got {h!r}", 0)
        uhash = h.get("hash")
        if not isinstance(uhash, str) or not uhash:
            return (False, f"hint hash must be non-empty string; got {uhash!r}", 0)
        try:
            stamp = int(h["stamp"])
            p_hat = float(h["p_hat"])
            lam = float(h["lambda"])
            # S2 (holder-count): preserve the daemon's n_holders through the
            # storage layer — without this it was dropped here, so hint_v_u
            # never saw it and the holder-count boost was inert. Optional for
            # back-compat (absent ⇒ 0 ⇒ _value falls back to max(1, ...)).
            n_holders = int(h.get("n_holders", 0) or 0)
        except (KeyError, TypeError, ValueError) as exc:
            return (False, f"hint {uhash!r}: bad numeric field ({exc})", 0)
        existing = cache._aginfer_hints.get(uhash)
        if existing is not None and stamp <= existing["stamp"]:
            # equal stamp = idempotent no-op; older = stale drop.
            continue
        cache._aginfer_hints[uhash] = {
            "p_hat": p_hat, "lambda": lam, "stamp": stamp,
            "n_holders": n_holders,
        }
        applied += 1
    return (True, "ok", applied)

def get_aginfer_hint(cache, uhash: str) -> "Optional[dict]":
    """T40 (#184): read the current hint entry for a unit hash, or
    None if the daemon has not pushed one (and no birth-seed
    exists yet — birth-seeding is a separate task).  Returns the
    stored ``{"p_hat", "lambda", "stamp"}`` dict."""
    return cache._aginfer_hints.get(uhash)

def clear_aginfer_hint(cache, uhash: str) -> bool:
    """T40 (#184, DESIGN §10 'Hint clear ordering'): drop the hint
    entry for a unit on its death (eviction / drop).  Returns True
    if an entry was removed.  Called by ``_remove_leaf_from_parent``
    (T27 #188) — the single death/commit chokepoint — AFTER the node
    is detached (scorer read → evict commit → hint clear)."""
    return cache._aginfer_hints.pop(uhash, None) is not None

# ---- T27 (#188): hint-table CONSUMER (DESIGN §3 / §10) ----

def _aginfer_unit_hash(cache, node) -> str:
    """The hint-table key for a node — IDENTICAL to the unit ``hash``
    the daemon receives in ``/aginfer/state`` (``hash_value[-1]`` or
    the ``node-{id}`` fallback for transient nodes).  Keeping this
    in lockstep with ``_dump_aginfer_state_*`` is what lets the
    scorer / clear find the entry the daemon PUT."""
    hv = node.get_last_hash_value()
    return hv if hv is not None else f"node-{node.id}"

def _init_aginfer_eviction_scoring(cache) -> None:
    """Resolve the eviction scorer (T27 #188 extends #177's
    ``_load_eviction_scorer``).  The sentinel
    ``SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u`` selects the hint-
    AWARE scorer — a cache-bound method reading ``_aginfer_hints``
    (a free ``module:callable`` can't reach the cache's dict).  Any
    other spec resolves through ``_load_eviction_scorer`` (default
    LRU / a custom module).  Always sets ``_aginfer_hint_aware`` so
    birth-seeding can gate on it.  Failure to import the adapter's
    ``hint_v_u`` falls back to LRU (logged) rather than crashing
    launch."""
    cache._aginfer_hint_aware = False
    cache._aginfer_hint_v_u_fn = None
    spec = os.environ.get("SGLANG_KV_POLICY_MODULE", "").strip()
    if spec == _AGINFER_HINT_SCORER_SPEC:
        try:
            # in-package import (refactor #251): sglang_adapter now lives in
            # this package, so this no longer depends on dev/aginfer being on
            # sys.path. Behaviour-identical — the old `baselines.sglang_adapter`
            # is a shim re-exporting this same module.
            from sglang.srt.mem_cache.aginfer.sglang_adapter import hint_v_u
            cache._aginfer_hint_v_u_fn = hint_v_u
            cache._eviction_scorer = cache._aginfer_eviction_score
            cache._aginfer_hint_aware = True
            logger.info("[aginfer] kv_policy_loaded=%s", spec)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[aginfer] kv_policy_loaded=default_lru "
                "(load_failed:%r exception=%s)", spec, e,
            )
            cache._eviction_scorer = _default_eviction_score
    else:
        cache._eviction_scorer = _load_eviction_scorer()
    # #253: "value-aware" iff a NON-default scorer is attached.  On the default
    # path the eviction heap routes through ``eviction_strategy.get_priority``
    # (stock — honors --radix-eviction-policy), NOT this LRU-only default; the
    # swa + full components branch on this flag (single source of truth).
    cache._aginfer_value_aware = cache._eviction_scorer is not _default_eviction_score

def _aginfer_eviction_score(cache, node, layer) -> float:
    """T27 (#188): hint-aware eviction heap key.  Looks up the
    node's daemon hint and computes the paper-§7 V_u via the adapter
    (one V_u formula — no reimplementation/drift).  Absent hint →
    the adapter falls back to the local hits/age derivation (never
    bare LRU).  Single-threaded scheduler → the dict read needs no
    lock (DESIGN §10 'Hint atomicity' satisfied by serialisation)."""
    uhash = cache._aginfer_unit_hash(node)
    hint = cache._aginfer_hints.get(uhash)
    return cache._aginfer_hint_v_u_fn(node, layer, hint)

def _aginfer_hint_should_write_through(cache, node, threshold) -> bool:
    """Hint-aware WRITE-THROUGH (cache-bound, mirrors the hint_v_u eviction
    scorer): write a unit through to the host/DRAM tier when the daemon's hint
    marks it reuse-imminent (high p_hat), so an evicted-then-reused prefix is
    RETAINED (cheap DRAM load-back) instead of dropped (recompute) — the S1
    value-aware retention the hit-count default cannot do under churn (nodes
    evict before any hit). Absent hint -> hit-count default (do-no-harm)."""
    hint = cache._aginfer_hints.get(cache._aginfer_unit_hash(node))
    if hint is not None:
        try:
            return float(hint.get("p_hat", 0.0)) >= 0.5
        except Exception:  # noqa: BLE001
            pass
    return int(node.hit_count) >= int(threshold)

def _init_aginfer_write_through(cache) -> None:
    """Resolve the write-through trigger policy (#178), the twin of
    ``_init_aginfer_eviction_scoring``.  Default = ``_default_should_write_through``
    (hit_count ≥ threshold, byte-for-byte stock); the sentinel
    ``SGLANG_WRITE_THROUGH_MODULE=aginfer:hint_write_through`` selects the
    cache-bound hint-aware trigger.  do-no-harm: the default path is unchanged.
    Sentinel checked FIRST (mirrors the eviction twin) so selecting the hint
    policy does not spuriously call — and log a load_failed WARNING for — the
    module loader."""
    if os.environ.get("SGLANG_WRITE_THROUGH_MODULE", "").strip() == _AGINFER_WRITE_THROUGH_SPEC:
        cache._write_through_policy = cache._aginfer_hint_should_write_through
        logger.info("[aginfer] write_through_loaded=%s", _AGINFER_WRITE_THROUGH_SPEC)
    else:
        cache._write_through_policy = _load_write_through_policy()

def _aginfer_seed_birth(cache, node) -> None:
    """T27 (#188, DESIGN §3 'Hint table covers every live unit'):
    seed a fresh-access entry (``p_hat = _AGINFER_BIRTH_PHAT``, a low reuse
    prior — NOT 1.0; see the constant) for a newborn unit so the scorer
    never sees an absent hint and the table tracks the live-unit set.  No-op
    unless hint-aware.  Never clobbers an
    existing (daemon-pushed or already-seeded) entry — overwrite-by-
    stamp is the daemon's job (#184); birth only fills the gap."""
    if not getattr(cache, "_aginfer_hint_aware", False):
        return
    uhash = cache._aginfer_unit_hash(node)
    if uhash in cache._aginfer_hints:
        return
    cache._aginfer_hints[uhash] = {
        "p_hat": _AGINFER_BIRTH_PHAT,
        "lambda": _AGINFER_BIRTH_LAMBDA,
        # Floor stamp (#188 audit C7): below any real daemon stamp so
        # the daemon's FIRST refinement always wins, even if the
        # counter didn't advance between birth and that unit's first
        # dump (equal-stamp would be skipped by overwrite-by-stamp).
        "stamp": _AGINFER_BIRTH_STAMP,
    }



# ---------------------------------------------------------------------------
# apply_aginfer_migrations (#251 Stage A.2 inc3): the §7 migrate EXECUTOR.
# ---------------------------------------------------------------------------
def apply_aginfer_migrations(cache, actions: list[dict]) -> dict:
    """Apply a batch of DESIGN §6 residence-set migrate actions.

    Each action: ``{"hash", "add_tiers": [...], "remove_tiers": [...],
                    "action_id": "<correlator>"}``.  See verify/t20/
    README.md for the full per-tier semantics + skip-reason table.

    Returns ``{"applied": int, "applied_hashes": [...],
                "skipped": [{"hash", "action_id", "reason"}, ...]}``.

    Order within one action: ``add_tiers`` applied first, then
    ``remove_tiers``.  Synthesise the new copy BEFORE freeing the
    old one so the ``{HBM} → {DRAM}`` path doesn't drop data
    between write_through and the device evict.

    Per-tier dispatch:
      add HBM   → ``load_back`` (host→device promote)
      add DRAM  → ``write_backup`` (device→host backup)
      add DISK  → ``write_backup_storage`` (P5 safe-subset: reuses stock
                  sglang's own async host→storage write-through path;
                  ADDITIVE ONLY, best-effort, requires the node to
                  already be DRAM-backed from a PRIOR action — see the
                  "Apply adds first" section below for the full rationale)
      remove HBM   → ``evict_component(target=DEVICE)``
      remove DRAM  → ``evict_component(target=HOST)``
      remove DISK  → rejected up front (``disk_remove_unsupported_upstream``):
                  none of sglang's storage backends (file/nixl/mooncake)
                  expose a delete API, so aginfer cannot honour a DISK
                  removal request — reporting a fake "applied" success
                  while doing nothing would be worse than an honest skip.
      remove all current tiers → DROP (full evict + tree leaf removal)

    Two more combinations are rejected up front for the same
    "no fake success" reason, this time to avoid racing the async
    storage write against a host-buffer mutation in the SAME batch
    (``disk_add_conflicts_with_dram_remove`` / ``_dram_add``) — see the
    inline comment where they're checked, a few lines below the
    ``remove DISK`` rejection.
    """
    # Build hash → node lookup with one DFS (O(N), same cost as
    # state walk).  Two hash schemes: HiCache-finalised nodes have
    # a real hash_value (hex SHA-256); transient nodes fall back
    # to ``node-<id>`` where ``id`` is a class-level monotonic
    # counter that is NEVER recycled (UnifiedTreeNode.counter
    # strictly increases), so the fallback name is also stable
    # for the daemon's lifetime.
    #
    # T24 (DESIGN §10) HASH_COLLISION detection lives here: if two
    # distinct radix nodes ever map to the same hash key, fire the
    # HASH_COLLISION webhook.  Probability is < 10⁻²² at any tree
    # size aginfer encounters, but the check is amortised free
    # (one extra dict-membership test per node).
    #
    # Detection is dedupe-guarded by the instance-level
    # ``_aginfer_collision_seen`` set.
    hash_to_node: dict[str, UnifiedTreeNode] = {}
    hash_collisions: list[dict] = []
    stack = [cache.root_node]
    root = cache.root_node
    while stack:
        node = stack.pop()
        if node is not root:
            if node.hash_value:
                key = str(node.hash_value[-1])
            else:
                key = f"node-{node.id}"
            # T24 collision check (lazy: only checks pairs that
            # actually share a key, which is O(1) per node).
            existing = hash_to_node.get(key)
            if existing is not None and existing is not node:
                pair = (
                    (existing.id, node.id) if existing.id < node.id
                    else (node.id, existing.id)
                )
                if pair not in cache._aginfer_collision_seen:
                    cache._aginfer_collision_seen.add(pair)
                    logger.warning(
                        "[aginfer] HASH_COLLISION key=%s nodes "
                        "%d vs %d (firing webhook)",
                        key, existing.id, node.id,
                    )
                    hash_collisions.append({
                        "key": key,
                        "node_a_summary": cache._aginfer_node_summary(existing),
                        "node_b_summary": cache._aginfer_node_summary(node),
                    })
            hash_to_node[key] = node
        stack.extend(node.children.values())

    # S1 whole-chain demote (DESIGN §7 "demote the session tail"): a
    # remove-HBM on a non-device-leaf becomes VALID once its device-children
    # are removed first — the device-leaf invariant peels a chain leaf-inward
    # (empirically: a 4 MB internal prefix reaches DRAM after its small
    # descendants are peeled).  When the daemon sends a program's whole
    # exclusive chain in one batch (so the bulk idle prefix can be demoted
    # during a tool gap), process the remove-HBM actions DEEPEST-NODE-FIRST
    # so each parent has already become a device-leaf when its own remove is
    # reached.  Stable for non-remove-HBM actions and within equal depth, so
    # single-action / non-chain batches are unaffected.
    def _node_depth(n: "UnifiedTreeNode") -> int:
        d, cur = 0, n
        while cur is not root and getattr(cur, "parent", None) is not None:
            cur = cur.parent
            d += 1
        return d

    def _peel_key(item):
        idx, a = item
        nd = hash_to_node.get(a["hash"])
        if nd is not None and "HBM" in set(a.get("remove_tiers") or []):
            return (0, -_node_depth(nd), idx)   # remove-HBM: deepest first
        return (1, 0, idx)                        # others: original order

    actions = [a for _, a in sorted(enumerate(actions), key=_peel_key)]

    applied = 0
    applied_hashes: list[str] = []
    skipped: list[dict] = []
    # Tracks nodes we've already mutated in this batch.  A duplicate
    # hash in `actions` would re-enter the DROP / evict code with
    # a stale view of the node (cd.value is left dangling because
    # FullComponent.evict_component DEFERS the ``cd.value = None``
    # to a later trigger -- see _cascade_evict).  The second pass
    # would then crash inside _remove_leaf_from_parent or double-
    # free the device buffer.
    acted_node_ids: set[int] = set()
    # S1 whole-chain demote: a remove-HBM device-evict nulls the node's
    # device cd.value VIA TOMBSTONE (deferred — see `_evict_to_host`), so a
    # parent processed LATER in this same batch still sees the just-evicted
    # child's stale `cd.value is not None` and fails the device-leaf guard
    # (`remove_hbm_not_device_leaf:dev_children`).  Track nodes whose HBM
    # was removed in THIS batch and treat them as device-cleared when
    # re-deriving a parent's device-leaf-ness — so an exclusive chain peels
    # leaf-inward in one batch (deepest-first sort guarantees the child is
    # evicted before its parent, so the descendant's device KV is already
    # being freed → evicting the parent now is invariant-safe).
    batch_removed_hbm: set[int] = set()

    def _is_device_leaf_in_batch(node) -> bool:
        if cache._is_device_leaf(node):
            return True
        ct = BASE_COMPONENT_TYPE
        if node is cache.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        for child in node.children.values():
            if (child.component_data[ct].value is not None
                    and child.id not in batch_removed_hbm):
                return False
        return True

    components = cache._components_tuple
    base = BASE_COMPONENT_TYPE
    _VALID_TIERS = {"HBM", "DRAM", "DISK"}

    def _skip(h, action_id, reason):
        skipped.append({"hash": h, "action_id": action_id, "reason": reason})
        # P1 metrics: bucket by the reason's first ':'-delimited token so a
        # detail-bearing reason (e.g. "promote_raised:ValueError:...:msg")
        # doesn't explode into a distinct counter key per exception message.
        bucket = reason.split(":", 1)[0]
        counters = cache._aginfer_migrate_skipped_counters
        counters[bucket] = counters.get(bucket, 0) + 1

    for action in actions:
        # Direct subscript: every action is contractually
        # required to carry hash + add_tiers + remove_tiers +
        # action_id (DESIGN §6 wire payload; verify/t20 enforces).
        # Malformed POSTs surface as 500 → ops sees the schema
        # break instead of silently being coerced to a noop.
        h = action["hash"]
        action_id = action["action_id"]
        add_tiers = set(action["add_tiers"])
        remove_tiers = set(action["remove_tiers"])

        # Validate tier strings.
        unknown_tiers = (add_tiers | remove_tiers) - _VALID_TIERS
        if unknown_tiers:
            _skip(h, action_id,
                  f"unknown_tier:{','.join(sorted(unknown_tiers))}")
            continue

        # remove DISK → reject up front (P5 safe-subset).  None of
        # sglang's storage backends (HiCacheFile / nixl / mooncake)
        # expose a delete API from the radix cache's perspective, so
        # aginfer cannot make this happen.  Pre-#252 this action would
        # silently pass every downstream check (DISK is never in
        # `current`, so `remove_tiers - current` treated it as
        # already-absent) and get counted as `applied=1`/
        # `transition="other"` while doing NOTHING — an honest skip is
        # strictly better than that fake success.
        if "DISK" in remove_tiers:
            _skip(h, action_id, "disk_remove_unsupported_upstream")
            continue

        # Review (PR #4, discussion_r3921269467): reject two same-action
        # combinations that would race the async storage backup against a
        # host-buffer mutation, rather than silently risking a use-after-
        # free / stale read on the storage backend's transfer thread:
        #
        #   add=[DISK], remove=[DRAM]: `write_backup_storage()` (below)
        #   starts an async H->Storage read AND `inc_host_lock_ref`s the
        #   node, but that lock is only drained by the non-blocking
        #   `drain_storage_control_queues` (a later scheduler tick) --
        #   `writing_check()` (the "drain pending write_through before
        #   removes" call a few lines down) only awaits D->H
        #   `ongoing_write_through` acks, NOT H->Storage `ongoing_backup`
        #   ones. Since the leaf-invariant check above (`_is_host_leaf`)
        #   runs BEFORE this action's adds, it sees host_lock_ref==0 and
        #   passes, then `evict_component(target=HOST)` (in "Apply
        #   removes") unconditionally frees `host_value` while the storage
        #   read may still be in flight against that same buffer.
        #
        #   add=[DRAM, DISK] together: `write_backup(node)` (device->host)
        #   returns as soon as the host buffer is ALLOCATED and the tree
        #   is committed (`node.backuped` becomes true synchronously), but
        #   the actual byte copy is async (drained by `writing_check`,
        #   which only runs later, gated on `remove_tiers` being
        #   non-empty). Starting `write_backup_storage()` immediately
        #   after would read a host buffer whose D->H copy has not
        #   necessarily finished yet.
        #
        # Both are same-batch ordering hazards, not something a per-node
        # leaf/lock check can catch with the primitives sglang exposes
        # today (no synchronous "await this node's backup ack" API) --
        # reject rather than risk silent corruption; the daemon can just
        # re-request `add DISK` alone in a LATER action once the DRAM leg
        # has actually landed (state_dump's next snapshot will show it).
        if "DISK" in add_tiers and "DRAM" in remove_tiers:
            _skip(h, action_id, "disk_add_conflicts_with_dram_remove")
            continue
        if "DISK" in add_tiers and "DRAM" in add_tiers:
            _skip(h, action_id, "disk_add_conflicts_with_dram_add")
            continue

        # Resolve hash.
        node = hash_to_node.get(h)
        if node is None:
            _skip(h, action_id, "not_in_tree")
            continue
        if node.id in acted_node_ids:
            _skip(h, action_id, "already_acted_this_batch")
            continue

        # No-op action: refuse rather than silently apply nothing.
        if not add_tiers and not remove_tiers:
            _skip(h, action_id, "noop_action")
            continue

        # Compute current residence from component_data.
        cd = node.component_data[base]
        has_device = cd.value is not None and len(cd.value) > 0
        has_host = cd.host_value is not None and len(cd.host_value) > 0
        current: set[str] = set()
        if has_device:
            current.add("HBM")
        if has_host:
            current.add("DRAM")
        # DISK is deliberately NEVER added to current_residence: sglang's
        # write_backup_storage is a fire-and-forget async write with no
        # synchronous confirmation and no delete API on any backend, so
        # aginfer has no basis to claim a residence GUARANTEE for it (see
        # the "add DISK" handling below for the full rationale). This also
        # means a re-requested "add DISK" is never blocked by
        # `add_already_present` — harmless, since write_backup_storage is
        # idempotent-ish (re-keys the same hash on the same backend).

        # Validate add: tiers must not already be in residence.
        already_in = add_tiers & current
        if already_in:
            _skip(h, action_id,
                  f"add_already_present:{','.join(sorted(already_in))}")
            continue

        # Validate remove: tiers must be in current residence.
        # (DISK can no longer reach here — rejected above.)
        missing = remove_tiers - current
        if missing:
            _skip(h, action_id,
                  f"remove_already_absent:{','.join(sorted(missing))}")
            continue

        # Will the unit be fully removed (post-add residence ⊆ remove)?
        post_residence = (current | add_tiers) - remove_tiers
        is_full_drop = not post_residence
        is_leaf = len(node.children) == 0
        if is_full_drop and not is_leaf:
            _skip(h, action_id, "remove_not_leaf")
            continue
        # Per-tier leaf invariant: sglang's `inc_lock_ref` walks
        # from a backed-up node up to root and asserts every
        # ancestor has cd.value (DEVICE).  If we device-evict a
        # non-leaf node, any later write_backup on that node's
        # descendants trips the assert + crashes the scheduler.
        # Same logic for HOST evict on a host-non-leaf.  Daemon's
        # policy SHOULD only emit migrate actions for leaves —
        # this is a defense-in-depth guard.
        if "HBM" in remove_tiers and not _is_device_leaf_in_batch(node):
            # Diagnostic detail: WHY is it not a device-leaf right now?
            _ct = BASE_COMPONENT_TYPE
            _locked = any(cd.lock_ref > 0 for cd in node.component_data)
            _dev_children = sum(
                1 for c in node.children.values()
                if c.component_data[_ct].value is not None)
            _why = ("locked" if _locked
                    else f"dev_children={_dev_children}/{len(node.children)}"
                    if _dev_children else
                    ("evicted" if node.evicted else "root_or_other"))
            _skip(h, action_id,
                  f"remove_hbm_not_device_leaf:{_why}")
            continue
        if "DRAM" in remove_tiers and not cache._is_host_leaf(node):
            _skip(h, action_id, "remove_dram_not_host_leaf")
            continue

        # ---- Apply adds first ----
        skip_this = False

        if "DRAM" in add_tiers:
            # write_through HBM → DRAM via cache_controller.
            if cache.cache_controller is None:
                _skip(h, action_id, "write_through_declined:no_hicache")
                skip_this = True
            else:
                try:
                    n_written = cache.write_backup(node)
                except Exception as exc:  # noqa: BLE001
                    import traceback as _tb
                    msg = str(exc) or "<empty>"
                    loc = "?"
                    st = _tb.extract_tb(exc.__traceback__)
                    if st:
                        last = st[-1]
                        fname = last.filename.rsplit("/", 1)[-1]
                        loc = f"{fname}:{last.lineno}:{last.name}"
                    short = "_".join(msg.split())[:60]
                    _skip(h, action_id,
                          f"write_through_raised:"
                          f"{type(exc).__name__}:{loc}:{short}")
                    skip_this = True
                else:
                    if n_written == 0:
                        _skip(h, action_id,
                              "write_through_declined:zero_tokens")
                        skip_this = True
        if skip_this:
            continue

        if "DISK" in add_tiers:
            # P5 (safe subset, per user confirmation): reuse sglang's OWN
            # host→storage write-through path (write_backup_storage)
            # rather than inventing a new aginfer-side storage writer or a
            # storage-only radix node state (stock sglang has neither a
            # "node lives only on DISK" tree state nor a delete API on any
            # storage backend — see the module + function docstrings).
            # This is strictly ADDITIVE and best-effort:
            #   - it only starts an async background write (the actual
            #     disk I/O + ack happens on cache_controller's storage
            #     thread, drained by the regular check_hicache_events /
            #     writing_check tick — same machinery sglang's own
            #     write-through-to-storage already relies on);
            #   - it requires the node to ALREADY be DRAM-backed
            #     (`node.backuped`) from a PRIOR action — never from a
            #     "DRAM" add earlier in THIS SAME action, which is
            #     rejected up front (`disk_add_conflicts_with_dram_add`,
            #     see above): `write_backup`'s device→host byte copy is
            #     itself async, so reading that host buffer immediately
            #     via `write_backup_storage` here could race an
            #     unfinished D→H copy (review PR #4, discussion_r3921269467);
            #   - it introduces ZERO new data-loss risk: at this point in
            #     the batch the bytes still live independently in
            #     HBM and/or DRAM (adds are applied before removes, see
            #     the docstring above), so this is purely an extra durable
            #     copy, never the only copy;
            #   - "applied" here means "write started", NOT "confirmed on
            #     disk" — there is no synchronous ack, and since no
            #     backend exposes delete, aginfer offers no DISK-residence
            #     GUARANTEE, only a best-effort extra backup (hence DISK
            #     is still deliberately never added to `current` residence
            #     above — a future action re-requesting `add DISK` on the
            #     same node will just re-fire this, which is harmless).
            if not cache.enable_storage or cache.cache_controller is None:
                _skip(h, action_id, "disk_add_declined:no_storage_backend")
                skip_this = True
            elif not node.backuped:
                # Reachable whenever the node has no pre-existing DRAM
                # residence (an in-batch "DRAM" add can no longer race
                # this — see disk_add_conflicts_with_dram_add above).
                # write_backup_storage would silently no-op on a
                # HBM-only node, so catch it here with a clear reason
                # instead of a fake "applied" success.
                _skip(h, action_id, "disk_add_declined:not_host_backed")
                skip_this = True
            else:
                try:
                    cache.write_backup_storage(node)
                except Exception as exc:  # noqa: BLE001
                    import traceback as _tb
                    msg = str(exc) or "<empty>"
                    loc = "?"
                    st = _tb.extract_tb(exc.__traceback__)
                    if st:
                        last = st[-1]
                        fname = last.filename.rsplit("/", 1)[-1]
                        loc = f"{fname}:{last.lineno}:{last.name}"
                    short = "_".join(msg.split())[:60]
                    _skip(h, action_id,
                          f"disk_backup_raised:"
                          f"{type(exc).__name__}:{loc}:{short}")
                    skip_this = True
        if skip_this:
            continue

        if "HBM" in add_tiers:
            # load_back DRAM → HBM via the same path the cache-hit
            # fast-path uses.
            try:
                ok = cache.load_back(node)
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                msg = str(exc) or "<empty>"
                loc = "?"
                st = _tb.extract_tb(exc.__traceback__)
                if st:
                    last = st[-1]
                    fname = last.filename.rsplit("/", 1)[-1]
                    loc = f"{fname}:{last.lineno}:{last.name}"
                short = "_".join(msg.split())[:60]
                _skip(h, action_id,
                      f"promote_raised:"
                      f"{type(exc).__name__}:{loc}:{short}")
                skip_this = True
            else:
                if not ok:
                    detail = (
                        getattr(cache, "_last_load_back_decline", None)
                        or "unknown"
                    )
                    category = ":".join(detail.split(":", 2)[:2])
                    _skip(h, action_id,
                          f"promote_load_back_declined:{category}")
                    skip_this = True
        if skip_this:
            continue

        # ---- Drain pending write_through before removes ----
        # write_backup (add=DRAM path) is ASYNC: it enqueues the
        # D→H copy on cache_controller's background thread and
        # records the pending lock in ongoing_write_through.  If
        # we now evict the device buffer (remove=HBM), the copy
        # would be reading freed memory + sglang's
        # invariant_checker would trip on the categories-no-
        # longer-disjoint state.
        #
        # writing_check(write_back=True) synchronizes ALL pending
        # write_through events (sees `finish_event.synchronize()`
        # per ack queue entry) AND properly releases locks via
        # dec_lock_ref(node, params).  Called only when this
        # action's adds actually produced async work (i.e.
        # DRAM was in add_tiers and write_backup returned > 0).
        if add_tiers and remove_tiers and cache.cache_controller is not None:
            cache.writing_check(write_back=True)

        # ---- Apply removes ----
        tracker = {ct: 0 for ct in cache.tree_components}
        base_comp = cache.components[BASE_COMPONENT_TYPE]
        if is_full_drop:
            # Full evict + remove leaf from tree.
            for comp in components:
                cache._evict_component_and_detach_lru(
                    node, comp, target=EvictLayer.ALL, tracker=tracker)
            cache.evictable_device_leaves.discard(node)
            cache.evictable_host_leaves.discard(node)
            cache._remove_leaf_from_parent(node)
            cache._iteratively_delete_tombstone_leaf(node, tracker)
        else:
            if "HBM" in remove_tiers:
                # Use the existing `_evict_to_host` helper rather
                # than rolling our own evict + cascade.  It does
                # FOUR things in the right order:
                #   1. evict_component_and_detach_lru DEVICE
                #      (frees the buffer + detaches from device LRU)
                #   2. _cascade_evict (nulls cd.value via tombstone
                #      since SWA's free_swa needed it earlier,
                #      then re-leaf-set the node)
                #   3. _for_each_component_lru insert_mru HOST
                #      (the node now has only host data → belongs
                #      in host LRU so future host-pressure can
                #      evict it)
                #   4. _update_evictable_leaf_sets(node.parent)
                #      (parent may now be a leaf again after
                #      child's tier transition)
                # Steps 3 + 4 were missing from the previous T20
                # impl, causing pool-accounting drift across
                # multiple migrates that triggered sglang's
                # invariant_checker (e2e_smoke 1st run).
                cache._evict_to_host(node, tracker=tracker)
                # Mark device-cleared for the in-batch leaf re-derivation
                # so this node's PARENT (processed later, deepest-first) is
                # recognised as a device-leaf despite the tombstoned (not
                # yet nulled) cd.value — lets the exclusive chain peel.
                batch_removed_hbm.add(node.id)
            if "DRAM" in remove_tiers:
                # Host eviction: evict_component sets cd.host_value
                # = None inline (no defer; SWA doesn't pin host
                # state).  Cascade still needed so aux components'
                # host state is consistent.
                cache._evict_component_and_detach_lru(
                    node, base_comp, target=EvictLayer.HOST,
                    tracker=tracker)
                cache._cascade_evict(
                    node, base_comp, tracker, target=EvictLayer.HOST)
            # (DISK can't appear in remove_tiers here — rejected up front.)
            cache._update_evictable_leaf_sets(node)

        applied += 1
        applied_hashes.append(h)
        acted_node_ids.add(node.id)
        # P1 metrics: tag by tier transition so get_aginfer_metrics can
        # report a HBM->DRAM / DRAM->HBM / *->DROP breakdown, not just a
        # single "applied" total.
        if is_full_drop:
            transition = "drop"
        elif "HBM" in add_tiers:
            transition = "dram_to_hbm"
        elif "HBM" in remove_tiers:
            transition = "hbm_to_dram"
        elif "DRAM" in remove_tiers:
            transition = "dram_drop_partial"
        elif "DISK" in add_tiers:
            transition = "disk_backup"
        else:
            transition = "other"
        migrate_counters = cache._aginfer_migrate_counters
        migrate_counters[transition] = migrate_counters.get(transition, 0) + 1

    return {"applied": applied, "applied_hashes": applied_hashes,
            "skipped": skipped,
            "hash_collisions": hash_collisions}


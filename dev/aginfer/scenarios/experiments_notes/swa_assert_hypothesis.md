# A3 promote AssertionError — code-tracing hypothesis

While the full traceback-capturing A3 cycle is running, here is
the educated guess from reading the load_back call graph.

## The likely culprit

`unified_radix_cache.py::load_back` calls
`self.inc_lock_ref(best_match_node)`.  That ends up in
`full_component.py::acquire_component_lock` (line ~197):

```python
cur = node
while cur is not root:
    cd = cur.component_data[ct]
    assert (
        cd.value is not None
    ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
    ...
    cur = cur.parent
```

**This walks node → root and asserts that EVERY ancestor has
`cd.value` (= is currently on the device).**

## Why this fires for daemon promote

Under HiCache `write_through_selective`, when sglang demotes a
node from HBM to DRAM (cd.value cleared, cd.host_value kept), it
typically demotes **the whole sub-tree** as one decision (eviction
LRU operates on leaves, but once a leaf is demoted its parent may
become evictable and also get demoted in a subsequent pass).

Daemon's V_u policy picks **individual leaf-ish nodes** in DRAM
that look "high reuse" and asks to promote them.  By the time
the daemon's POST arrives, the entire chain `root → ... → node`
is often all-DRAM.  load_back's first inc_lock_ref step then
hits an evicted ancestor → assert fires.

This is the canonical "promote requires promoting ancestors first"
problem — load_back assumed it would only be called from the
cache-hit code path, where `best_match_node` is by definition
the deepest device-on prefix (so ancestors are already on device).

## Fix candidates (lowest effort first)

* **(F1) Convert the assert into a soft skip**: change
  `assert cd.value is not None` to `if cd.value is None: raise
  EvictedAncestorError(...)`.  load_back catches that, returns
  False, and `apply_aginfer_migrations` reports
  `promote_requires_ancestor_promote`.  Daemon's V_u will see the
  ancestor in DRAM on the next cycle and promote it.  This is
  iterative — multiple events to walk the chain.
* **(F2) Recursive promote in apply_aginfer_migrations**: before
  calling load_back on the target node, walk up the chain, call
  load_back on each evicted ancestor bottom-up.  One POST →
  multiple loads.  Higher throughput but more complex semantics.
* **(F3) Allow load_back to internally handle evicted ancestors**:
  modify load_back to do (F2) inside itself.  Cleanest end-state
  but changes load_back's contract used by the cache-hit path.

(F1) is the v1 fix: minimal change, daemon picks up the slack
via its event loop.  Slower convergence but doesn't change
load_back's existing contract.

## Validation

The current A3 cycle (started ~21:25 UTC 2026-05-30) uses the
patched exception handler that captures
`{filename}:{lineno}:{funcname}`.  Once it finishes:

* If the reason text contains `full_component.py:NNN:acquire_component_lock`
  with NNN near 198, hypothesis confirmed → implement F1.
* If it contains a different location → patch that one instead.
* If multiple distinct locations dominate → handle each.

## Why we missed this in A3 v3

A3 v3 had 2 130 successful promotes alongside 12 003 failures.
Successes are presumably the lucky cases where the chain was
not fully demoted (e.g. shared-prefix system-prompt nodes that
many trials still actively touch → never demoted).  Failures
are nodes where the entire chain happened to be DRAM-only.

A3 v4 had 1 300 successful + only 28 failures — likely because
A3 v3 left some warm device state in the radix tree (Mooncake
backed pool persists?) and v4 inherited a hotter HBM, making
demote → all-DRAM-chain less common.

If F1 lands, A3 vN should show:
* `promote_requires_ancestor_promote` (new skip reason)
* `applied` count rises because daemon iteratively promotes
  chains over consecutive events.

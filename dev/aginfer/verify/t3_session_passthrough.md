# T3 — session_id passthrough → `UnifiedTreeNode.session_ids`

## WHAT WE PROMISED

**Capability**
* When a request carries `extra_body.program_id`, every tree node touched
  by that request's prefix path picks up `program_id` in its
  `session_ids: set[str]` field.
* `/aginfer/state` exposes the resulting set.
* No request with `program_id` set ever lands on a node that does NOT
  have that id in `session_ids`.
* Concurrent program_ids on the same shared prefix all appear (the
  set grows; never overwrites).

**Cost ceiling**
* < 30 lines added to sglang.
* Per-request overhead: one `set.add` call per node along the insert path.
* No measurable latency impact (< 0.1 ms per request).

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Request without `program_id` | Send 16 chat-completions with no `extra_body.program_id`; mix with 16 well-tagged ones | Untagged nodes carry `session_ids = ∅`; admission can't pause them; OTHER 16 tagged programs unaffected | `/aginfer/state` dump; assert each node's `session_ids` is either set or empty, never undefined / crashed |
| Bogus `program_id` (non-string, very long) | Send request with `program_id={"oh":"no"}` and `program_id="x"*10000` | Sanitizer coerces to str + truncates to 64 chars; no exception | inspect resulting tree node's `session_ids` |
| Conflicting `program_id` on shared prefix | Issue 32 program-distinct requests with identical 1 k-token prefix | Shared prefix node's `session_ids` contains all 32 ids; tail nodes have only their own id | `/aginfer/state` schema check |
| `program_id` cache unbounded growth | Issue 10 k unique program_ids each touching a distinct prefix | Cache stays bounded by `program_tracker` GC (T6); if T6 not yet shipped, hard cap kicks in at 50 k | memory check via psutil; assert < 100 MB |

## HOW WE VERIFY

Mechanism. `verify/t3_session_passthrough.py`:

```
1. Launch sglang. Issue request A with extra_body.program_id="prog-A"
   and a unique-ish suffix. Same for "prog-B" with overlapping prefix
   (e.g. shared 1 K-token system prompt + unique tail).
2. GET /aginfer/state. Walk results:
   - Shared-prefix nodes should have both {"prog-A", "prog-B"} in session_ids.
   - Tail nodes for A should have only {"prog-A"}.
   - Tail nodes for B should have only {"prog-B"}.
3. Issue 32 program-distinct requests with a shared prefix. Verify
   shared-prefix node's session_ids has all 32.
```

## RESULTS
* date: _pending_
* sglang sha:
* lines added:
* shared prefix has both ids: _pending_
* concurrent 32-program test: _pending_
* per-request latency delta vs baseline: _pending_
* raw log: `verify/results/t3_<datetime>.log`

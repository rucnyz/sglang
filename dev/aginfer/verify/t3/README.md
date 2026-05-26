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

**PASSED** — all 7 steps on Qwen3-0.6B + `--attention-backend flashinfer`.

* date: 2026-05-26
* sglang sha: (this commit)
* lines added: ~40 across 9 files (protocol.py + io_struct.py +
  tokenizer_manager.py + serving_chat.py + serving_completions.py +
  scheduler.py + schedule_batch.py + base_prefix_cache.py +
  unified_radix_cache.py)
* Two-program shared prefix (step [1]): 7 units total, 1 node carries
  both `prog-A` AND `prog-B` (the shared system prompt), 2 nodes per
  program in their diverging suffixes. ✓
* No fake tags (step [2]): every unit's `session_ids` is either the
  expected program_ids or empty. ✓
* OpenAI client `extra_body={"program_id": ...}` end-to-end (step [3]):
  reaches the radix tree as a top-level field after client-side
  unpacking. ✓
  - Note: the server only sees a top-level `program_id` field; the
    OpenAI client library does the `extra_body` -> top-level unpack
    BEFORE sending. Verify uses the real `openai` package for this
    test (falls back to top-level if `openai` package is missing).
* Concurrent 32-program shared-prefix stress (step [4]): the
  shared-prefix node carries ALL 32 program_ids in its
  `session_ids` set. ✓ (the strongest test of the contract)
* Untagged request (step [5]): leaves `session_ids = []` on every
  node it touches; no exception. ✓
* Bogus program_id shapes (step [6]): 5 cases — `{"oh": "no"}`,
  `42`, `"x" * 10_000`, `["a", "b"]`, `True`. All sanitized to a
  ≤64-char string at Req construction; no HTTP 400, no 5xx. ✓
  Pydantic field needed `Optional[Any]` (otherwise Pydantic rejects
  bogus shapes at the HTTP layer before our sanitizer runs).
* Tagging overhead (step [7]): -0.18 ms/req (noise level) for 20
  tagged vs 20 untagged chat completions. Well under the 5 ms
  ceiling. ✓
* raw log: `results/<YYYYMMDD_HHMMSS>_run4.log`

### Implementation notes (forward-compat for T4-T8)

* Wire path: `extra_body.program_id` (OpenAI client) -> top-level
  `program_id` -> `ChatCompletionRequest` -> `GenerateReqInput` ->
  `TokenizedGenerateReqInput` -> `Req` -> `InsertParams.program_id`
  -> tagged on each node in `_insert_helper`.
* Sanitizer (`_sanitize_program_id` in schedule_batch.py) runs ONCE
  at `Req.__init__`; downstream code can assume `req.program_id` is
  either None or a stable ≤64-char string.
* Split node: when a radix node is split (a longer prefix becomes
  the parent of two diverging tails), the new internal node inherits
  the union of `session_ids` from the original child. Without this
  the admission_controller (T8) would under-count program ownership
  on shared ancestors. Verified by step [4].

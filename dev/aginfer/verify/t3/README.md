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
* ~120 lines added to sglang across the wire path (protocol, io_struct,
  tokenizer_manager, serving_chat/completions, scheduler, schedule_batch,
  base_prefix_cache, unified_radix_cache, encode_receiver for the EPD
  disaggregation path).
* Per-request overhead: one `set.add` per node along the insert path.
  Set sizes are bounded by the number of distinct programs that touch a
  given prefix node — typically O(concurrent_programs).
* End-to-end overhead is dominated by inference; tag handling is far
  below the noise floor of OpenAI-API completions (~30 ms each for
  short prompts). The verify measures < 5 ms/req added overhead (the
  noise envelope), not the unrealistically tight 0.1 ms originally
  promised.
* `session_ids` wire format: emitted as a sorted JSON `list[str]` for
  byte-stable JSON; daemon must treat the parsed value as a set
  (membership, union, weighting by `1 / len(session_ids)`).

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Request without `program_id` | Send 16 chat-completions with no `extra_body.program_id`; mix with 16 well-tagged ones | Untagged nodes carry `session_ids = ∅`; admission can't pause them; OTHER 16 tagged programs unaffected | `/aginfer/state` dump; assert each node's `session_ids` is either set or empty, never undefined / crashed |
| Bogus `program_id` (non-string, very long, whitespace-only) | Send `program_id={"oh":"no"}`, `program_id=42`, `program_id=["a","b"]`, `program_id="x"*10000`, `program_id="   "` | Sanitizer at Req construction: `.strip()` then coerce to str then truncate to 64 chars. Whitespace-only or empty becomes None (untagged). List collapses to the first **non-empty** sanitized element (a leading None / "" doesn't silently kill a valid later element). No HTTP 400; no exception. | inspect resulting tree node's `session_ids`; ≤64 chars per entry |
| Conflicting `program_id` on shared prefix | Issue 32 program-distinct requests with identical 1 k-token prefix | Shared prefix node's `session_ids` contains all 32 ids; tail nodes have only their own id | `/aginfer/state` schema check |
| Chunked prefill | Send a long-generation chat (>1 chunk worth of prompt) with a `program_id`; sglang's chunked prefill triggers multiple `cache_unfinished_req` calls on the same `Req` | program_id tags every chunk's nodes (insert path is hit each chunk; sanitizer ran once at Req construction so the value is stable) | /aginfer/state shows the program_id on the chunked Req's full path, no missing nodes |
| Untagged-then-tagged retro-tagging | Send untagged request first (creates an untagged node); then send a tagged request that shares the prefix | The tagged request's pid is added to the previously-untagged ancestor's session_ids (set semantics, additive) | check the shared-prefix node now has the pid that was added later |
| Wire format vs in-memory: SET semantics, LIST wire | `node.session_ids` is a Python `set`; the dump emits `sorted(list)` for deterministic byte-stable JSON. Daemon must treat the parsed list AS a set (membership / union / weighting by `1 / len`). | assert list is sorted; assert no duplicates within a single response | T1 schema validator already enforces |

### Tagging-path coverage and v1 limitations

The general contract is "every radix-tree node touched by a tagged
request gets the tag", but the v1 wire-up does NOT cover three
paths.  These are documented here so T9 / T10 can address each
when they're exercised:

1. **HiCache `prefetch_from_storage` (`unified_radix_cache._insert_helper_host`)**.
   When a tagged request triggers a host-side prefetch from
   Mooncake / disk, the host nodes are created with empty
   `session_ids`.  The tag is added later, when the matching
   device-side insert runs (`_insert_helper`).  Between those two
   events, the daemon's `/aginfer/state` snapshot may show the
   prefetched host nodes as untagged.  T9 Run K will exercise this;
   we'll either thread `program_id` through `prefetch_from_storage`
   then or document the race window's acceptance criterion.
2. **`StreamingSession.try_cache_*`** (streaming-session API):
   requests bound to a streaming session route through the
   `StreamingSession` short-circuit in `cache_finished_req` /
   `cache_unfinished_req` and never reach `_insert_helper`.
   v1 simply doesn't tag streaming-session reqs; if T6 / T7 need
   them tagged, plumb `program_id` into the StreamingSession path
   separately.
3. **EPD-disaggregation** (`encode_receiver.create_req`): now FIXED
   (commit follow-up to T3 audit-round-2).  The `program_id` is
   forwarded through the EPD path so tagged requests retain their
   tag when sglang runs in encode-prefill-decode disaggregation
   mode.  Without the fix, `Req.program_id` would default to None
   in EPD mode and the tag would die silently.

### Design decision: `session_ids` is unbounded in v1

The original WORST CASE table promised a 50 k hard cap on
`session_ids` per node "until T6 ships". We removed that promise after
a design audit: the 50 k number was a placeholder, not derived from
any real workload or paper §7 constraint, and a per-node cap would
silently drop the (50 k+1)-th tag — a hard-to-debug functional
regression for the daemon's admission_controller.

The right owner of program-lifecycle bookkeeping is T6's
`program_tracker`, not sglang's radix cache. v1 sglang carries
`session_ids` for as long as the node lives; when the radix cache
evicts the node (LRU or daemon-driven DROP), the set dies with it
("DROP path drops the set with the node" — verified naturally by T2
verify's tier_usage delta).

Concrete risk during v1 development:

* A buggy client emitting a fresh `program_id` per request can grow
  the busiest shared-prefix node's set unboundedly, costing ~64 B per
  unique id and O(n log n) dump cost. Real workloads have ≤ ~1 000
  concurrent programs and ≤ ~10 000 in a daily window — no measurable
  impact. Adversarial clients are out of scope for v1; T9 / T10
  may revisit if profiling reveals a concrete issue.
* paper §9 deployment model assumes the daemon (T6) owns program
  lifecycle, so the cap responsibility is layered out of sglang.

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
* raw log: `results/<YYYYMMDD_HHMMSS>_run5_postaudit.log` (10 steps including
  audit-driven additions: hard-required OpenAI client + raw-POST negative
  test, retro-tag, chunked prefill, list-broadcast sanitizer)

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

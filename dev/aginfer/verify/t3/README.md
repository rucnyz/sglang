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
request gets the tag", but several edge paths need explicit
documentation:

1. **HiCache `prefetch_from_storage` (`unified_radix_cache._insert_helper_host`)** — host-side nodes from a tagged prefetch land with empty `session_ids` until the device-side insert tags them. v1 limitation; T9 Run K will exercise the race window.
2. **`StreamingSession.try_cache_*`** — requests bound to a streaming session bypass `_insert_helper` entirely. v1 simply doesn't tag streaming-session reqs.
3. **EPD-disaggregation** (`encode_receiver.create_req`) — FIXED in audit-round-2; tag forwards through the encode-prefill-decode path.
4. **Session multi-turn** (`Session.create_req`) — FIXED in audit-round-3; tag forwards through `session.create_req` for requests that reuse a session_id. Demonstrated by `verify/t3/regression_probe.py [A]`: pre-fix tagged 0 nodes, post-fix tags 4.
5. **OpenAI chat handler does NOT forward `session_params`** to the underlying GenerateReqInput. So the multi-turn session path is only reachable via sglang's native `/generate` endpoint. The OpenAI-API surface goes through the non-session branch in scheduler.py (which already plumbs `program_id` correctly). Documented because future maintainers may want to wire `session_params` into the OpenAI handler.
6. **Multi-DP / `per_rank` aggregation** — when DP > 1, `/aginfer/state` returns `{"per_rank": [...]}` with each rank's units separately. Each rank's tree is independent; same `program_id` may appear on different rank-local hashes. v1 verify only tests single-DP; daemon (T6/T7) is responsible for merging the per-rank view. Per-rank `node-<id>` names use a process-local counter — id collisions ACROSS ranks are expected and the daemon namespace must include rank.
7. **HBM ↔ DRAM tier transition** — `_evict_to_host` keeps the same Python node object; `session_ids` survives the tier flip automatically. No verify pinned for this (requires HiCache); T9 / T10 will exercise.
8. **Speculative-decoding draft model** — draft-model KV is managed by its own pool, NOT by `UnifiedRadixCache`. Draft-model nodes are not tagged. v1 limitation; T9 will exercise if speculative is enabled and document the gap.

### `program_id` distribution semantics for list inputs

There are two list-shaped paths and they MEAN different things — the
daemon should know which it's invoking:

1. **Single request + list `program_id`** (sanitizer-collapse path):
   `program_id=["a", "b"]` on a single-completion (`n=1`) request
   collapses to `"a"` (first non-empty after sanitization).  Verify
   step [10] guards this.
2. **Batched / parallel-sample + list `program_id`** (per-item slicing
   path): `program_id=["a", "b"]` on a request with `n=2` (or a
   batched `text=[...]` of length 2) dispatches `"a"` to the first
   item and `"b"` to the second via `GenerateReqInput.__getitem__`.
   Each completion gets its own tag.  Out-of-range index falls back
   to broadcasting the whole list, which the sanitizer then collapses
   to first.

If the daemon emits a list, it's signalling per-item.  If it emits a
scalar, it's broadcasting.  The wire format does NOT differentiate
"intended-list" from "intended-broadcast"; the daemon must pick.

### Reserved sentinel namespace

The daemon (T6/T7) is free to define internal program_id sentinels
(e.g. `"__aginfer_paused__"` for the program-tracker's PAUSED state).
v1 sglang's sanitizer accepts any string ≤ 64 chars without
reservation; callers are responsible for picking namespaces that
don't collide with their own runtime sentinels. **Convention**:
daemon-internal sentinels should use the `__aginfer_*` prefix; user
programs should not use that prefix.

### Design decision: `session_ids` is unbounded in v1

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

## REPRODUCING

T3 has TWO scripts and an EXTRA launch flag.  Step [9] in
`verify.py` exercises the chunked-prefill path; without
`--chunked-prefill-size 32` the request stays in one chunk and the
chunked-path test degrades to "long generation only" (still passes,
but no longer pins the chunked branch).

`CUDA_VISIBLE_DEVICES=4` is a default — pick any free GPU per
`nvidia-smi` (typical convention is GPU 5 or 6 free).  Capture launch
PID so the tear-down is precise.

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

# openai package is required for the OpenAI-client extra_body test
# (step [3] of verify.py).  Force-use the conda-env python so a stray
# system python on $PATH doesn't install to the wrong site-packages.
"$CONDA_PREFIX/bin/python" -c "import openai" \
  || "$CONDA_PREFIX/bin/pip" install openai

cd /scratch/yuzhou/projects/sglang/dev/aginfer
lsof -i:30001 && { echo "port 30001 already in use"; exit 1; }
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=4 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --chunked-prefill-size 32 \
  > logs/sglang_t3.log 2>&1 &
SGLANG_PID=$!

# Wait for the listener.
until grep -q "Uvicorn running on http://127.0.0.1:30001" logs/sglang_t3.log; do sleep 3; done

# Main 13-step verify (production-shape end-to-end coverage).  The
# verify itself preflights /get_server_info and refuses to run if
# chunked_prefill_size > 64 (catches a launch that forgot the flag,
# the round-4 audit's regression-class for fake-chunked).
AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python verify/t3/verify.py
# expected last line: "=== T3 PASSED (post-audit round 3) ==="

# Bisect regression probe (round-3 audit BLOCKERs):
#   [fix-state] introspect production code, fail loudly if the bisect
#               revert was forgotten
#   [A] Session.create_req forwards program_id
#   [B] sanitizer recursion cap
# This is the "first prove the bug exists, then prove the fix works"
# demo.  Pre-fix logs (in verify/t3/results/*PREFIX*.log) show both
# FAIL; post-fix logs show both PASS.
AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python verify/t3/regression_probe.py
# expected last lines: "PASS  Session.create_req forwards program_id"
#                      "PASS  sanitizer recursion cap"

kill "$SGLANG_PID"
```

To re-run the pre-fix demo for either BLOCKER:
* BLOCKER A: in `python/sglang/srt/session/session_controller.py`,
  comment out the ``program_id=req.program_id`` line in the Req
  constructor (~line 243), restart sglang, re-run probe — probe [A]
  must FAIL.  Restore + restart + re-run, probe [A] must PASS.
* BLOCKER B: in `python/sglang/srt/managers/schedule_batch.py`, set
  ``_PROGRAM_ID_MAX_RECURSION = 999_999``, restart, re-run probe —
  probe [B] must FAIL.  Restore to 8 + restart + re-run, probe [B]
  must PASS.

## RESULTS

**PASSED** — all 14 verify steps + 3-section regression probe (post
audit round-5) on Qwen3-0.6B + `--attention-backend flashinfer
--chunked-prefill-size 32`.

* date: 2026-05-26
* sglang sha: (this commit)
* lines added: ~140 lines across 10 files (protocol.py + io_struct.py +
  tokenizer_manager.py + serving_chat.py + serving_completions.py +
  scheduler.py + schedule_batch.py + base_prefix_cache.py +
  unified_radix_cache.py + session_controller.py +
  disaggregation/encode_receiver.py — every Req construction site that
  takes a TokenizedGenerateReqInput now forwards program_id).
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
* Sanitizer overhead (step [7], audit-of-tests round-2 rewrite):
  ``_sanitize_program_id`` direct microbench — 5 runs × 10 000
  calls over a representative input mix (happy str / None / long /
  list).  Result: **0.22 ± 0.00 µs/call** (mean + 3σ ≈ 0.23 µs)
  vs the new 10 µs/call ceiling.  The previous version compared
  end-to-end tagged vs untagged ``chat()`` latency where inference
  jitter (hundreds of ms) drowned the < 0.1 ms/req cost claim — a
  30× regression in the sanitizer would have been invisible.  Per
  memory:feedback-latency-multi-run. ✓
* Session multi-turn (step [11]): a request with `session_params.id`
  set via the native `/generate` endpoint tags the shared-session
  nodes via `Session.create_req`. Round-3 audit caught the silent
  drop here; bisect probe (`regression_probe.py`) confirms 0 tagged
  nodes pre-fix and 4 post-fix.
* Recursion DoS (step [12]): a 20-level deeply-nested-list
  `program_id` sent as raw JSON to `/generate` does not crash the
  scheduler; the sanitizer's depth cap (8) drops the tag silently.
  Bisect probe confirms: pre-fix the buried tag bypasses, post-fix
  it does not.
* Pydantic regression (step [13]): `ChatCompletionRequest.model_config.extra`
  must NOT be `"allow"`; introspection check guards the contract.
* raw logs (relative to this directory):
  * `results/20260526_023126_run5_postaudit.log` — round-1+2 fixes,
    10 steps passing
  * `results/20260526_025258_run6_audit2.log` — round-2 audit fixes
    (depth-audit complete)
  * `results/20260526_031612_PREFIX_demo_v4.log` — **bisect PRE-fix**
    (round-3 BLOCKERs reverted, probe FAILs both)
  * `results/20260526_031737_POSTFIX_demo.log` — **bisect POST-fix**
    (fixes restored, probe PASSes both)
  * `results/20260526_031841_run9_audit3_clean.log` — round-3 audit
    fixes, 13 steps passing
  * `results/20260526_034309_run10_audit4.log` — round-4 audit fixes
    (step [11] rewritten to use /generate, preflight added)
  * `results/20260526_035348_run11_audit5.log` — round-5 audit fixes
    (recursion-must-reach-sanitizer + EPD self-check), 14 steps + 3
    probe sections all passing — **current state**

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

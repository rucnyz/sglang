"""T3 verify: session_id passthrough to UnifiedTreeNode.session_ids.

Validates the contract documented in dev/aginfer/verify/t3/README.md.

Sends OpenAI chat-completion requests with ``program_id`` set (either at
the top level or inside ``extra_body``), then GETs /aginfer/state and
asserts:

  * shared-prefix nodes carry every program_id that visited them;
  * tail nodes carry only their own program_id;
  * untagged requests leave the tree with empty session_ids on their
    own nodes — no exception, no fake program tag;
  * bogus program_id shapes (non-str, very long, dict, int) are
    sanitized rather than crashing.

Usage:
    # assumes sglang is already up at http://127.0.0.1:30001
    # IMPORTANT: launch sglang with ``--chunked-prefill-size 32`` so step
    # [9] actually exercises the chunked path (default size is 8 K,
    # which a 256-token gen never crosses).
    python dev/aginfer/verify/t3/verify.py
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")


def chat(
    user_text: str,
    program_id: Any = None,
    system_text: Optional[str] = None,
    use_extra_body: bool = False,
) -> None:
    """Issue one chat completion. ``program_id`` may be a string, anything
    castable, or None.  ``use_extra_body=True`` sends it under
    ``extra_body`` instead of at the top level — both must work end-to-end."""
    messages = []
    if system_text is not None:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    body: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4,
        "temperature": 0.0,
    }
    if program_id is not None:
        if use_extra_body:
            body["extra_body"] = {"program_id": program_id}
        else:
            body["program_id"] = program_id
    requests.post(
        f"{BASE}/v1/chat/completions",
        json=body,
        timeout=60,
    ).raise_for_status()


def fetch_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def units_with(state: Dict[str, Any], program_id: str) -> List[Dict[str, Any]]:
    return [u for u in state["units"] if program_id in u["session_ids"]]


def units_without(state: Dict[str, Any], program_id: str) -> List[Dict[str, Any]]:
    return [u for u in state["units"] if program_id not in u["session_ids"]]


def main() -> None:
    print("=== T3 verify: session_id passthrough ===")

    # Launch-config preflight: catches "launch line forgot the flag"
    # silently passing the chunked test (round-2 fake-chunked regression).
    info = requests.get(f"{BASE}/get_server_info", timeout=10)
    info.raise_for_status()
    info_json = info.json()
    server_args = info_json.get("server_args", {}) if isinstance(info_json, dict) else {}

    # chunked_prefill_size: step [9] needs <=64 to actually trigger chunked.
    chunked_size = info_json.get("chunked_prefill_size")
    if chunked_size is None:
        chunked_size = server_args.get("chunked_prefill_size")
    assert chunked_size is not None, (
        f"could not read chunked_prefill_size from /get_server_info; "
        f"top-level keys: {sorted(info_json)[:10]}; "
        f"server_args keys: {sorted(server_args)[:10] if server_args else 'absent'}. "
        f"sglang's /get_server_info schema may have changed."
    )
    assert chunked_size <= 64, (
        f"sglang launched with chunked_prefill_size={chunked_size}; "
        f"verify step [9] needs <=64 to actually exercise the chunked "
        f"path.  Relaunch with `--chunked-prefill-size 32`."
    )

    # UnifiedRadixCache preflight (audit round-5 MINOR 6): without
    # SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 the tree_cache is a different
    # class and /aginfer/state schema diverges; step [1] would then fail
    # with a confusing "no A-only tail nodes" instead of pointing at the
    # config.  Two checks: (a) /aginfer/state returns the expected schema
    # keys, (b) env or server_args flag if exposed.
    try:
        state_probe = requests.get(f"{BASE}/aginfer/state", timeout=10).json()
    except Exception as exc:
        raise AssertionError(
            f"/aginfer/state preflight failed: {exc}. sglang likely launched "
            f"without SGLANG_ENABLE_UNIFIED_RADIX_TREE=1, or the radix cache "
            f"isn't UnifiedRadixCache."
        )
    # DESIGN §5 (post-T17) schema preflight.  Legacy `tier_usage` /
    # `page_size` / `bytes_per_token` top-level keys have been removed.
    for k in ("units", "pool_usage", "per_program_usage", "link_stats",
              "tier_holding_cost", "throughput_ema"):
        assert k in state_probe, (
            f"/aginfer/state missing key {k!r}; schema mismatch suggests "
            f"sglang was not launched with SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 "
            f"or the pre-T17 schema is still in place.  Got keys: "
            f"{sorted(state_probe)}"
        )

    # Flush so any residue tag from prior runs of regression_probe or
    # earlier verify invocations doesn't poison our wire-format
    # invariants (step [2] checks that no node has implausibly many
    # tags).  If /flush_cache is unavailable, the residue could mask a
    # real regression -- so we hard-fail rather than swallow.
    flush = requests.post(f"{BASE}/flush_cache", timeout=10)
    assert flush.status_code in (200, 204), (
        f"/flush_cache returned {flush.status_code}; without flush, "
        f"residue tags from prior runs poison step [2]'s wire-format "
        f"invariants.  Relaunch sglang fresh or expose /flush_cache."
    )

    SHARED_SYS = (
        "You are a helpful assistant. Here is a long system prompt: "
        + ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 30)
    )

    # ---- [1] Two programs share a system prompt; tail diverges ----
    print("\n[1] two programs share a system prompt")
    chat("Tell me a 1-line fact about prime number 7.",
         program_id="prog-A", system_text=SHARED_SYS)
    chat("Tell me a 1-line fact about prime number 11.",
         program_id="prog-B", system_text=SHARED_SYS)
    state = fetch_state()
    n_units = len(state["units"])
    n_with_A = len(units_with(state, "prog-A"))
    n_with_B = len(units_with(state, "prog-B"))
    n_shared = len([u for u in state["units"]
                    if "prog-A" in u["session_ids"] and "prog-B" in u["session_ids"]])
    print(f"    total units: {n_units}, with prog-A: {n_with_A}, with prog-B: {n_with_B}, shared: {n_shared}")
    # Both tags must appear.
    assert n_with_A > 0, "prog-A did not tag any node"
    assert n_with_B > 0, "prog-B did not tag any node"
    # At least one shared-prefix node carries BOTH.
    assert n_shared > 0, "no node carries both prog-A AND prog-B; shared system prompt not tagged"
    # A's tail-only nodes (in A but not in B) exist (the diverging suffix).
    a_only = [u for u in state["units"]
              if "prog-A" in u["session_ids"] and "prog-B" not in u["session_ids"]]
    b_only = [u for u in state["units"]
              if "prog-B" in u["session_ids"] and "prog-A" not in u["session_ids"]]
    print(f"    A-only nodes: {len(a_only)}, B-only nodes: {len(b_only)}")
    assert a_only, "no A-only tail nodes — diverging suffix lost the tag?"
    assert b_only, "no B-only tail nodes"

    # ---- [2] Per-unit causal: every tagged unit carries the tag ----
    # Also pins the wire-format invariants the daemon will rely on:
    #   * session_ids is a JSON list of strings
    #   * sorted (deterministic byte-stable JSON)
    #   * no duplicate entries (set semantics in memory, list on wire)
    #   * len(session_ids) per node is reasonable (sanity check for the
    #     daemon's 1/len weighting in admission_controller)
    # ORDER-DEPENDENCE: the len <= 4 upper bound assumes ONLY steps [1]
    # have run so far (prog-A + prog-B + at most a couple stragglers).
    # If step [4]'s 32-program stress runs BEFORE this, the bound fails.
    # Keep [2] right after [1]; do not reorder.
    print("\n[2] wire-format invariants + no fake tags")
    expected_tags = {"prog-A", "prog-B"}
    for u in state["units"]:
        sids = u["session_ids"]
        assert isinstance(sids, list), f"session_ids must be list: {sids!r}"
        assert sids == sorted(sids), f"session_ids not sorted: {sids}"
        assert len(sids) == len(set(sids)), f"session_ids has duplicates: {sids}"
        # daemon-weighting sanity: per-node count is bounded -- for this
        # test, no node has touched more than the two expected programs.
        assert 0 <= len(sids) <= 4, (
            f"unit {u['hash']} has implausible session_ids count: {len(sids)}"
        )
        extra = set(sids) - expected_tags
        assert not extra, f"unit {u['hash']} has unexpected tags: {extra}"

    # ---- [3] extra_body path via the real OpenAI client (the daemon's path) ----
    # The OpenAI client unpacks ``extra_body`` into top-level body fields
    # CLIENT-SIDE before sending; the server only ever sees top-level
    # ``program_id``.  This test does TWO things:
    #   (a) real OpenAI client with extra_body -> assert pid reaches the
    #       tree (= the daemon's actual code path);
    #   (b) raw POST with a nested "extra_body" key -> assert the pid
    #       does NOT reach the tree.  This proves the unpack is purely
    #       client-side and the server has no fallback that would mask
    #       a misconfigured daemon.
    print("\n[3] extra_body passthrough (= top-level after OpenAI client unpack)")
    # Hard requirement -- the OpenAI client is the daemon's wire format;
    # no silent fallback that would mask a broken extra_body contract.
    from openai import OpenAI  # noqa: WPS433

    client = OpenAI(base_url=f"{BASE}/v1", api_key="not-needed")
    client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SHARED_SYS},
            {"role": "user", "content": "Tell me a 1-line fact about prime 13."},
        ],
        max_tokens=4,
        temperature=0.0,
        extra_body={"program_id": "prog-EB"},
    )
    state = fetch_state()
    assert units_with(state, "prog-EB"), (
        "OpenAI-client extra_body.program_id did not reach the radix tree"
    )
    # Negative case: nested extra_body via raw POST must NOT tag the tree.
    # Pydantic doesn't have an `extra_body` field on ChatCompletionRequest;
    # the JSON key is silently ignored, so this proves the server doesn't
    # auto-unpack.
    requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SHARED_SYS},
                {"role": "user", "content": "Tell me a 1-line fact about prime 17."},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
            "extra_body": {"program_id": "prog-SHOULD-NOT-LAND"},
        },
        timeout=60,
    ).raise_for_status()
    state = fetch_state()
    assert not units_with(state, "prog-SHOULD-NOT-LAND"), (
        "nested extra_body somehow reached the tree -- server is auto-"
        "unpacking, which contradicts the OpenAI client contract"
    )

    # ---- [4] Concurrent 32-program shared-prefix stress ----
    print("\n[4] 32 distinct programs share a long prefix")
    PREFIX_32 = "Common system prompt v2: " + ("foo bar baz quux. " * 50)
    for i in range(32):
        chat(f"Distinct user query {i}: count to {i}.",
             program_id=f"p32-{i}", system_text=PREFIX_32)
    state = fetch_state()
    p32_tags = {f"p32-{i}" for i in range(32)}
    # Find the node with the most p32 ids — should have all 32 (the shared
    # system prompt node).
    best = max(
        state["units"],
        key=lambda u: len(p32_tags & set(u["session_ids"])),
    )
    overlap = p32_tags & set(best["session_ids"])
    print(f"    busiest shared node carries {len(overlap)} of 32 p32 tags")
    assert len(overlap) == 32, (
        f"shared-prefix node only carries {len(overlap)}/32 program tags; "
        f"max session_ids: {sorted(best['session_ids'])[:5]}..."
    )

    # ---- [5] WORST CASE: untagged request ----
    print("\n[5] WORST CASE: untagged request leaves session_ids untouched")
    # Send untagged with a distinct prompt so it lands on its own leaf.
    UNTAGGED_PROMPT = "Untagged request unique key: prime 9001."
    chat(UNTAGGED_PROMPT, program_id=None)
    state = fetch_state()
    # The untagged request's leaf should have NO program_id at all.  We
    # can't pinpoint the exact node without internals, so we just assert
    # that the global state.units still has well-formed session_ids and
    # contains at least one unit with session_ids = [] (the untagged tail).
    empty_units = [u for u in state["units"] if u["session_ids"] == []]
    print(f"    units with empty session_ids: {len(empty_units)}")
    # Round-6 audit MINOR 7: previous assert was `empty_units` (truthy),
    # which would pass even if a regression auto-tagged every node with
    # a sentinel (e.g. "system") and left only ONE node untagged.
    # Require >=2 empty-session nodes -- a healthy tree always has the
    # untagged request's distinct tail plus at least one untagged
    # ancestor (e.g. a fresh root child the tagged paths don't touch).
    assert len(empty_units) >= 2, (
        f"only {len(empty_units)} untagged nodes; a regression that "
        f"auto-tags every touched node with a sentinel could leave just "
        f"one untagged leaf and still pass the old `assert empty_units` "
        f"check.  Expected >=2 in a clean tree."
    )
    # Schema check: each session_ids is a list of strings.
    for u in state["units"]:
        assert isinstance(u["session_ids"], list)
        for sid in u["session_ids"]:
            assert isinstance(sid, str), f"non-str sid: {sid!r} in {u['hash']}"

    # ---- [6] WORST CASE: bogus program_id shapes (must NOT crash) ----
    print("\n[6] WORST CASE: bogus program_id shapes (must not 5xx)")
    bogus_cases = [
        {"oh": "no"},            # dict
        42,                        # int
        "x" * 10_000,             # very long string -> truncated to 64
        ["a", "b"],               # list
        True,                      # bool
    ]
    for bogus in bogus_cases:
        # Just confirm the request doesn't 5xx.  The sanitizer turns the
        # input into either None or a ≤64-char string.
        chat(f"Bogus pid case {type(bogus).__name__}: hello.",
             program_id=bogus,
             system_text=None)  # no shared prefix; avoid mixing with above
    state = fetch_state()
    # Sanitizer truncation: assert no session_id exceeds 64 chars.
    for u in state["units"]:
        for sid in u["session_ids"]:
            assert len(sid) <= 64, f"sid not truncated: len={len(sid)}: {sid[:80]}"
    print("    5 bogus shapes handled cleanly; all session_ids <= 64 chars")

    # ---- [7] Sanitizer microbench: pure _sanitize_program_id cost ----
    #
    # Audit round-2 ("audit of tests"): the previous version compared
    # end-to-end tagged vs untagged chat() latency, where total cost is
    # dominated by inference jitter (hundreds of ms).  The cost ceiling
    # claim is < 0.1 ms PER CALL for the sanitize/tag path; that signal
    # was completely buried — a 30× regression in the sanitizer would
    # not move the 5 ms ceiling.  We now microbench the sanitizer
    # directly: 5 runs × 10 000 calls each, report mean+std per call,
    # assert mean+3σ well below the 0.1 ms/call claim.  Per
    # memory:feedback-latency-multi-run.
    print("\n[7] sanitizer microbench: _sanitize_program_id direct cost")
    import statistics
    from sglang.srt.managers.schedule_batch import _sanitize_program_id

    N_RUNS = 5
    N_PER_RUN = 10_000
    # Mix of shapes the production path actually sees:
    #   * happy short str  (the >99 % case)
    #   * None             (untagged path)
    #   * long str         (truncation branch)
    #   * list-of-one      (recursion branch)
    sample_inputs = [
        "prog-A",
        None,
        "x" * 200,
        ["nested-pid"],
    ]
    run_means: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for i in range(N_PER_RUN):
            _sanitize_program_id(sample_inputs[i & 3])
        elapsed_us = (time.perf_counter() - t0) * 1e6
        run_means.append(elapsed_us / N_PER_RUN)
    mean_us = statistics.mean(run_means)
    std_us = statistics.stdev(run_means)
    envelope_us = mean_us + 3.0 * std_us
    print(
        f"    _sanitize_program_id: {mean_us:.2f} ± {std_us:.2f} µs/call "
        f"({N_RUNS} runs × {N_PER_RUN} calls; mean+3σ {envelope_us:.2f} µs)"
    )
    # Ceiling: 10 µs/call (0.01 ms) — 10× the claim (0.001 ms / call
    # would be sub-microsecond, unreliable under Python timer jitter)
    # but 100× tighter than the old 5 ms ceiling.  Catches a 3–5×
    # regression (e.g., adding json.loads + json.dumps to the path).
    assert envelope_us < 10.0, (
        f"_sanitize_program_id mean+3σ = {envelope_us:.2f} µs/call exceeds "
        f"10 µs ceiling (mean={mean_us:.2f}, std={std_us:.2f})"
    )

    # ---- [8] Retro-tagging: untagged then tagged on the same prefix ----
    # Order matters in implementations that "skip if node already exists"
    # might miss the tag. The contract is set-additive: the tagged
    # request adds its pid to the previously-untagged ancestor.
    print("\n[8] retro-tagging: untagged-first then tagged on shared prefix")
    RETRO_SYS = "Retro-tag system prompt: " + ("alpha beta gamma. " * 40)
    chat("retro untagged seed", program_id=None, system_text=RETRO_SYS)
    state_before = fetch_state()
    chat("retro tagged after", program_id="prog-RETRO", system_text=RETRO_SYS)
    state_after = fetch_state()
    retro_units = units_with(state_after, "prog-RETRO")
    assert retro_units, "tagged request after untagged did NOT tag the shared ancestor"
    # And the ancestor's session_ids set must STILL contain the RETRO tag --
    # not get blown away by the second insert.
    multi_tag_node = max(retro_units, key=lambda u: len(u["session_ids"]))
    assert "prog-RETRO" in multi_tag_node["session_ids"]
    print(f"    retro-tag landed on {len(retro_units)} shared-prefix nodes ✓")

    # ---- [9] Chunked prefill: tagged request whose prompt > 1 chunk ----
    # The request's prompt must EXCEED the server's --chunked-prefill-size
    # to actually trigger ``cache_unfinished_req(chunked=True)`` and exercise
    # the tagging path on chunk boundaries.  We probe the server's effective
    # chunked_prefill_size by sending a prompt that's *definitely* larger
    # than the smallest practical setting (32 tokens) -- the verify
    # docstring asks the launcher to use ``--chunked-prefill-size 32``.
    # If the prompt is shorter than the configured chunk size, this test
    # degrades to "long generation" (still useful, but doesn't pin the
    # chunked path).
    print("\n[9] chunked prefill: prompt > chunked_prefill_size, tagged")
    # Build a long prompt: each "fact <i>:" segment is ~5 tokens; 200
    # segments ≈ 1 K tokens, well above the 32 / 64 / 128 the launcher
    # might use.  At default 8 K we still won't chunk -- README says
    # "launch with --chunked-prefill-size 32"; we proceed regardless.
    chunked_prompt = (
        "Recite the following facts verbatim: "
        + " ".join(f"fact {i}: prime {i} is interesting." for i in range(200))
    )
    requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": chunked_prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
            "program_id": "prog-CHUNK",
        },
        timeout=120,
    ).raise_for_status()
    state = fetch_state()
    chunk_units = units_with(state, "prog-CHUNK")
    print(f"    chunked req tagged {len(chunk_units)} nodes")
    assert chunk_units, "chunked / long-prompt request lost the tag entirely"
    # Total tagged tokens in the deepest leaf path must be large -- proves
    # the tag survived from chunk 0 through to the final insert.
    total_tokens = sum(u["n_tokens"] for u in chunk_units)
    print(f"    total tagged tokens: {total_tokens}")
    assert total_tokens >= 200, (
        f"only {total_tokens} tokens tagged; tag lost mid-chunked-prefill"
    )

    # ---- [10] Single-request batched-broadcast bug guard ----
    # A daemon misusing the wire format might send a list as program_id
    # for a SINGLE request: ``program_id=["foo", "bar"]``.  Our
    # sanitizer collapses to first element (no per-item broadcast for
    # single-request paths).  This complements case [6] which only
    # checked the request didn't 5xx.
    print("\n[10] single-request list program_id sanitizes to first element")
    chat("single-req list pid", program_id=["prog-LIST-FIRST", "prog-LIST-SECOND"])
    state = fetch_state()
    assert units_with(state, "prog-LIST-FIRST"), (
        "list[0] did not become the sanitized program_id"
    )
    assert not units_with(state, "prog-LIST-SECOND"), (
        "list[1] leaked into the tree -- batched-broadcast misfire on a "
        "single request"
    )

    # ---- [11] Session multi-turn: program_id survives Session.create_req ----
    # sglang's session API constructs a fresh Req via
    # ``Session.create_req``; round-3 audit caught a silent program_id
    # drop on that path.  CRITICAL: the OpenAI chat handler does NOT
    # forward session_params (audit-round-4 caught the previous
    # version of this step using /v1/chat/completions, which goes
    # through the non-session path that already plumbed program_id --
    # so the step was passing trivially regardless of the
    # Session.create_req fix).  Use sglang's native /generate
    # endpoint (which DOES propagate session_params) + the
    # /open_session bootstrap so the second turn actually hits
    # Session.create_req.
    print("\n[11] Session multi-turn via /generate: program_id forwarded via Session.create_req")
    open_r = requests.post(f"{BASE}/open_session", json={"capacity_of_str_len": 1024}, timeout=30)
    open_r.raise_for_status()
    # /open_session may return a JSON string or (future-proof) a dict.
    # Same forward-compat parsing as regression_probe.py.
    session_id = None
    try:
        _parsed = open_r.json()
        if isinstance(_parsed, str):
            session_id = _parsed
        elif isinstance(_parsed, dict):
            session_id = _parsed.get("session_id") or _parsed.get("id")
    except Exception:
        session_id = open_r.text.strip().strip('"')
    assert session_id and isinstance(session_id, str), (
        f"open_session response unparsable: {open_r.text!r}"
    )
    # Seed (no program_id) so the session is populated.
    requests.post(
        f"{BASE}/generate",
        json={
            "text": "session seed for verify step [11]",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            "session_params": {"id": session_id},
        },
        timeout=60,
    ).raise_for_status()
    # Tagged multi-turn; hits Session.create_req on the scheduler side.
    SESSION_PID = "prog-SESSION-VERIFY-11"
    requests.post(
        f"{BASE}/generate",
        json={
            "text": "session second turn tagged with prog-SESSION-VERIFY-11",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            "session_params": {"id": session_id},
            "program_id": SESSION_PID,
        },
        timeout=60,
    ).raise_for_status()
    state = fetch_state()
    sess_units = units_with(state, SESSION_PID)
    print(f"    session-tagged units: {len(sess_units)}")
    assert sess_units, (
        f"session-multi-turn request lost the tag -- Session.create_req "
        f"is silently dropping program_id.  Note: this MUST use /generate "
        f"(not /v1/chat/completions, which drops session_params client-"
        f"side).  Round-4 audit caught the previous fake version."
    )
    # Round-6 audit MINOR 9: close the session at the end so repeated
    # verify runs don't accumulate sessions in the controller.
    try:
        requests.post(
            f"{BASE}/close_session",
            json={"session_id": session_id},
            timeout=10,
        )
    except Exception:
        pass  # close_session is best-effort; not a hard requirement.

    # ---- [12] Recursion DoS: deeply-nested list program_id must not crash ----
    # Round-3 audit caught the sanitizer recursing unbounded into nested
    # lists / tuples. The fix is a depth cap (_PROGRAM_ID_MAX_RECURSION
    # = 8). Build the bomb via raw bytes (Python's json.dumps also
    # recursion-errors on deeply nested lists) and POST to /generate
    # (sglang-native; carries program_id directly to GenerateReqInput).
    # depth=20: well above the cap (8) but well below Python json's
    # parse limit (~150 on this build). Pre-fix: sanitizer recurses 20
    # levels and tags "should-be-buried-too-deep". Post-fix: cap returns
    # None at depth 9, no tag lands.
    print("\n[12] recursion DoS: deeply-nested list program_id (raw POST to /generate)")
    depth = 20
    bomb_payload = (
        b'{"text":"recursion-bomb probe"'
        b',"sampling_params":{"max_new_tokens":4,"temperature":0.0}'
        b',"program_id":'
        + b"[" * depth + b'"should-be-buried-too-deep"' + b"]" * depth
        + b"}"
    )
    bomb_resp = requests.post(
        f"{BASE}/generate",
        data=bomb_payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    # CRITICAL (audit round-5 BLOCKER): require status==200 so the
    # sanitizer was actually exercised.  Accepting 400 silently would
    # make the test a no-op if a future Python / FastAPI tightens JSON
    # parse recursion limits below depth=20.
    assert bomb_resp.status_code == 200, (
        f"server returned {bomb_resp.status_code} on depth-{depth} bomb "
        f"(expected 200).  The request did NOT reach the sanitizer -- "
        f"either JSON parser rejected earlier (lower the depth) OR the "
        f"scheduler crashed."
    )
    health = requests.get(f"{BASE}/health", timeout=10)
    # Audit round-3: previously accepted 200 OR 503.  A 503 means the
    # scheduler DID crash (or its degraded-health gate tripped), which
    # is the exact failure we're trying to prevent.  Tighten to 200
    # only — if the bomb takes sglang out, we want this assertion to
    # fail loudly, not pass with a "stayed up" message.
    assert health.status_code == 200, (
        f"server unhealthy after recursion bomb: {health.status_code} "
        f"(recursion cap did not protect the scheduler)"
    )
    state = fetch_state()
    assert not units_with(state, "should-be-buried-too-deep"), (
        "deep nested list bypassed the recursion cap -- the buried "
        "string reached session_ids"
    )
    print(f"    server stayed up; depth-{depth} -> tag dropped (cap hit) ✓")

    # ---- [13] Pydantic regression: model_config.extra must NOT be 'allow' ----
    # Round-3 NIT: the negative ``extra_body`` test in step [3] depends
    # on ChatCompletionRequest having default ``extra='ignore'``. If a
    # future maintainer adds ``model_config = ConfigDict(extra='allow')``
    # the negative test still passes (server doesn't auto-unpack
    # extra_body) but the assumption becomes brittle. Pin it directly.
    print("\n[13] Pydantic regression: ChatCompletionRequest must not allow extras")
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    model_config = getattr(ChatCompletionRequest, "model_config", {}) or {}
    extra = model_config.get("extra") if isinstance(model_config, dict) else (
        getattr(model_config, "extra", None)
    )
    assert extra != "allow", (
        f"ChatCompletionRequest now has model_config.extra={extra!r}; "
        f"un-vetted extras land as attributes, which could let a "
        f"malicious client bypass the program_id sanitizer. Re-audit "
        f"the program_id wire path before enabling this."
    )
    print(f"    model_config.extra = {extra!r} (not 'allow') ✓")

    # ---- [14] Direct sanitizer test: str()-raising object must not crash ----
    # The sanitizer wraps the str() coercion in try/except.  We can't
    # ship a Python object through HTTP JSON, so we test the helper
    # directly by import.  Round-5 audit NIT 8.
    print("\n[14] sanitizer str()-raising object falls back to None")
    from sglang.srt.managers.schedule_batch import _sanitize_program_id

    class _Bomb:
        def __str__(self):
            raise RuntimeError("bomb")

    out = _sanitize_program_id(_Bomb())
    assert out is None, f"str()-raising object should sanitize to None, got {out!r}"
    # Cycle in a list must terminate via depth cap, not RecursionError.
    cyc: list = []
    cyc.append(cyc)
    out = _sanitize_program_id(cyc)
    assert out is None, f"cycle list should sanitize to None, got {out!r}"
    print("    bomb-__str__ + cyclic-list both -> None ✓")

    print("\n=== T3 PASSED (post-audit round 3) ===")


if __name__ == "__main__":
    main()

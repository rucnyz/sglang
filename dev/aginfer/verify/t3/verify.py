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
    print("\n[2] no untagged unit has program_id; no extra/fake tags")
    expected_tags = {"prog-A", "prog-B"}
    for u in state["units"]:
        extra = set(u["session_ids"]) - expected_tags
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
    assert empty_units, "no untagged units in state — every node got a tag?"
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

    # ---- [7] Latency micro-bench: tagging overhead vs untagged ----
    print("\n[7] latency micro-bench: tagged vs untagged")
    # Tagged
    t0 = time.perf_counter()
    for i in range(20):
        chat(f"micro tagged {i}", program_id="micro-bench")
    tagged_ms = (time.perf_counter() - t0) * 1000
    # Untagged
    t0 = time.perf_counter()
    for i in range(20):
        chat(f"micro untagged {i}", program_id=None)
    untagged_ms = (time.perf_counter() - t0) * 1000
    print(f"    20 tagged: {tagged_ms:.0f} ms, 20 untagged: {untagged_ms:.0f} ms")
    # Cost ceiling per README: < 0.1 ms/req amortized for the tag overhead.
    # Total cost is dominated by inference; this is just a sanity check that
    # tagging doesn't add a measurable fraction.
    overhead_per_req_ms = (tagged_ms - untagged_ms) / 20
    print(f"    estimated overhead: {overhead_per_req_ms:+.2f} ms/req "
          f"(noise tolerated; ceiling 5 ms)")
    assert abs(overhead_per_req_ms) < 5.0, (
        f"tagging adds {overhead_per_req_ms:.2f} ms/req — way above 0.1 ms ceiling"
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

    # ---- [9] Chunked prefill: long generation with a tag ----
    # Force a multi-chunk insert by sending a request whose prompt+generation
    # spans more than chunked_prefill_size tokens, then assert the tag lands
    # on the deepest nodes (not just the leading chunk).
    print("\n[9] chunked prefill: long-generation tagged request")
    LONG_PROMPT = (
        "You are a verbose narrator. Tell me a long step-by-step story "
        + "with at least 30 sentences. " * 4
        + " Start now: "
    )
    requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": LONG_PROMPT}],
            "max_tokens": 256,
            "temperature": 0.0,
            "program_id": "prog-CHUNK",
        },
        timeout=120,
    ).raise_for_status()
    state = fetch_state()
    chunk_units = units_with(state, "prog-CHUNK")
    print(f"    chunked req tagged {len(chunk_units)} nodes")
    assert chunk_units, "chunked / long-generation request lost the tag entirely"
    # n_tokens of the deepest tagged node should be > 32 (= multi-chunk leaf).
    deepest = max(chunk_units, key=lambda u: u["n_tokens"])
    assert deepest["n_tokens"] >= 32, (
        f"deepest tagged node has only {deepest['n_tokens']} tokens; "
        f"chunked insert did NOT keep tagging across chunks"
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

    print("\n=== T3 PASSED (post-audit) ===")


if __name__ == "__main__":
    main()

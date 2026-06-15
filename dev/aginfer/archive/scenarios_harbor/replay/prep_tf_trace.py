#!/usr/bin/env python3
"""Prepare a TEACHER-FORCING-ready replay trace from raw multi-turn
conversations (wherewewin prerequisite, task #234).

Length-only replay (``max_tokens`` + ``ignore_eos``, content = whatever the
model argmaxes) breaks multi-turn KV continuation: turn N's *generated* output
≠ the *captured* output embedded in turn N+1's prompt, so turn N+1 re-prefills
the output segment instead of hitting cache (proven in
``wherewewin/harness/teacher_forcing`` Part B: 272 vs 16 re-prefilled tokens).

Teacher-forcing fixes it by forcing turn N to emit the EXACT tokens that the
chat template renders for turn N's assistant content, so turn N's KV ==
``tokenize(messages[:N+1])`` and turn N+1's prompt has that as a prefix → hits.

The faithful way to get those tokens — without re-implementing the model's chat
template — is to ask the SERVER's ``/tokenize`` endpoint (it applies the exact
launched template).  Per assistant turn at message index j:

    prefix = tokenize(messages[:j],   add_generation_prompt=True)   # server prefill
    full   = tokenize(messages[:j+1], add_generation_prompt=False)  # + assistant turn
    assert full[:len(prefix)] == prefix            # template prefix-stable
    forced_O_j = full[len(prefix):]                # exactly turn j's emitted tokens

Because the server prefills exactly ``prefix`` and we force exactly
``forced_O_j``, turn j's KV becomes ``full`` — the EXACT prefix of turn j+1's
prefill prompt (``tokenize(messages[:j+1] + next user, gen=True)`` extends
``full``).  So turn j+1 hits ALL of turn j's KV.  We assert prefix-stability per
turn and skip a turn that violates it (logged) — a short-content boundary effect
that would make the forced tokens unfaithful — rather than emit a bad record.

Output: one JSONL record per assistant turn, in the replay_driver record shape
plus ``forced_output_ids`` (build_payload TF mode plumbs it into
``custom_params.forced_output_ids``):

    {"program_id", "turn", "t", "ref_e2e_ms", "body": {"model", "messages"},
     "output_len", "forced_output_ids": [int, ...]}

Run against a LIVE sglang for this model (any state — /tokenize is stateless):
    python prep_tf_trace.py --in data/cc_long_traces.jsonl --out tf.jsonl \
        --server-url http://127.0.0.1:30000 --limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def iter_conversations(path: str, limit: int | None):
    n = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages")
            if not msgs:
                continue
            yield msgs
            n += 1
            if limit and n >= limit:
                return


def _normalize(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep role+string-content messages the chat template accepts.  List
    (multimodal / CC tool-block) content is flattened to text; a non-standard
    role collapses to user/assistant/system so the template tokenizes."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text") or p.get("content") or ""))
                else:
                    parts.append(str(p))
            content = "\n".join(x for x in parts if x)
        if content is None:
            content = ""
        if role not in ("user", "assistant", "system"):
            role = "user"
        out.append({"role": role, "content": str(content)})
    return out


class _Tokenizer:
    """Thin client over sglang's POST /tokenize (applies the launched chat
    template server-side).  Memoises by (len, gen) is unsafe — messages differ
    — so we just POST each call; tokenization is cheap."""
    def __init__(self, server_url: str) -> None:
        self.url = server_url.rstrip("/") + "/tokenize"
        self.sess = requests.Session()

    def tokens(self, messages: List[Dict[str, Any]], gen: bool) -> List[int]:
        r = self.sess.post(self.url, json={"messages": messages,
                                           "add_generation_prompt": gen},
                           timeout=60)
        r.raise_for_status()
        return [int(x) for x in r.json()["tokens"]]


def build_records(messages: List[Dict[str, Any]], tok: _Tokenizer, model: str,
                  program_id: str, max_turns: int | None,
                  skips: Dict[str, int]) -> List[Dict[str, Any]]:
    msgs = _normalize(messages)
    records: List[Dict[str, Any]] = []
    turn = 0
    for j, m in enumerate(msgs):
        if m["role"] != "assistant" or j == 0:
            continue
        try:
            prefix = tok.tokens(msgs[:j], gen=True)
            full = tok.tokens(msgs[:j + 1], gen=False)
        except Exception:
            skips["tokenize_error"] += 1
            continue
        if full[:len(prefix)] != prefix:
            skips["not_prefix_stable"] += 1
            continue
        forced = full[len(prefix):]
        if not forced:
            skips["empty_output"] += 1
            continue
        records.append({
            "program_id": program_id,
            "turn": turn,
            "t": float(turn),                  # synthetic monotonic arrival
            "ref_e2e_ms": float(len(forced)),  # placeholder; real timing later
            "body": {"model": model, "messages": msgs[:j]},
            "output_len": len(forced),
            "forced_output_ids": forced,
        })
        turn += 1
        if max_turns and turn >= max_turns:
            break
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="raw conversations JSONL ({\"messages\":[...]})")
    ap.add_argument("--out", required=True, help="TF-ready records JSONL")
    ap.add_argument("--server-url", default="http://127.0.0.1:30000",
                    help="live sglang base url (for /tokenize)")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--limit", type=int, default=None,
                    help="max conversations (programs) to convert")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="cap assistant turns per program")
    ap.add_argument("--program-prefix", default="cc")
    a = ap.parse_args()

    tok = _Tokenizer(a.server_url)
    # fail fast if the server is unreachable / wrong
    try:
        tok.tokens([{"role": "user", "content": "ping"}], gen=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[prep_tf] cannot reach /tokenize at {a.server_url}: {exc}",
              file=sys.stderr)
        return 2

    skips = {"tokenize_error": 0, "not_prefix_stable": 0, "empty_output": 0}
    n_prog = n_rec = tot_forced = 0
    with open(a.out, "w") as out:
        for idx, msgs in enumerate(iter_conversations(a.inp, a.limit)):
            pid = f"{a.program_prefix}{idx}"
            recs = build_records(msgs, tok, a.model, pid, a.max_turns, skips)
            if not recs:
                continue
            n_prog += 1
            for r in recs:
                out.write(json.dumps(r) + "\n")
                n_rec += 1
                tot_forced += r["output_len"]
            if n_prog % 10 == 0:
                print(f"[prep_tf] {n_prog} programs, {n_rec} records ...",
                      file=sys.stderr)
    print(f"[prep_tf] wrote {n_rec} TF records across {n_prog} programs "
          f"({tot_forced} forced tokens) -> {a.out}", file=sys.stderr)
    print(f"[prep_tf] skips: {skips}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

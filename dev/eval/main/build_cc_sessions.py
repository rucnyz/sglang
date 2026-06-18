#!/usr/bin/env python3
"""Faithful Claude-Code session store for an open-loop concurrent replay.

Goal: reconstruct *what the server actually saw* — every LLM request stream a
real CC run produced, on its real timeline, including subagent fan-out. This is
the substrate for an open-loop simulator (requests arrive by timestamp; the
server's own speed does not pace them), i.e. a faithful "concurrent Claude Code"
load rather than a fixed-concurrency hammer.

A *stream* = one conversation thread = one growing context the server serves:
  - the main session thread, and
  - each subagent thread (its own context; born when the parent spawned it).
Main ∩ subagent = ∅ — the main thread only carries the subagent's returned
tool_result, never its internal turns, so adding subagents is pure new load.

Output (one file per stream, `cc_sessions/<name>.jsonl`), one event per line:
  {"ts": float|null,            # epoch seconds (None for cc_long; no timing)
   "role": "user"|"assistant",
   "kind": "user_text"|"tool_result"|"assistant"|"compact",
   "chars": int,
   "text": str}                 # full content — the prompt material, lossless
Each `assistant` event = one LLM request: its input is the cumulative stream
text since the last `compact`; its output length is this event's tokens. A
`compact` event resets that stream's context (real in-session compaction).

`cc_sessions/manifest.jsonl`, one line per stream, carries the timeline + tree
linkage (parent_session_id, spawn_ts) and a coarse length estimate so a case is
built by filtering the manifest. Token counts here are char/CHARS_PER_TOK
estimates; the simulator tokenizes the served model's tokenizer on the text.
"""

import argparse
import glob
import json
import os
from datetime import datetime

CHARS_PER_TOK = 4.67  # measured on this corpus with the Qwen3.5 tokenizer
SKIP_TYPES = {"file-history-snapshot", "permission-mode", "queue-operation",
              "attachment", "last-prompt", "ai-title", "system"}


def parse_ts(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def text_of(content):
    """Flatten message content (str | list of blocks) to text, lossless for the
    prompt: tool_result bodies kept, tool_use serialized as the model emitted."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    parts.append(b["text"])
                elif b.get("type") == "tool_result":
                    parts.append(text_of(b.get("content", "")))
                elif b.get("type") == "tool_use":
                    parts.append(json.dumps(b.get("input", "")))
                elif "content" in b:
                    parts.append(text_of(b["content"]))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return str(content)


def event_kind(ev_type, content):
    if isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ):
        return "tool_result"
    return "assistant" if ev_type == "assistant" else "user_text"


def emit_stream(out_dir, name, events):
    """events = list of dicts {ts, role, kind, text}. Write the stream file and
    return a manifest stub (or None if below the floor)."""
    llm = [e for e in events if e["role"] == "assistant"]
    if not llm or sum(len(e["text"]) for e in events) < 500:
        return None
    with open(os.path.join(out_dir, name), "w") as fh:
        for e in events:
            fh.write(json.dumps({
                "ts": e["ts"], "role": e["role"], "kind": e["kind"],
                "chars": len(e["text"]), "text": e["text"],
            }) + "\n")
    cur = peak = user_tok = 0
    ncomp = 0
    for e in events:
        if e["kind"] == "compact":
            ncomp += 1
            cur = len(e["text"])
        else:
            cur += len(e["text"])
        peak = max(peak, cur)
        if e["role"] == "user":
            user_tok += len(e["text"])
    ts = [e["ts"] for e in events if e["ts"] is not None]
    return {
        "file": name, "n_events": len(events), "n_llm_requests": len(llm),
        "n_compactions": ncomp, "first_ts": min(ts) if ts else None,
        "last_ts": max(ts) if ts else None,
        "est_peak_ctx_tok": round(peak / CHARS_PER_TOK),
        "est_user_tok": round(user_tok / CHARS_PER_TOK),
    }


def read_thread(path):
    """One raw CC jsonl -> ordered conversation events for that single thread
    (non-sidechain main file, or one subagent file). Compaction kept as a
    marker; non-conversational event types dropped."""
    out = []
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = parse_ts(r.get("timestamp"))
        if r.get("isCompactSummary"):
            msg = r.get("message") or {}
            out.append({"ts": ts, "role": "user", "kind": "compact",
                        "text": text_of(msg.get("content", ""))})
            continue
        if r.get("isMeta") or r.get("type") in SKIP_TYPES:
            continue
        if r.get("type") not in ("user", "assistant"):
            continue
        msg = r.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        out.append({"ts": ts, "role": r["type"],
                    "kind": event_kind(r["type"], content),
                    "text": text_of(content)})
    return out


def ingest_local(main_path, out_dir, manifest):
    """A local session = its main thread + every subagent thread under it. Each
    becomes its own stream; subagents are tagged with parent + spawn_ts."""
    base = main_path[:-len(".jsonl")]
    project = os.path.basename(os.path.dirname(main_path))
    sess = os.path.basename(base)
    n = 0

    main_events = [e for e in read_thread(main_path)]  # main file holds no sidechain
    stub = emit_stream(out_dir, f"local__{project[-22:]}__{sess[:8]}.jsonl", main_events)
    if stub:
        stub.update(source="local_claude", project=project, session_id=sess,
                    agent_id=None, parent_session_id=None, spawn_ts=None)
        manifest.append(stub); n += 1

    for sf in sorted(glob.glob(base + "/**/*.jsonl", recursive=True)):
        events = read_thread(sf)
        first = next((json.loads(l) for l in open(sf) if l.strip()), {})
        agent_id = first.get("agentId") or os.path.basename(sf)[:16]
        stub = emit_stream(out_dir, f"local__{project[-22:]}__{sess[:8]}__{agent_id}.jsonl", events)
        if stub:
            spawn = next((e["ts"] for e in events if e["ts"] is not None), None)
            stub.update(source="local_subagent", project=project, session_id=sess,
                        agent_id=agent_id, parent_session_id=first.get("sessionId", sess),
                        spawn_ts=spawn)
            manifest.append(stub); n += 1
    return n


def ingest_cc_long(path, out_dir, manifest):
    """cc_long_traces: pre-extracted {messages:[...]} per line, no timing, no
    subagents. Kept as timing-less streams (open-loop must assign synthetic
    arrivals; usable as-is closed-loop)."""
    n = 0
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        rec = json.loads(line)
        events = [{"ts": None, "role": m.get("role", "user"),
                   "kind": event_kind(m.get("role"), m.get("content", "")),
                   "text": text_of(m.get("content", ""))}
                  for m in rec.get("messages", [])]
        stub = emit_stream(out_dir, f"cclong__{i:03d}.jsonl", events)
        if stub:
            stub.update(source="cc_long", project="hyperswitch", session_id=f"cclong{i:03d}",
                        agent_id=None, parent_session_id=None, spawn_ts=None)
            manifest.append(stub); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc-long", default="dev/eval/datasets/cc_long_traces.jsonl")
    ap.add_argument("--local-glob", default=os.path.expanduser("~/.claude/projects/*/*.jsonl"))
    ap.add_argument("--out", default="dev/eval/datasets/cc_sessions")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    n_cc = ingest_cc_long(args.cc_long, args.out, manifest)
    n_local = sum(ingest_local(f, args.out, manifest) for f in sorted(glob.glob(args.local_glob)))

    cols = ["file", "source", "project", "session_id", "agent_id", "parent_session_id",
            "spawn_ts", "first_ts", "last_ts", "n_events", "n_llm_requests",
            "n_compactions", "est_peak_ctx_tok", "est_user_tok"]
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fh:
        for r in sorted(manifest, key=lambda r: -r["est_peak_ctx_tok"]):
            fh.write(json.dumps({c: r.get(c) for c in cols}) + "\n")

    main_streams = [r for r in manifest if r["source"] in ("cc_long", "local_claude")]
    subs = [r for r in manifest if r["source"] == "local_subagent"]
    print(f"main streams: cc_long={n_cc} local={len(main_streams)-n_cc}")
    print(f"subagent streams: {len(subs)}")
    print(f"total streams: {len(manifest)} -> {args.out}")
    for thr in (16000, 32000, 64000, 100000):
        print(f"  main streams est_peak_ctx >= {thr//1000:>3}k: "
              f"{sum(1 for r in main_streams if r['est_peak_ctx_tok'] >= thr)}")
    print(f"  subagent streams median LLM-requests: "
          f"{sorted(s['n_llm_requests'] for s in subs)[len(subs)//2] if subs else 0}")


if __name__ == "__main__":
    main()

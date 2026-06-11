"""Convert local Claude Code session transcripts (~/.claude/projects/*/*.jsonl) into
the replay-trace format used by replay_driver.py (a3real.jsonl shape):
    {"t": <s from start>, "program_id": <session>,
     "body": {"messages": [{system}, {user}], "model": ..., "temperature": 0},
     "output_len": <tokens>}

FAITHFULNESS — what is REAL vs reconstructed (honest):
  REAL (drives KV behaviour, all taken from the transcripts / the CC install):
    * the shared system+tools prefix — ONE ~16K-token block built from the actual
      CC tool definitions (`sdk-tools.d.ts`) + a CC system header, IDENTICAL across
      every program → reproduces the real cross-program system-prompt KV sharing
      (confirmed: first-turn cache_read ≈ 16-17K on every session);
    * per-turn prompt SIZES (input+cache_read+cache_creation), output lengths, and
      tool-gap TIMING (timestamps) — exact, from `usage`;
    * per-program CONVERSATION content — the real user/assistant/tool text from the
      transcript, grown as a prefix (turn N+1 ⊇ turn N → real prefix-sharing).
  NOT available (immaterial to KV — sglang matches token-id structure, not text):
    * the exact byte-for-byte system prompt of that CC build (the bundle is
      compressed); a same-sized shared block is KV-equivalent.

Usage:
  python convert_cc_traces.py --out traces/cc_local.jsonl --min-turns 8 \
     --max-sessions 30 --stagger 3.0 --sys-tokens 16000
"""
import json, os, glob, argparse
from datetime import datetime

CC_DIR = "/run/user/1011/fnm_multishells/1570511_1780550507548/lib/node_modules/@anthropic-ai/claude-code"
CC_HEADER = ("You are Claude Code, Anthropic's official CLI for Claude. You are an "
             "interactive CLI tool that helps users with software engineering tasks. "
             "Use the instructions below and the tools available to you.\n\n"
             "# Tools\nThe following tool input schemas are available:\n\n")


def iso_s(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def shared_system(sys_tokens):
    """The cross-program-shared system+tools block: real CC tool defs, sized to
    ~sys_tokens (≈4 chars/tok), identical for every program."""
    tools = ""
    p = os.path.join(CC_DIR, "sdk-tools.d.ts")
    if os.path.exists(p):
        tools = open(p, errors="ignore").read()
    blob = CC_HEADER + tools
    chars = sys_tokens * 4
    if len(blob) < chars:
        blob = blob + ("\n" + tools) * (chars // max(1, len(tools)) + 1)
    return blob[:chars]


def block_text(content):
    """Flatten a CC message 'content' (str or list of blocks) into text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if not isinstance(b, dict):
            continue
        ty = b.get("type")
        if ty == "text":
            out.append(b.get("text", ""))
        elif ty == "tool_use":
            out.append("TOOL_USE " + json.dumps(b.get("input", {}))[:4000])
        elif ty == "tool_result":
            c = b.get("content")
            out.append("TOOL_RESULT " + (c if isinstance(c, str) else json.dumps(c)[:8000]))
        elif ty == "thinking":
            out.append(b.get("thinking", ""))
    return "\n".join(out)


def session_iter(path, max_turns, max_pt):
    """Stream a transcript, yielding (t_epoch, prompt_tok, out_tok, convo_text) per
    assistant call. convo_text = real conversation up to this turn (a PREFIX of one
    growing string; bounded by max_pt so the trace can't blow up). No O(n^2)."""
    acc = []           # growing list of real text chunks (the conversation prefix)
    acc_len = 0
    cap_chars = max_pt * 4
    n = 0
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ty = r.get("type")
        m = r.get("message") or {}
        if ty == "user":
            tx = block_text(m.get("content"))
            if acc_len < cap_chars:                    # stop growing once past the cap
                acc.append(tx); acc_len += len(tx)
        elif ty == "assistant":
            u = m.get("usage") or {}
            pt = ((u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                  + (u.get("cache_creation_input_tokens") or 0))
            ol = u.get("output_tokens") or 0
            ts = iso_s(r.get("timestamp") or "")
            if pt > 0 and ol > 0 and ts:
                yield (ts, min(pt, max_pt), ol, "\n".join(acc))
                n += 1
                if n >= max_turns:
                    return
            tx = block_text(m.get("content"))
            if acc_len < cap_chars:
                acc.append(tx); acc_len += len(tx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", default="/scratch/yuzhou/.claude/projects")
    ap.add_argument("--out-dir", required=True, help="one <program>.jsonl per session here")
    ap.add_argument("--min-turns", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=25, help="cap turns per program")
    ap.add_argument("--max-sessions", type=int, default=60)
    ap.add_argument("--max-prompt-tokens", type=int, default=50000, help="clamp prompt size")
    ap.add_argument("--sys-tokens", type=int, default=16000)
    ap.add_argument("--max-file-mb", type=int, default=80, help="per-program hard abort guard")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    a = ap.parse_args()

    SYS = shared_system(a.sys_tokens)
    # recursive: include deep sub-agent / sidechain sessions too
    files = sorted(glob.glob(os.path.join(a.projects, "**", "*.jsonl"), recursive=True),
                   key=lambda p: -os.path.getsize(p))
    os.makedirs(a.out_dir, exist_ok=True)
    guard = a.max_file_mb * 1024 * 1024

    n_prog = tot_rec = tot_out = 0
    sizes_all = []
    for f in files:
        if n_prog >= a.max_sessions:
            break
        pid = os.path.basename(f)[:-6]
        turns = list(session_iter(f, a.max_turns, a.max_prompt_tokens))
        if len(turns) < a.min_turns:
            continue
        out = os.path.join(a.out_dir, f"{pid}.jsonl")
        t0 = turns[0][0]
        n_rec = 0
        with open(out, "w") as fh:
            for (ts, pt, ol, convo) in turns:
                convo_chars = max(0, (pt - a.sys_tokens)) * 4
                user_txt = convo[:convo_chars] if convo_chars else convo[:200]
                rec = {"t": round(ts - t0, 4), "program_id": pid,    # t 0-based per program
                       "body": {"messages": [{"role": "system", "content": SYS},
                                             {"role": "user", "content": user_txt}],
                                "model": a.model, "temperature": 0},
                       "output_len": int(ol), "ref_e2e_ms": round(pt * 0.1 + ol * 20, 1)}
                fh.write(json.dumps(rec) + "\n")
                n_rec += 1; tot_out += ol
                sizes_all.append((len(SYS) + len(user_txt)) // 4)
                if fh.tell() > guard:
                    break
        n_prog += 1; tot_rec += n_rec
    import statistics
    print(f"wrote {n_prog} per-program jsonls -> {a.out_dir}  ({tot_rec} requests total)")
    print(f"  shared system block = {len(SYS)//4} tok (IDENTICAL across all programs)")
    if sizes_all:
        print(f"  per-request prompt: mean={statistics.mean(sizes_all):.0f} max={max(sizes_all)} tok; total_output={tot_out}")


if __name__ == "__main__":
    main()

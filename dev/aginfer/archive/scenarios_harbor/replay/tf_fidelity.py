#!/usr/bin/env python3
"""TF replay fidelity check — does teacher-forcing the prepped output token ids
reproduce real multi-turn KV reuse, where length-only replay does not?

This is the integration gate for plumbing teacher-forcing into the replay
(task #234 → wherewewin): Part B proved the *mechanism* is faithful in an
isolated 2-turn token-space test; here we prove the prep+replay PIPELINE on a
REAL multi-turn conversation, measured by sglang's own ``cached_tokens``.

For one multi-turn program from a prepped TF trace (prep_tf_trace.py), replay
its assistant turns sequentially and, at each turn j>=1, read how many of the
prompt tokens hit the radix cache (``cached_tokens``):

  TF arm        : force turn j-1 to emit its captured tokens ``F_{j-1}`` → turn
                  j-1's KV == the chat-template rendering ``full_{j-1}``, which
                  is the prefix of turn j's prompt → turn j reuses ~all of it.
  length-only   : let turn j-1 argmax (content != captured) → turn j's prompt
                  (which embeds the CAPTURED content) diverges at the assistant
                  segment → reuse collapses to the pre-content prefix.

We replay at the /generate level using the EXACT chat-template token sequences
(prompt = server /tokenize(messages[:j], gen=True)), so cached_tokens is clean
and authoritative; these are the identical tokens the chat path would prefill,
so the reuse measured here is the reuse the chat replay gets.  (The chat path's
custom_params plumbing itself is covered by harness C4.)

PASS: mean TF reuse >> mean length-only reuse, and TF reuse ≈ prior turn's full
length (faithful continuation).

Run: sglang up with --enable-cache-report.
  python tf_fidelity.py --trace tf.jsonl --program cc0 --base-url http://127.0.0.1:30000
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List, Optional

import requests


def load_program(trace_path: str, program: Optional[str]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with open(trace_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs.append(r)
    if program is None:
        program = recs[0]["program_id"]
    recs = [r for r in recs if r.get("program_id") == program]
    recs.sort(key=lambda r: r.get("turn", 0))
    return recs


def tokenize(base: str, messages, gen: bool) -> List[int]:
    r = requests.post(base.rstrip("/") + "/tokenize",
                      json={"messages": messages, "add_generation_prompt": gen},
                      timeout=60)
    r.raise_for_status()
    return [int(x) for x in r.json()["tokens"]]


def generate(base: str, input_ids: List[int], max_new: int,
             forced: Optional[List[int]]) -> Dict[str, Any]:
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    r = requests.post(base.rstrip("/") + "/generate",
                      json={"input_ids": input_ids, "sampling_params": sp,
                            "stream": False}, timeout=600)
    r.raise_for_status()
    mi = r.json()["meta_info"]
    return {"cached": int(mi.get("cached_tokens") or 0),
            "prompt": int(mi.get("prompt_tokens") or len(input_ids))}


def replay_arm(base: str, recs: List[Dict[str, Any]], *, force: bool) -> List[Dict[str, Any]]:
    """Replay the program's turns sequentially; return per-turn reuse rows.
    flush first so the only cache content is what THIS arm builds."""
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)
    rows: List[Dict[str, Any]] = []
    prev_full = 0
    for j, r in enumerate(recs):
        prompt_ids = tokenize(base, r["body"]["messages"], gen=True)
        forced = r["forced_output_ids"] if force else None
        out = generate(base, prompt_ids, r["output_len"], forced)
        rows.append({
            "turn": r.get("turn", j),
            "prompt_len": out["prompt"],
            "cached": out["cached"],
            "reuse_frac": (out["cached"] / out["prompt"]) if out["prompt"] else 0.0,
            "prev_full": prev_full,
        })
        # turn j's KV after this call = prompt_ids + emitted (full_j under TF)
        prev_full = len(prompt_ids) + r["output_len"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--program", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    recs = load_program(a.trace, a.program)
    if len(recs) < 2:
        print(f"need a multi-turn program; got {len(recs)} turns", file=sys.stderr)
        return 2
    pid = recs[0]["program_id"]
    print(f"=== TF replay fidelity — program {pid}, {len(recs)} turns ===")

    tf = replay_arm(a.base_url, recs, force=True)
    lo = replay_arm(a.base_url, recs, force=False)

    # turn 0 has nothing to reuse (cold); compare turns >= 1
    tf_r = [r["cached"] for r in tf[1:]]
    lo_r = [r["cached"] for r in lo[1:]]
    print(f"{'turn':>4} {'prompt':>7} {'TF_cached':>10} {'TF_reuse%':>9} "
          f"{'LO_cached':>10} {'LO_reuse%':>9}")
    for t, l in zip(tf[1:], lo[1:]):
        print(f"{t['turn']:>4} {t['prompt_len']:>7} {t['cached']:>10} "
              f"{100*t['reuse_frac']:>8.1f}% {l['cached']:>10} "
              f"{100*l['reuse_frac']:>8.1f}%")
    tf_mean = statistics.mean(tf_r) if tf_r else 0
    lo_mean = statistics.mean(lo_r) if lo_r else 0
    # faithful continuation: TF turn-j cache hit ≈ prior turn's full length
    faithful = all(t["cached"] >= 0.95 * t["prev_full"] - 256 for t in tf[1:])
    win = tf_mean > 1.5 * max(lo_mean, 1)
    print(f"\nmean cached tokens (turns>=1): TF={tf_mean:.0f}  length-only={lo_mean:.0f}  "
          f"ratio={tf_mean/max(lo_mean,1):.1f}x")
    print(f"TF reuse ≈ prior full length (faithful continuation): {faithful}")
    verdict = win and faithful
    print(f"RESULT: {'PASS' if verdict else 'REVIEW'} — TF reproduces multi-turn "
          f"reuse; length-only loses it" if verdict else
          f"RESULT: REVIEW (win={win} faithful={faithful})")
    print(json.dumps({"program": pid, "turns": len(recs), "tf_mean_cached": tf_mean,
                      "lo_mean_cached": lo_mean, "faithful": faithful, "pass": verdict}))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

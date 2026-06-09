#!/usr/bin/env python3
"""S1 driver — predictive promote-back across tool gaps (pure token-space).

DESIGN §3/§7 + wherewewin/s1. Each "program" runs a reasoning↔tool loop:
emit an LLM turn, go off-GPU for a tool of KNOWN ETA, resume with the result
appended. Under HBM pressure the program's idle tail is demoted during the gap;
the action-timeline schedules a promote-back at ``T_start+ETA−load_back`` so the
tail is HBM-resident when the resume prefill arrives → resume pays ~0 load_back.

Why pure token-space (not the chat replay_driver): the model's default chat
template is not prefix-stable across turns, so the chat path can't reproduce
multi-turn KV reuse; and the proxy emits TOOL_CALL_START with no ETA. So we:
  * build ONE growing token sequence per program (faithful continuation BY
    CONSTRUCTION — Part B), forcing each assistant segment via
    custom_params.forced_output_ids, sent as /generate(input_ids) to sglang
    directly with ``program_id`` so sglang tags the KV units' session_ids;
  * drive the daemon's belief by POSTing events to /aginfer/event ourselves,
    carrying ``tool_eta_s`` on TOOL_CALL_START so the promote schedules.

Arms (``--arm``):
  ours  — full daemon (kv+admission), events injected, promote active.
  b     — HiCache only: still inject SESSION/PREFILL so attribution works, but
          NO tool-gap events (no demote/promote signal) → on-access load_back.
  (ThunderAgent arm added later — admission-only.)

Milestone-1 goal (this script): show the promote actually FIRES end-to-end
(daemon promotes_scheduled>0 → promotes>0) and the resume prefill lands on HBM.
Measurement of resume TTFT vs B/TA is layered on top once firing is confirmed.

Run against a live a3 stack (sglang :30000 + daemon :9100):
  python s1_driver.py --programs 8 --turns 4 --prefix-tokens 24000 \
      --output-tokens 1500 --gap-s 6 --tool-eta-s 6 --arm ours
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests


def post_event(daemon: str, kind: str, session: str,
               extra: Optional[Dict[str, Any]] = None) -> None:
    body = {"kind": kind, "session": session}
    if extra:
        body.update(extra)
    try:
        requests.post(daemon.rstrip("/") + "/aginfer/event", json=body, timeout=10)
    except Exception as e:  # noqa: BLE001
        print(f"[s1] event {kind} for {session} failed: {e}", file=sys.stderr)


def generate(base: str, input_ids: List[int], max_new: int,
             forced: Optional[List[int]], program_id: str,
             stream: bool) -> Dict[str, Any]:
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    body = {"input_ids": input_ids, "sampling_params": sp,
            "program_id": program_id, "stream": stream}
    if not stream:
        t0 = time.perf_counter()
        r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=900)
        e2e = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        mi = r.json()["meta_info"]
        return {"ttft_ms": None, "e2e_ms": e2e,
                "cached": int(mi.get("cached_tokens") or 0),
                "prompt": int(mi.get("prompt_tokens") or len(input_ids))}
    # streaming: measure TTFT = time to first token chunk
    t0 = time.perf_counter()
    ttft = None
    cached = prompt = 0
    with requests.post(base.rstrip("/") + "/generate", json=body, timeout=900,
                       stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            s = line.decode() if isinstance(line, bytes) else line
            if s.startswith("data:"):
                s = s[5:].strip()
            if s and s != "[DONE]":
                try:
                    mi = json.loads(s).get("meta_info") or {}
                    cached = int(mi.get("cached_tokens") or cached)
                    prompt = int(mi.get("prompt_tokens") or prompt)
                except Exception:
                    pass
    e2e = (time.perf_counter() - t0) * 1000.0
    return {"ttft_ms": ttft, "e2e_ms": e2e, "cached": cached, "prompt": prompt}


def run_program(idx: int, args) -> Dict[str, Any]:
    pid = f"s1p{idx}"
    base, daemon = args.base_url, args.daemon_url
    inject = (args.arm in ("ours", "ta"))           # b = HiCache-only, no gap events
    # Stagger program starts so the per-program prefills don't pile into one
    # giant concurrent batch. Staggering bounds the instantaneous prefill width
    # while still building enough working set to pressure a small KV pool.
    if args.stagger_s:
        time.sleep(idx * args.stagger_s)
    # distinct, page-aligned base prefix per program (no cross-program dedup)
    salt = 100000 + idx * 50000
    seq = list(range(salt, salt + args.prefix_tokens))
    if inject:
        post_event(daemon, "session_arrival", pid)
    resume_rows: List[Dict[str, Any]] = []
    for turn in range(args.turns):
        out = list(range(salt + 1_000_000 + turn * 10000,
                         salt + 1_000_000 + turn * 10000 + args.output_tokens))
        if inject:
            post_event(daemon, "llm_prefill", pid)
        is_resume = turn > 0
        g = generate(base, seq, len(out), out, pid, stream=is_resume)
        if is_resume:
            resume_rows.append({"turn": turn, **g})
        seq = seq + out
        if turn < args.turns - 1:
            if inject:
                post_event(daemon, "tool_call_start", pid,
                           {"tool_eta_s": args.tool_eta_s})
            time.sleep(args.gap_s)
            if inject:
                post_event(daemon, "tool_call_end", pid)
    if inject:
        post_event(daemon, "session_end", pid)
    return {"pid": pid, "resume": resume_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--daemon-url", default="http://127.0.0.1:9100")
    ap.add_argument("--programs", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--prefix-tokens", type=int, default=24000)
    ap.add_argument("--output-tokens", type=int, default=1500)
    ap.add_argument("--gap-s", type=float, default=6.0)
    ap.add_argument("--tool-eta-s", type=float, default=6.0)
    ap.add_argument("--stagger-s", type=float, default=1.5,
                    help="delay between program starts (avoids prefill pile-up)")
    ap.add_argument("--arm", choices=["ours", "b", "ta"], default="ours")
    ap.add_argument("--out", default=None, help="write per-resume rows JSONL")
    a = ap.parse_args()
    print(f"=== S1 driver arm={a.arm} programs={a.programs} turns={a.turns} "
          f"prefix={a.prefix_tokens} out={a.output_tokens} gap={a.gap_s}s "
          f"eta={a.tool_eta_s}s ===")
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=a.programs) as ex:
        futs = [ex.submit(run_program, i, a) for i in range(a.programs)]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"[s1] program failed: {e}", file=sys.stderr)
    wall = time.perf_counter() - t0
    resume = [r for res in results for r in res["resume"]]
    ttfts = [r["ttft_ms"] for r in resume if r.get("ttft_ms") is not None]
    cached = [r["cached"] for r in resume]
    print(f"\n=== S1 arm={a.arm} result (wall {wall:.1f}s) ===")
    print(f"resume prefills: {len(resume)}")
    if ttfts:
        ttfts_s = sorted(ttfts)
        n = len(ttfts_s)
        mean = sum(ttfts_s) / n
        p50 = ttfts_s[n // 2]
        p99 = ttfts_s[min(n - 1, int(0.99 * n))]
        print(f"resume TTFT ms: mean={mean:.1f} p50={p50:.1f} p99={p99:.1f}")
    if cached:
        print(f"resume cached_tokens: mean={sum(cached)/len(cached):.0f} "
              f"(prefix={a.prefix_tokens})")
    if a.out:
        with open(a.out, "w") as fh:
            for r in resume:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote per-resume rows -> {a.out}")
    print(json.dumps({"arm": a.arm, "programs": a.programs, "resume_n": len(resume),
                      "ttft_mean": (sum(ttfts)/len(ttfts) if ttfts else None),
                      "cached_mean": (sum(cached)/len(cached) if cached else None)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

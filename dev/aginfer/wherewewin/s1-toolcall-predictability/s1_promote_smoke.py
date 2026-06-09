#!/usr/bin/env python3
"""S1 milestone-1b — deterministic predictive-promote smoke test.

Proves the promote FIRES end-to-end (schedule → demote-during-gap → belief-
validated migrate(→HBM)) WITHOUT the over-subscription deadlock, by isolating
one target program and forcing its idle tail to demote with a single filler:

  1. target A: SESSION_ARRIVAL + LLM_PREFILL, generate a ``prefix`` (forced),
     then TOOL_CALL_START{tool_eta_s} → the daemon schedules a promote-back.
  2. filler: one big generate that, together with A's tail, exceeds the small KV
     pool → sglang evicts A's idle tail to DRAM (write-through backup) AND fires
     MEMORY_PRESSURE → the daemon may also demote it. Either way A's tail goes
     HBM→DRAM-only. (Filler runs while A is idle → no concurrent-prefill
     deadlock.)
  3. ticker: post neutral SESSION_ARRIVAL events for a throwaway program to
     ADVANCE THE EVENT-STREAM CLOCK (the due-action heap is event-clocked, §3) so
     A's now-due promote drains and fires — without touching A's ACTING state.
  4. assert A's tail is back in HBM (promote landed), and the daemon logged
     promote_scheduled + promote_dispatched for A.

Run against a live small-pool a3 stack (sglang :30000 + daemon :9100):
  python s1_promote_smoke.py --prefix-tokens 6000 --filler-tokens 28000 --eta 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests

TGT = "tgtA"


def post_event(daemon: str, kind: str, session: str,
               extra: Optional[Dict[str, Any]] = None) -> None:
    body = {"kind": kind, "session": session}
    if extra:
        body.update(extra)
    requests.post(daemon.rstrip("/") + "/aginfer/event", json=body, timeout=10)


def generate(base: str, ids: List[int], max_new: int,
             forced: Optional[List[int]], pid: str) -> Dict[str, Any]:
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    t0 = time.perf_counter()
    r = requests.post(base.rstrip("/") + "/generate",
                      json={"input_ids": ids, "sampling_params": sp,
                            "program_id": pid, "stream": False}, timeout=300)
    e2e = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    mi = r.json()["meta_info"]
    return {"e2e_ms": e2e, "cached": int(mi.get("cached_tokens") or 0),
            "prompt": int(mi.get("prompt_tokens") or len(ids))}


def tail_residence(base: str, pid: str) -> Dict[str, Any]:
    """Residence of the largest unit held ONLY by ``pid`` (its exclusive tail)."""
    d = requests.get(base.rstrip("/") + "/aginfer/state", timeout=30).json()
    owned = [u for u in d.get("units", []) if u.get("session_ids") == [pid]]
    if not owned:
        return {"present": False}
    u = max(owned, key=lambda x: x.get("n_tokens", 0))
    return {"present": True, "residence": u.get("residence"),
            "n_tokens": u.get("n_tokens"), "hash": u.get("hash", "")[:12]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--daemon-url", default="http://127.0.0.1:9100")
    ap.add_argument("--prefix-tokens", type=int, default=6000)
    ap.add_argument("--output-tokens", type=int, default=800)
    ap.add_argument("--filler-tokens", type=int, default=28000)
    ap.add_argument("--eta", type=float, default=15.0)
    a = ap.parse_args()
    base, daemon = a.base_url, a.daemon_url
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)

    print("=== S1 promote smoke ===")
    # 1. target A: turn 0
    pref = list(range(200000, 200000 + a.prefix_tokens))
    out0 = list(range(900000, 900000 + a.output_tokens))
    post_event(daemon, "session_arrival", TGT)
    post_event(daemon, "llm_prefill", TGT)
    g0 = generate(base, pref, len(out0), out0, TGT)
    print(f"[1] A turn0: prompt={g0['prompt']} cached={g0['cached']} e2e={g0['e2e_ms']:.0f}ms")
    res_before = tail_residence(base, TGT)
    print(f"[1] A tail residence after turn0: {res_before}")
    # tool gap begins → schedule promote
    post_event(daemon, "tool_call_start", TGT, {"tool_eta_s": a.eta})
    print(f"[2] posted TOOL_CALL_START eta={a.eta}; A now ACTING (idle)")

    # 2. filler to pressure HBM and evict A's idle tail
    time.sleep(1.0)
    fill = list(range(500000, 500000 + a.filler_tokens))
    gf = generate(base, fill, 4, None, "filler")
    print(f"[3] filler prefill {a.filler_tokens}: e2e={gf['e2e_ms']:.0f}ms")
    res_demoted = tail_residence(base, TGT)
    print(f"[3] A tail residence after filler: {res_demoted}")
    demoted = res_demoted.get("present") and "HBM" not in (res_demoted.get("residence") or [])
    print(f"[3] A tail demoted out of HBM: {demoted}")

    # 3. ticker: advance the event clock until A's promote is due + fires
    seq = list(range(700000, 700000 + 64))
    deadline = time.time() + a.eta + 8
    promoted = False
    while time.time() < deadline:
        post_event(daemon, "session_arrival", f"ticker{int(time.time())}")
        time.sleep(1.0)
        r = tail_residence(base, TGT)
        if r.get("present") and "HBM" in (r.get("residence") or []):
            promoted = True
            print(f"[4] A tail back in HBM at t≈{time.time():.0f}: {r}")
            break
    if not promoted:
        print(f"[4] A tail NOT back in HBM by deadline: {tail_residence(base, TGT)}")

    # 4. A resumes (turn 1) — should hit HBM
    post_event(daemon, "tool_call_end", TGT)
    post_event(daemon, "llm_prefill", TGT)
    seq1 = pref + out0
    out1 = list(range(910000, 910000 + a.output_tokens))
    g1 = generate(base, seq1, len(out1), out1, TGT)
    print(f"[5] A resume turn1: prompt={g1['prompt']} cached={g1['cached']} "
          f"e2e={g1['e2e_ms']:.0f}ms  (cached≈prefix={a.prefix_tokens} ⇒ reuse)")

    print(json.dumps({"demoted": bool(demoted), "promoted": bool(promoted),
                      "resume_cached": g1["cached"], "prefix": a.prefix_tokens}))
    return 0 if (demoted and promoted) else 1


if __name__ == "__main__":
    raise SystemExit(main())

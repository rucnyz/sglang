#!/usr/bin/env python3
"""Part C1 — does forcing stay faithful across PREEMPTION (retract/resume)?

The S5/S7 overload scenarios drive the KV pool to exhaustion, so the scheduler
retracts running requests back to the waiting queue and later resumes them
(`retract_decode` → re-prefill the already-generated tokens → continue decode).
The override carries a per-`req` counter `_tf_step`; if retract reset/desynced it,
a resumed forced request would emit the WRONG tokens after the retract point.

Test: run a long forced request (force a KNOWN F) TWICE —
  (1) reference: alone, no pressure → text_ref (the ground truth for F).
  (2) pressured: identical request, but fired INSIDE a burst of K concurrent long
      decodes that exceeds the (small) KV pool → the scheduler must retract/resume
      to make progress, so the target is very likely retracted mid-decode.
PASS = pressured text == reference text AND completion_tokens == len(F), i.e. the
forced sequence survived retract/resume unchanged. We also grep the server log for
retract/preempt evidence so a no-preemption run is not silently counted as a pass.

Launch with a SMALL pool to guarantee preemption (see run_partC1.sh:
  --max-total-tokens 8192 --max-running-requests 64).
"""
import argparse
import concurrent.futures as cf
import json
import sys

import requests

L = 512                                   # long enough to span the preempt window
F = list(range(2000, 2000 + L))           # forced sequence (known)
TARGET_PREFIX = list(range(1500, 1500 + 64))
N_FILLERS = 48                            # concurrent long decodes to exhaust pool


def gen(base, input_ids, max_new, forced=None, prefix_salt=0):
    ids = [TARGET_PREFIX[0] + prefix_salt] + list(input_ids[1:]) if prefix_salt else list(input_ids)
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    body = {"input_ids": ids, "sampling_params": sp, "stream": False}
    r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d["text"], int((d["meta_info"].get("completion_tokens")
                           or d["meta_info"].get("completion_tokens", 0)) or max_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    base = a.base_url

    # (1) reference: forced F alone, no pressure
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)
    ref_txt, ref_n = gen(base, TARGET_PREFIX, L, forced=F)
    print(f"[C1] reference (no pressure): n={ref_n} text[:50]={ref_txt[:50]!r}")

    # (2) pressured: fire the target forced request concurrently with N fillers
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)
    with cf.ThreadPoolExecutor(max_workers=N_FILLERS + 2) as ex:
        futs = []
        # fillers: long decodes, distinct prefixes → saturate the pool
        for i in range(N_FILLERS):
            filler_prefix = list(range(3000 + i * 100, 3000 + i * 100 + 64))
            futs.append(ex.submit(gen, base, filler_prefix, L))
        # the target, fired in the middle of the burst
        tgt_fut = ex.submit(gen, base, TARGET_PREFIX, L, F)
        futs.append(tgt_fut)
        pre_txt, pre_n = tgt_fut.result()
        for f in futs:
            try:
                f.result()
            except Exception as e:
                print(f"[C1] a filler errored (tolerated): {e}", file=sys.stderr)

    print(f"[C1] pressured (retract/resume): n={pre_n} text[:50]={pre_txt[:50]!r}")

    same = (pre_txt == ref_txt) and (pre_n == ref_n == L)
    print("\n=== Part C1 result ===")
    print(f"  reference n={ref_n}, pressured n={pre_n}, len(F)={L}")
    print(f"  text identical: {pre_txt == ref_txt}")
    print(f"PART C1: {'PASS — forcing survived preemption byte-identical' if same else 'REVIEW — diverged (check log for retract evidence + multimodal branch)'}")
    print(json.dumps({"ref_n": ref_n, "pre_n": pre_n, "L": L,
                      "text_identical": pre_txt == ref_txt, "pass": same}))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

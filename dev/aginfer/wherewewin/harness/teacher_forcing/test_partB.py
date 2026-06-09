#!/usr/bin/env python3
"""Part B — does teacher-forcing reproduce the multi-turn KV continuation?

The point of TF replay: in a real agent, turn N's decoded output O becomes part
of turn N+1's input, and [context+O] is ALREADY cached (prefill of context +
decode of O), so turn N+1 hits it and only prefills the new delta. Length-only
replay breaks this: turn N's forced content O' != the real O embedded in turn
N+1's captured input, so [O] misses the cached [O'] and gets re-prefilled.

We measure turn-2 RE-PREFILLED tokens (= prompt_len - cached_tokens) across three
arms, in token-id space (exact, no detokenize round-trip), flushing the radix
cache between arms for independence:

  REAL       turn1 greedy -> O1 ;            turn2 [P + O1 + delta]   (the real continuation)
  TF         turn1 forced to O1 ;            turn2 [P + O1 + delta]   (should == REAL)
  LENGTH-ONLY turn1 forced to O1' (!=O1) ;   turn2 [P + O1 + delta]   (negative control)

PASS: TF re-prefill ~= REAL re-prefill (both ~= len(delta)); LENGTH-ONLY
re-prefill ~= len(O1) + len(delta) (the artifact TF removes).

Run: sglang up with --enable-cache-report (override needs no flag). See run_partB.sh.
"""
import argparse
import json
import sys

import requests

# arbitrary but valid, non-special token ids (content is irrelevant to the cache test)
P_IDS = list(range(1000, 1000 + 200))      # "context" prefix
DELTA_IDS = list(range(5000, 5000 + 8))     # "tool result / next-turn delta"
L = 64                                       # turn-1 output length


def flush(base):
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)


def gen_ids(base, input_ids, max_new, forced=None):
    """/generate over input_ids. Returns (output_ids, cached_tokens, prompt_len)."""
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    body = {"input_ids": input_ids, "sampling_params": sp,
            "return_logprob": True, "stream": False}
    r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=600)
    r.raise_for_status()
    mi = r.json()["meta_info"]
    otl = mi.get("output_token_logprobs") or []
    out_ids = [int(t[1]) for t in otl]
    cached = mi.get("cached_tokens")
    cached = int(cached) if cached is not None else None
    prompt_len = mi.get("prompt_tokens") or len(input_ids)
    return out_ids, cached, int(prompt_len)


def turn2_reprefill(base, o1_for_input):
    """Run turn 2 = [P + o1 + delta], 1 token; return (reprefilled, cached, prompt_len)."""
    inp = P_IDS + list(o1_for_input) + DELTA_IDS
    _, cached, plen = gen_ids(base, inp, 1)
    if cached is None:
        print("ERROR: server did not return cached_tokens — launch with "
              "--enable-cache-report", file=sys.stderr)
        sys.exit(2)
    return plen - cached, cached, plen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    base = a.base_url

    # ---- REAL: turn1 greedy -> capture O1_real, then turn2 ----
    flush(base)
    o1_real, _, _ = gen_ids(base, P_IDS, L)
    if len(o1_real) != L:
        print(f"note: turn1 produced {len(o1_real)} tokens (expected {L})")
    rp_real, c_real, pl = turn2_reprefill(base, o1_real)

    # a DIFFERENT turn-1 output of the same length, using only KNOWN-VALID token
    # ids (reuse the real outputs, reversed) so forcing can't hit out-of-vocab
    # ids — same length, different content, all guaranteed in-vocab.
    o1_diff = list(reversed(o1_real))
    if o1_diff == o1_real:  # palindrome: rotate by one
        o1_diff = o1_real[1:] + o1_real[:1]

    # ---- TF: turn1 FORCED to O1_real, then turn2 with the same real O1 ----
    flush(base)
    o1_tf, _, _ = gen_ids(base, P_IDS, len(o1_real), forced=o1_real)
    tf_forced_ok = (o1_tf == o1_real)
    rp_tf, c_tf, _ = turn2_reprefill(base, o1_real)

    # ---- LENGTH-ONLY: turn1 FORCED to O1_diff, then turn2 with the REAL O1 ----
    flush(base)
    o1_lo, _, _ = gen_ids(base, P_IDS, len(o1_real), forced=o1_diff)
    lo_forced_ok = (o1_lo == o1_diff)
    rp_lo, c_lo, _ = turn2_reprefill(base, o1_real)  # turn2 carries the REAL o1

    delta = len(DELTA_IDS)
    print("\n=== Part B result ===")
    print(f"prompt_len (turn2) = {pl}  (P={len(P_IDS)} + O1={len(o1_real)} + delta={delta})")
    print(f"forcing worked: TF turn1==O1_real {tf_forced_ok}; "
          f"LEN-ONLY turn1==O1_diff {lo_forced_ok}")
    print(f"turn-2 re-prefilled tokens:")
    print(f"  REAL        = {rp_real:4d}  (cached {c_real})   ~ expect delta≈{delta}")
    print(f"  TF          = {rp_tf:4d}  (cached {c_tf})   ~ expect == REAL")
    print(f"  LENGTH-ONLY = {rp_lo:4d}  (cached {c_lo})   ~ expect ≈ len(O1)+delta = {len(o1_real)+delta}")
    # NOTE: `*_forced_ok` compares the model's argmax readback (output_token_
    # logprobs) to the forced sequence — but that readback reports the SAMPLED
    # token, not the committed/overridden one (the override runs after the
    # sampler), so it equals the forced seq only when forcing == natural argmax
    # (the TF arm). It is NOT a forcing check; the substantive evidence is the
    # turn-2 re-prefill (which reflects the actual KV the override produced).
    tf_matches_real = abs(rp_tf - rp_real) <= 2
    lo_shows_artifact = (rp_lo - rp_real) >= 0.5 * len(o1_real)
    verdict = tf_matches_real and lo_shows_artifact
    print(f"\nTF reproduces real continuation (rp_tf==rp_real): {tf_matches_real}")
    print(f"length-only shows the re-prefill artifact (rp_lo >> rp_real): {lo_shows_artifact}")
    print(f"PART B: {'PASS' if verdict else 'REVIEW'}")
    print(json.dumps({"rp_real": rp_real, "rp_tf": rp_tf, "rp_lenonly": rp_lo,
                      "o1_len": len(o1_real), "delta": delta,
                      "tf_matches_real": tf_matches_real,
                      "lenonly_artifact": lo_shows_artifact, "pass": verdict}))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

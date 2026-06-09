#!/usr/bin/env python3
"""Part A-state (C5) — does the override write the EXACT forced tokens into the
KV/radix cache, at TOKEN resolution? Closes two gaps the audit found:

  A1: Part A's "forced==O*" is circular (forcing the model's own argmax reads
      back as O* even if forcing did nothing) — so it never proved the override
      fired.
  A2: Part B's re-prefill match is page-aligned (256), so it can't see a
      sub-page (<=255-token) KV corruption.

Method (re-feed): force a KNOWN sequence F (NOT the natural argmax), caching
[P+F]; then re-feed the EXACT [P+F] token ids and read cached_tokens.
  - If the override truly wrote F, the re-fed F matches the cache -> cached ~ full
    [P+F]. (A natural-argmax cache would diverge at F[0] -> only P cached. So a
    high hit INDEPENDENTLY proves the override fired -> closes A1.)
  - Negative control: force F' = F with ONE token changed inside a full page;
    re-feed the ORIGINAL F -> the cache diverges at that token -> hit drops by a
    whole page. Detecting a single-token change at token resolution -> closes A2.

PASS: cached_full ~ full [P+F]  AND  cached_full - cached_corrupt >= one page
(the override wrote the exact tokens, and a 1-token error is detectable).

Run: sglang up with --enable-cache-report. See run_partAstate.sh.
"""
import argparse
import json
import sys

import requests

PAGE = 256
# F must span >=2 pages so its FIRST page lands in a CACHED (non-last) page —
# the radix cache does not return the final page of a sequence as a hit (it
# reserves it / excludes the last token), so a single-page F is invisible.
P = list(range(1000, 1000 + PAGE))           # 1 page, page-aligned
F = list(range(2000, 2000 + 2 * PAGE))       # forced seq, 2 pages (512)
CORRUPT_IDX = 50                              # F[50] -> global pos 306 -> page 2 of [P+F] (cached)


def gen(base, input_ids, max_new, forced=None):
    sp = {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True}
    if forced is not None:
        sp["custom_params"] = {"forced_output_ids": list(forced)}
    body = {"input_ids": input_ids, "sampling_params": sp, "stream": False}
    r = requests.post(base.rstrip("/") + "/generate", json=body, timeout=600)
    r.raise_for_status()
    mi = r.json()["meta_info"]
    return mi.get("cached_tokens"), int(mi.get("prompt_tokens") or len(input_ids))


def flush(base):
    requests.post(base.rstrip("/") + "/flush_cache", timeout=30)


def refeed_hit(base, force_seq):
    """Force `force_seq` after P (caching [P+force_seq]); then re-feed the EXACT
    [P+F] and return how many tokens hit the cache."""
    flush(base)
    gen(base, P, len(force_seq), forced=force_seq)          # cache [P + force_seq]
    cached, _ = gen(base, P + F, 1)                          # re-feed the ORIGINAL [P+F]
    if cached is None:
        print("ERROR: no cached_tokens — launch with --enable-cache-report",
              file=sys.stderr)
        sys.exit(2)
    return int(cached)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    base = a.base_url
    full_len = len(P) + len(F)  # 512

    cached_full = refeed_hit(base, F)                        # faithful: cache==F
    F_corrupt = list(F); F_corrupt[CORRUPT_IDX] = F[CORRUPT_IDX] + 1
    cached_corrupt = refeed_hit(base, F_corrupt)            # cache==F' (1 tok off)
    # extra: force NOTHING (natural argmax) then re-feed [P+F] — should hit only P
    flush(base); gen(base, P, len(F)); cached_nat, _ = gen(base, P + F, 1)

    print("\n=== Part A-state (C5) result ===")
    print(f"[P]={len(P)} [F]={len(F)} [P+F]={full_len}  page={PAGE}  corrupt at F[{CORRUPT_IDX}] (global pos {len(P)+CORRUPT_IDX})")
    print(f"  faithful  (force F, re-feed F):   cached {cached_full:4d}  (expect ~{full_len})")
    print(f"  corrupted (force F', re-feed F):  cached {cached_corrupt:4d}  (expect ~{len(P)} = P only)")
    print(f"  natural   (no force, re-feed F):  cached {int(cached_nat):4d}  (expect ~{len(P)} = P only)")

    override_fired = cached_full > len(P)                       # some of F is cached => override wrote F (not natural)
    token_level = (cached_full - cached_corrupt) >= PAGE        # 1-token change in a cached page drops a whole page
    verdict = override_fired and token_level
    print(f"\noverride actually wrote F (closes A1, faithful>>corrupt): {override_fired}")
    print(f"token-level fidelity — 1-token error detected (closes A2): {token_level}")
    print(f"PART A-state: {'PASS' if verdict else 'REVIEW'}")
    print(json.dumps({"cached_full": cached_full, "cached_corrupt": cached_corrupt,
                      "cached_natural": int(cached_nat), "full_len": full_len,
                      "page": PAGE, "override_fired": override_fired,
                      "token_level": token_level, "pass": verdict}))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

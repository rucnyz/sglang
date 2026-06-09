#!/usr/bin/env python3
"""Part C4 — does forced_output_ids reach the override via the OpenAI CHAT path?

The real replay sends OpenAI /v1/chat/completions (through the daemon proxy),
not /generate. This verifies the plumbing end-to-end: a chat request carrying
`custom_params.forced_output_ids` must (a) reach SamplingParams.custom_params and
(b) make the override fire — i.e. the output is the FORCED sequence, not the
natural greedy one, and exactly len(F) tokens.

We send custom_params two ways the OpenAI client may produce (top-level, and
nested under extra_body) and accept whichever the server honors.

--base-url is the endpoint under test: sglang directly (:30000) OR the daemon
proxy (:9100) — run it against both to cover C4 fully.
Run: server up (override in-code; no special flag). See run_partC4.sh.
"""
import argparse
import json
import sys

import requests

MESSAGES = [{"role": "user", "content": "Write one paragraph about caching."}]
F = list(range(2000, 2000 + 48))   # forced seq, valid non-special ids


def chat(base, *, forced=None, nest=False):
    body = {"model": "deepseek-ai/DeepSeek-V4-Flash", "messages": MESSAGES,
            "temperature": 0.0, "max_tokens": len(F), "stream": False}
    body["ignore_eos"] = True
    if forced is not None:
        cp = {"forced_output_ids": list(forced)}
        if nest:
            body["extra_body"] = {"custom_params": cp}
        else:
            body["custom_params"] = cp
    r = requests.post(base.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    # reasoning model: output lands in reasoning_content (content empty) — read both
    txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    n = (d.get("usage") or {}).get("completion_tokens")
    return txt, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    a = ap.parse_args()
    base = a.base_url

    nat_txt, nat_n = chat(base)
    print(f"[C4] natural: n={nat_n} text[:60]={nat_txt[:60]!r}")

    # try top-level custom_params, then nested extra_body
    forced_txt = forced_n = None
    used = None
    for nest in (False, True):
        try:
            t, n = chat(base, forced=F, nest=nest)
        except Exception as e:
            print(f"[C4] forced (nest={nest}) errored: {e}")
            continue
        # forcing took effect iff the output changed AND length == len(F)
        if t != nat_txt and (n == len(F) or abs((n or 0) - len(F)) <= 1):
            forced_txt, forced_n, used = t, n, ("extra_body" if nest else "top-level")
            break
        print(f"[C4] forced (nest={nest}): n={n} changed={t != nat_txt} — not honored")
    print("\n=== Part C4 result ===")
    if used is None:
        print("forcing did NOT take effect via the chat API on EITHER custom_params "
              "placement → the OpenAI path does not plumb forced_output_ids to the "
              "override. FAIL — must fix the chat→SamplingParams.custom_params hop.")
        print(json.dumps({"endpoint": base, "honored": False}))
        return 1
    print(f"forcing honored via chat `{used}` custom_params: n={forced_n}=={len(F)}, "
          f"output changed from natural → custom_params.forced_output_ids reaches the "
          f"override through the OpenAI chat path. PASS")
    print(f"  natural text[:50]={nat_txt[:50]!r}")
    print(f"  forced  text[:50]={forced_txt[:50]!r}")
    print(json.dumps({"endpoint": base, "honored": True, "placement": used,
                      "forced_n": forced_n, "F_len": len(F)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

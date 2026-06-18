"""Token-exact forcing validator (task #334 e2e gate) — dump mode.

Sends teacher-forced /generate requests to a RUNNING sglang server and dumps,
per request, the server's emitted `text` and `completion_tokens`. The harness
then compares the dump from an overlap-OFF boot (the known-correct CPU
`_maybe_force_output_token` hook) against an overlap-ON boot (the new GPU
override under test).

We compare `text` (the detokenization of the EMITTED, i.e. forced, output ids),
not `output_token_logprobs` — those carry the SAMPLED token recorded BEFORE the
force override, so they would mislead. `text` reflects the forced emit exactly
and is robust to greedy-argmax numerical noise across boots. If the overlap
override is broken (feeds the sampled token forward instead of the forced one),
`text` diverges from the overlap-off baseline.

Usage: validate_client.py <url> <label> <outfile.json>
"""
import sys
import json
import urllib.request

URL = sys.argv[1]
LABEL = sys.argv[2]
OUTFILE = sys.argv[3]

# Distinct forced sequences / lengths; arbitrary valid vocab ids (forcing
# overrides sampling, so the logits do not matter for the emitted token).
REQS = [
    {"input_ids": list(range(10, 26)), "forced": list(range(1000, 1040))},
    {"input_ids": list(range(40, 52)), "forced": list(range(2000, 2071))},
    {"input_ids": list(range(60, 88)), "forced": list(range(3000, 3025))},
    {"input_ids": list(range(90, 99)), "forced": list(range(4000, 4096))},
]


def _post(input_ids, forced):
    body = json.dumps({
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": len(forced),
            "ignore_eos": True,
            "custom_params": {"forced_output_ids": forced},
        },
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    dump = []
    for r in REQS:
        resp = _post(r["input_ids"], r["forced"])
        mi = resp.get("meta_info") or {}
        dump.append({
            "forced_len": len(r["forced"]),
            "n_out": mi.get("completion_tokens"),
            "text": resp.get("text"),
        })
        print(f"  [{LABEL}] req: n_out={mi.get('completion_tokens')} "
              f"forced_len={len(r['forced'])} text_len={len(resp.get('text') or '')}")
    json.dump(dump, open(OUTFILE, "w"))
    print(f"  [{LABEL}] dumped -> {OUTFILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

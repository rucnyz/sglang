#!/usr/bin/env python3
"""Convert pd_exp generated JSONL ({prompt, input_len, output_len}) into the
ShareGPT-style conversations format bench_serving's `--dataset-name custom`
expects: {"conversations": [{"value": prompt}, {"value": completion_dummy}]}

Usage:
    python _convert_jsonl_to_sharegpt.py <in.jsonl> <out.jsonl>
"""

import json
import sys

if len(sys.argv) != 3:
    print("usage: convert_jsonl_to_sharegpt.py <in.jsonl> <out.jsonl>", file=sys.stderr)
    sys.exit(1)

in_path, out_path = sys.argv[1], sys.argv[2]
n_in, n_out = 0, 0
with open(in_path) as f, open(out_path, "w") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_in += 1
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = d.get("prompt") or d.get("text") or ""
        if not prompt or len(prompt) < 16:
            continue
        # Make completion long enough to exceed `output_len < 2` filter and
        # provide a plausible target output len signal (bench will tokenize
        # both and use len of completion as output_len unless --sharegpt-output-len is set).
        out_len = int(d.get("output_len") or 16)
        completion = "x " * max(8, out_len)
        g.write(json.dumps({
            "conversations": [
                {"value": prompt},
                {"value": completion},
            ]
        }) + "\n")
        n_out += 1
print(f"converted: {in_path} -> {out_path}  in={n_in} out={n_out}")

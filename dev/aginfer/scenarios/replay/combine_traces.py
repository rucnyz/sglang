"""Combine per-program trace jsonls (traces/cc/*.jsonl) into ONE replay trace with
staggered program start times (arrival), for replay_driver.py. Each per-program
file has t relative to its own start; this staggers them so N programs run
concurrently (the pressure regime). Programs share the (identical) system block ->
cross-program KV sharing is preserved.

Usage:
  python combine_traces.py --in-dir traces/cc --out traces/cc_combined.jsonl \
     --n 30 --stagger 4.0  [--shuffle-seed 0]
"""
import json, os, glob, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30, help="how many programs to include")
    ap.add_argument("--stagger", type=float, default=4.0, help="seconds between program starts")
    ap.add_argument("--order", choices=["size", "name"], default="size",
                    help="size = largest programs first (more pressure)")
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.in_dir, "*.jsonl"))
    if a.order == "size":
        files.sort(key=lambda p: -os.path.getsize(p))
    else:
        files.sort()
    files = files[:a.n]

    records = []
    for i, f in enumerate(files):
        off = i * a.stagger
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["t"] = round(off + float(r.get("t", 0.0)), 4)
            records.append(r)
    records.sort(key=lambda r: r["t"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    pids = len({r["program_id"] for r in records})
    span = records[-1]["t"] - records[0]["t"] if records else 0
    mb = os.path.getsize(a.out) / 1024 / 1024
    print(f"wrote {a.out}: {pids} programs, {len(records)} requests, span={span:.0f}s, {mb:.1f}MB")


if __name__ == "__main__":
    main()

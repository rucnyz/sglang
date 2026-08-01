#!/usr/bin/env python
"""Aggregate all RQ1 run JSONs into main-table cells (out-TPS + P99 TTFT) per
model/baseline. Run after run_rq1_campaign.sh completes. Missing dirs are
reported as '(missing)' so partial progress is visible."""
import json
import glob
import statistics
import os

RQ1 = "/scratch/yuzhou/projects/sglang/reproduce/RQ1"


def load(pattern):
    fs = sorted(glob.glob(pattern))
    out = []
    for f in fs:
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def stat(ds, key_path):
    vals = []
    for d in ds:
        v = d
        for k in key_path:
            v = v.get(k, {}) if isinstance(v, dict) else 0
        if isinstance(v, (int, float)) and v:
            vals.append(v)
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, [round(x, 1) for x in vals]


def cell(label, ds):
    if not ds:
        print(f"  {label:16s}: (missing)")
        return
    tps = stat(ds, ["throughput_tok_s"])
    p99 = stat(ds, ["ttft_ms", "p99"])
    err = sum(d.get("n_error", 0) for d in ds)
    lm = stat(ds, ["len_match_rate"])
    tps_s = f"{tps[0]:.1f}±{tps[1]:.1f} {tps[2]}" if tps else "?"
    p99_s = f"{p99[0]:.0f}" if p99 else "?"
    lm_s = f" len_match={lm[0]:.2f}" if lm else ""
    print(f"  {label:16s}: tps={tps_s}  P99_TTFT={p99_s}ms  err={err}{lm_s}")


def model_block(name, official_tag, camp_prefix):
    print(f"\n=== {name} (Claude Code replay = Case1 t6_v2@64) ===")
    cell("default(base)", load(f"{RQ1}/case1/runs/{official_tag}/base_r*.json"))
    cell("sys",           load(f"{RQ1}/case1/runs/{official_tag}/sys_r*.json"))
    cell("vLLM",          load(f"{RQ1}/campaign/{camp_prefix}_vllm/vllm_r*.json"))
    cell("static R0.8",   load(f"{RQ1}/campaign/{camp_prefix}_static_r0.8/base_r*.json"))
    # static-best = max(default, static R0.8); default already measured above.
    print("  (static-best = better of default / static R0.8)")
    # Case2/Case3 for the detailed/ablation rows:
    for c in ("case2", "case3"):
        b = load(f"{RQ1}/{c}/runs/{official_tag}/base_r*.json")
        s = load(f"{RQ1}/{c}/runs/{official_tag}/sys_r*.json")
        if b or s:
            print(f"  -- {c} --")
            cell(f"{c} base", b)
            cell(f"{c} sys", s)


def kimi_block():
    """Kimi-Linear-48B (branch HiMA-latest, 2026-08-01 campaign): different
    layout — figures/data/kimi48b/<wl>/{base,sys,static_r*,vllm}/*.json."""
    K = "/data/yuzhou/projects/hybrid-inference/figures/data/kimi48b"
    for wl, label in (("lh", "long-horizon"), ("swarm", "agent swarm"),
                      ("shift", "shifting")):
        print(f"\n=== Kimi-Linear-48B-A3B — {label} ===")
        cell("default(base)", load(f"{K}/{wl}/base/base_r*.json"))
        cell("sys",           load(f"{K}/{wl}/sys/sys_r*.json"))
        cell("vLLM",          load(f"{K}/{wl}/vllm/vllm_r*.json"))
        for r in ("0.7", "0.8", "0.95"):
            cell(f"static R{r}", load(f"{K}/{wl}/static_r{r}/base_r*.json"))
        print("  (static-best = best of default mean / clean sweep singles; "
              "shift R0.95 has 1 pool-truncation err -> ineligible)")


model_block("Qwen3.5-9B", "official", "9b")
model_block("Qwen3.5-35B-A3B", "official_35b", "35b")
kimi_block()

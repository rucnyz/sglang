"""pristine_saturation — engine-level vs hand-verified constants.

For each (model, kv_dtype, ssm_dtype) cell that the driver
(`run_pristine.sh`) booted, parses the sglang server.log for pool
sizes and asserts:

  (a) actual mamba pool bytes / slot count == HAND_VERIFIED_MAMBA
      (from sibling folder ../dtype_unit_sizes/)
  (b) actual KV pool bytes / token count == HAND_VERIFIED_KV
      (from sibling folder ../dtype_unit_sizes/)

(a) catches: sglang booted with a different mamba dtype than we
requested, or our env var failed to take effect, or sglang's allocator
inflates the per-slot bytes (padding, alignment).

(b) catches: kv_cache_dtype wasn't honored, or model arch read wrong.

This is the ENGINE-LEVEL verification of the spec-level test in
`../dtype_unit_sizes/`. That test proved sglang's API matches
hand-verified arithmetic; sglang could still in principle allocate
more memory than its API says it needs, and this script rules out
that gap.
"""
import json
import os
import re
import sys

# Import hand-verified constants from the spec-level test (sibling).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "dtype_unit_sizes",
    ),
)
from test_dtype_unit_sizes import HAND_VERIFIED_MAMBA, HAND_VERIFIED_KV

GB = 1024 ** 3


def parse_boot_log(path):
    """Extract pool sizes from sglang server.log.

    The mamba regex is anchored to whitespace+end-of-line so that the
    speculative-decode log format (which appends two extra
    intermediate_*_cache fields,
    `sglang/srt/mem_cache/memory_pool.py:599-639`) does NOT match the
    non-speculative regex. If it did, we'd silently undercount
    allocator bytes by ~10-30%.
    """
    out = {}
    # Non-speculative format only — anchored to whitespace+end-of-line.
    pat_mamba = re.compile(
        r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+),\s+"
        r"conv_state size: ([0-9.]+)GB, ssm_state size: ([0-9.]+)GB\s*$"
    )
    pat_mamba_spec = re.compile(
        r"Mamba Cache is allocated\..*intermediate_ssm_state_cache"
    )
    pat_kv = re.compile(
        r"KV Cache is allocated\. #tokens: (\d+),\s+"
        r"K size: ([0-9.]+) GB, V size: ([0-9.]+) GB"
    )
    pat_page = re.compile(r"page_size=(\d+)")
    with open(path) as f:
        for line in f:
            if pat_mamba_spec.search(line):
                # Sentinel: speculative log format is currently unsupported.
                # Refuse to silently undercount.
                out["spec_format_detected"] = True
            if m := pat_mamba.search(line.rstrip("\n")):
                out["mamba_slots"] = int(m.group(1))
                out["mamba_conv_GB"] = float(m.group(2))
                out["mamba_ssm_GB"]  = float(m.group(3))
                out["mamba_total_bytes"] = (out["mamba_conv_GB"] + out["mamba_ssm_GB"]) * GB
            if m := pat_kv.search(line):
                out["kv_tokens"] = int(m.group(1))
                out["kv_K_GB"] = float(m.group(2))
                out["kv_V_GB"] = float(m.group(3))
                out["kv_total_bytes"] = (out["kv_K_GB"] + out["kv_V_GB"]) * GB
            # MHATokenToKVPool log: "...size=N, page_size=K, layer_num=..."
            if "page_size=" in line and "MHATokenToKVPool" in line:
                if m := pat_page.search(line):
                    out["page_size"] = int(m.group(1))
    return out


def validate_cell(label, log_path, model, tp, kv_dtype, ssm_dtype,
                   tol_pct=0.1):
    """Compare sglang's actual allocator output to HAND_VERIFIED constants.

    Slot accounting: the log reports `max_mamba_cache_size: N` and
    `#tokens: M`, but sglang's allocator allocates one extra padded slot
    (the "dummy output" slot 0) for both pools:
      - mamba: torch.zeros(size=(num_layers, N + 1, ...), ...)
        (sglang/srt/mem_cache/memory_pool.py: conv_state at :415, temporal_state at :584/:592)
      - KV:    torch.zeros((M + page_size, ...), ...)
        (sglang/srt/mem_cache/memory_pool.py:1688, 1696)
    So actual per-slot bytes = total_bytes / (reported + page_size).
    Without this correction the residual is +1/N (e.g. +0.24% on 35B
    with 416 slots) — that's the off-by-one, not allocator padding.

    Tolerance: 0.1%. After the +page_size correction, the *only*
    residual is sglang's "%.2f GB" log-print precision (±5 MB per
    pool component → ~0.02% on a 50 MB/slot pool with 1000 slots,
    larger when fewer slots). 0.1% is a 2-5× cushion above that
    floor. Tightening further requires the allocator to expose a
    byte-exact API (out of scope here).

    Fail-closed: if HAND_VERIFIED_{MAMBA,KV} has no entry for this
    cell, return FAIL — silent skip would let coverage rot.
    """
    info = parse_boot_log(log_path)
    if info.get("spec_format_detected"):
        return {"label": label, "ok": False,
                "reason": "speculative-decode log format detected — "
                          "validator only supports non-speculative; add "
                          "intermediate_*_cache parsing if you enabled spec"}
    if "mamba_slots" not in info or "kv_tokens" not in info:
        return {"label": label, "ok": False,
                "reason": f"log missing mamba/KV size lines (boot failed?)"}
    if "page_size" not in info:
        return {"label": label, "ok": False,
                "reason": "could not parse page_size from MHATokenToKVPool log"}

    page_size = info["page_size"]

    # Mamba allocator allocates (size + 1) slots, not `size` (sglang's
    # padded slot 0). Log reports `max_mamba_cache_size: size`.
    measured_mamba_per_req = info["mamba_total_bytes"] / (info["mamba_slots"] + 1)
    expected_mamba = HAND_VERIFIED_MAMBA.get((model, tp, ssm_dtype))
    if expected_mamba is None:
        return {"label": label, "ok": False,
                "reason": f"no HAND_VERIFIED_MAMBA entry for "
                          f"({model}, tp={tp}, {ssm_dtype}) — add it to "
                          f"0_page_state_machine/dtype_unit_sizes/"}
    delta_mamba_pct = (measured_mamba_per_req - expected_mamba) / expected_mamba * 100

    # KV allocator allocates (#tokens + page_size) slots. Log reports `#tokens`.
    measured_kv_per_token = info["kv_total_bytes"] / (info["kv_tokens"] + page_size)
    kv_resolved = "bfloat16" if kv_dtype in ("auto", "bf16", "bfloat16") else kv_dtype
    expected_kv = HAND_VERIFIED_KV.get((model, tp, kv_resolved))
    if expected_kv is None:
        return {"label": label, "ok": False,
                "reason": f"no HAND_VERIFIED_KV entry for "
                          f"({model}, tp={tp}, {kv_resolved}) — add it to "
                          f"0_page_state_machine/dtype_unit_sizes/"}
    delta_kv_pct = (measured_kv_per_token - expected_kv) / expected_kv * 100

    return {
        "label": label, "model": model, "tp": tp,
        "kv_dtype": kv_dtype, "ssm_dtype": ssm_dtype,
        "mamba_slots": info["mamba_slots"],
        "kv_tokens": info["kv_tokens"],
        "mamba_per_req_measured": measured_mamba_per_req,
        "mamba_per_req_expected": expected_mamba,
        "delta_mamba_pct": delta_mamba_pct,
        "kv_per_token_measured": measured_kv_per_token,
        "kv_per_token_expected": expected_kv,
        "delta_kv_pct": delta_kv_pct,
        "ok": abs(delta_mamba_pct) < tol_pct and abs(delta_kv_pct) < tol_pct,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="dir containing <label>.server.log + <label>.cell.json")
    args = ap.parse_args()

    # Find all cells
    out_dir = args.out_dir
    cells = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".cell.json"):
            cells.append(f[:-len(".cell.json")])

    if not cells:
        print(f"No cell descriptors found in {out_dir}")
        return 1

    print(f"\n{'='*92}")
    print(f"pristine_saturation — sglang actual pool sizes vs HAND_VERIFIED constants")
    print(f"{'='*92}")
    print(f"\n{'cell':22s} {'model':20s} {'tp':>2s} {'kv':>9s} {'ssm':>9s} "
          f"{'mamba Δ%':>10s} {'kv Δ%':>8s}")
    print("-" * 92)

    all_ok = True
    for label in cells:
        cell_desc = json.load(open(f"{out_dir}/{label}.cell.json"))
        log_path = f"{out_dir}/{label}.server.log"
        result = validate_cell(label, log_path, **cell_desc)

        if result.get("reason"):
            print(f"  FAIL {label:18s}  {result['reason']}")
            all_ok = False
            continue

        mark = "✓" if result["ok"] else "✗"
        print(f"  {mark} {label:20s} {result['model']:20s} "
              f"{result['tp']:>2d} {result['kv_dtype']:>9s} {result['ssm_dtype']:>9s} "
              f"{result['delta_mamba_pct']:>9.3f}% {result['delta_kv_pct']:>7.3f}%")
        if not result["ok"]:
            all_ok = False

    print()
    print(f"pristine_saturation: {'ALL PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

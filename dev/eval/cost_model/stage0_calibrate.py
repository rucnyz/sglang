"""
Stage-0 deployment-time calibration: measure recovery-cost curves c_σ(L) for
the current (GPU, model) tuple and emit budgeter-consumable outputs.

What it does
------------
1. Detects the local GPU and (optionally takes) the model identifier.
2. Times the bare attention-prefill kernel and bare DeltaNet recurrent kernel
   on the model's real layer dims, sweeping L on a log grid.
3. Least-squares-fits the parametric forms:
       c_KV(L) = α_KV · L² + β_KV · L + γ_KV
       c_M(L)  = α_M  · L     + β_M
4. Solves for the crossover L* (where c_KV = c_M).
5. Writes a JSON record to a deterministic path under
   ~/.cache/sglang/cost_calibration/<gpu>__<model>.json
6. Prints `export SGLANG_CSIGMA_*=...` lines on stdout that the user can
   `eval $(...)` to pin the numbers as environment variables for the
   subsequent serving run.

The budgeter (task #61) reads either the JSON (preferred path) or the env
vars (override) at engine boot.

Usage
-----
    # Run on current GPU (default model = Qwen3.5-35B-A3B):
    .venv/bin/python dev/eval/cost_model/stage0_calibrate.py

    # Pin in shell, then launch server:
    eval "$(.venv/bin/python dev/eval/cost_model/stage0_calibrate.py --print-env)"
    .venv/bin/python -m sglang.launch_server ...

    # Force re-run even if cached output exists:
    .venv/bin/python dev/eval/cost_model/stage0_calibrate.py --force

Currently shipped models with built-in dim presets:
  - Qwen/Qwen3.5-35B-A3B  (10 attn + 30 linear DeltaNet, hidden 2048)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch


# ---------- Model presets ----------

@dataclass
class ModelDims:
    name: str
    n_layers_attn: int
    n_layers_linear: int
    hidden: int
    n_qheads: int
    n_kvheads: int
    head_dim: int
    lin_num_qheads: int
    lin_num_vheads: int
    lin_key_dim: int
    lin_value_dim: int
    dtype: str  # "bfloat16" / "float16"


MODELS = {
    "Qwen/Qwen3.5-35B-A3B": ModelDims(
        name="Qwen/Qwen3.5-35B-A3B",
        n_layers_attn=10,
        n_layers_linear=30,
        hidden=2048,
        n_qheads=16,
        n_kvheads=2,
        head_dim=256,
        lin_num_qheads=16,
        lin_num_vheads=32,
        lin_key_dim=128,
        lin_value_dim=128,
        dtype="bfloat16",
    ),
}


# ---------- Microbench ----------

def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_attn_prefill(L, dims, dtype, device):
    from sgl_kernel.flash_attn import flash_attn_varlen_func
    q = torch.randn(L, dims.n_qheads, dims.head_dim, dtype=dtype, device=device)
    k = torch.randn(L, dims.n_kvheads, dims.head_dim, dtype=dtype, device=device)
    v = torch.randn(L, dims.n_kvheads, dims.head_dim, dtype=dtype, device=device)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, L], dtype=torch.int32, device=device)

    def fn():
        flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=L, max_seqlen_k=L, causal=True,
        )
    return time_fn(fn)


def bench_gdn_prefill(L, dims, dtype, device):
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    q = torch.randn(1, L, dims.lin_num_qheads, dims.lin_key_dim, dtype=dtype, device=device)
    k = torch.randn(1, L, dims.lin_num_qheads, dims.lin_key_dim, dtype=dtype, device=device)
    v = torch.randn(1, L, dims.lin_num_vheads, dims.lin_value_dim, dtype=dtype, device=device)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, L, dims.lin_num_vheads, dtype=dtype, device=device)
    )
    beta = torch.sigmoid(torch.randn(1, L, dims.lin_num_vheads, dtype=dtype, device=device))
    pool = torch.randn(
        8, dims.lin_num_vheads, dims.lin_key_dim, dims.lin_value_dim,
        dtype=dtype, device=device,
    ) * 0.1
    pool = pool.transpose(-2, -1).contiguous().transpose(-2, -1)
    idx = torch.tensor([0], dtype=torch.int32, device=device)
    cu = torch.tensor([0, L], dtype=torch.long, device=device)

    def fn():
        chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=pool, initial_state_indices=idx,
            cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True,
        )
    return time_fn(fn)


# ---------- Fit ----------

def fit_curves(rows):
    import numpy as np
    L = np.array([r["L"] for r in rows], dtype=float)
    c_kv = np.array([r["tot_kv_ms"] for r in rows])
    c_m = np.array([r["tot_m_ms"] for r in rows])

    A_kv = np.stack([L * L, L, np.ones_like(L)], axis=1)
    coef_kv, *_ = np.linalg.lstsq(A_kv, c_kv, rcond=None)
    a_kv2, b_kv, g_kv = (float(x) for x in coef_kv)

    A_m = np.stack([L, np.ones_like(L)], axis=1)
    coef_m, *_ = np.linalg.lstsq(A_m, c_m, rcond=None)
    a_m, b_m = (float(x) for x in coef_m)

    a, b, c = a_kv2, b_kv - a_m, g_kv - b_m
    disc = b * b - 4 * a * c
    if disc >= 0 and a > 0:
        L_star = float((-b + disc ** 0.5) / (2 * a))
    else:
        L_star = float("inf")

    return dict(
        c_kv=dict(form="alpha*L**2 + beta*L + gamma",
                  alpha_ms_per_tok2=a_kv2,
                  beta_ms_per_tok=b_kv,
                  gamma_ms=g_kv),
        c_m=dict(form="alpha*L + beta",
                 alpha_ms_per_tok=a_m,
                 beta_ms=b_m),
        crossover_L_star=L_star,
    )


# ---------- IO ----------

def cache_root():
    return os.path.expanduser("~/.cache/sglang/cost_calibration")


def cache_path(gpu_tag, model_tag):
    safe_gpu = gpu_tag.replace(" ", "_").replace("/", "_")
    safe_model = model_tag.replace("/", "_")
    return os.path.join(cache_root(), f"{safe_gpu}__{safe_model}.json")


def stable_key(dims, dtype_str):
    payload = json.dumps(asdict(dims), sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B",
                    help=f"One of {list(MODELS.keys())}")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--print-env", action="store_true",
                    help="Suppress human-readable output; print only env exports.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run calibration even if cached output is fresh.")
    ap.add_argument("--out", default=None,
                    help="Override output JSON path.")
    args = ap.parse_args()

    if args.model not in MODELS:
        print(f"ERROR: unknown model '{args.model}'. Known: {list(MODELS.keys())}",
              file=sys.stderr)
        return 2
    dims = MODELS[args.model]
    dtype = getattr(torch, dims.dtype)
    device = "cuda"

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        return 2

    gpu_name = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()

    out_path = args.out or cache_path(gpu_name, args.model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Reuse cached result unless --force
    if not args.force and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if (cached.get("model") == args.model
                and cached.get("device") == gpu_name
                and cached.get("dims_key") == stable_key(dims, dims.dtype)):
                if not args.print_env:
                    print(f"# Stage-0 calibration cached at {out_path}", file=sys.stderr)
                emit(cached, out_path, args.print_env)
                return 0
        except Exception:
            pass  # fall through and re-run

    log = (lambda *a, **kw: None) if args.print_env else print

    log(f"Stage-0 calibration: {gpu_name} (SM {cap[0]}{cap[1]}) / {args.model}")
    log(f"  dtype={dims.dtype}  attn_layers={dims.n_layers_attn}  "
        f"linear_layers={dims.n_layers_linear}")
    log(f"  L sweep: {args.lengths}")
    log()

    rows = []
    t0 = time.time()
    for L in args.lengths:
        attn = bench_attn_prefill(L, dims, dtype, device)  # one attn-layer ms
        gdn = bench_gdn_prefill(L, dims, dtype, device)    # one linear-layer ms
        tot_kv = attn * dims.n_layers_attn
        tot_m = gdn * dims.n_layers_linear
        rows.append(dict(
            L=L,
            attn_ms=attn,
            gdn_ms=gdn,
            tot_kv_ms=tot_kv,
            tot_m_ms=tot_m,
            ratio=tot_m / tot_kv if tot_kv > 0 else float("inf"),
        ))
        log(f"  L={L:>6}  attn={attn:7.4f}ms  gdn={gdn:7.4f}ms  "
            f"c_KV={tot_kv:7.3f}ms  c_M={tot_m:7.3f}ms  ratio={rows[-1]['ratio']:.2f}x")
    elapsed = time.time() - t0

    fit = fit_curves(rows)

    record = dict(
        schema_version=1,
        model=args.model,
        device=gpu_name,
        sm=f"{cap[0]}{cap[1]}",
        dtype=dims.dtype,
        dims=asdict(dims),
        dims_key=stable_key(dims, dims.dtype),
        lengths=list(args.lengths),
        rows=rows,
        fit=fit,
        elapsed_s=elapsed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    log()
    log(f"wrote {out_path}  (elapsed {elapsed:.1f}s)")
    log(f"  c_KV(L) = {fit['c_kv']['alpha_ms_per_tok2']:.3e} L^2 "
        f"+ {fit['c_kv']['beta_ms_per_tok']:+.3e} L "
        f"+ {fit['c_kv']['gamma_ms']:.3e}   ms")
    log(f"  c_M(L)  = {fit['c_m']['alpha_ms_per_tok']:.3e} L "
        f"+ {fit['c_m']['beta_ms']:.3e}   ms")
    log(f"  L* = {fit['crossover_L_star']:.0f} tokens")

    emit(record, out_path, args.print_env)
    return 0


def emit(record, path, print_env):
    """Print export lines so the user can `eval $(stage0_calibrate.py --print-env)`."""
    fit = record["fit"]
    pairs = [
        ("SGLANG_CSIGMA_JSON", path),
        ("SGLANG_CSIGMA_KV_ALPHA", f"{fit['c_kv']['alpha_ms_per_tok2']:.6e}"),
        ("SGLANG_CSIGMA_KV_BETA", f"{fit['c_kv']['beta_ms_per_tok']:.6e}"),
        ("SGLANG_CSIGMA_KV_GAMMA", f"{fit['c_kv']['gamma_ms']:.6e}"),
        ("SGLANG_CSIGMA_M_ALPHA", f"{fit['c_m']['alpha_ms_per_tok']:.6e}"),
        ("SGLANG_CSIGMA_M_BETA", f"{fit['c_m']['beta_ms']:.6e}"),
        ("SGLANG_CSIGMA_LSTAR", f"{fit['crossover_L_star']:.1f}"),
        ("SGLANG_CSIGMA_MODEL", record["model"]),
        ("SGLANG_CSIGMA_DEVICE", record["device"]),
    ]
    if print_env:
        # only env exports on stdout (eval-friendly)
        for k, v in pairs:
            print(f'export {k}="{v}"')
    else:
        print()
        print("# Pin these env vars for the budgeter (eval-able):")
        for k, v in pairs:
            print(f'export {k}="{v}"')


if __name__ == "__main__":
    sys.exit(main())

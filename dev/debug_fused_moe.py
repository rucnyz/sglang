"""
Debug entry-point for SGLang's "traditional" Triton fused MoE path.

This script builds a *tiny* MoE problem on a single GPU and calls
`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_moe`,
so you can step through every stage that dev/a.md maps out:

    moe_align_block_size  ──►  invoke_fused_moe_kernel (gate/up)
                          ──►  silu_and_mul
                          ──►  invoke_fused_moe_kernel (down)
                          ──►  moe_sum_reduce

Suggested breakpoints (matched to dev/a.md):
  - fused_moe.py:401      `_fused_moe_kernel_sequence`   <- orchestrator (read top-to-bottom)
  - fused_moe.py:484      first invoke_fused_moe_kernel (GEMM #1: gate/up)
  - fused_moe.py:530      silu_and_mul
  - fused_moe.py:666      second invoke_fused_moe_kernel (GEMM #2: down)
  - fused_moe.py:711      moe_sum_reduce (combine stage)
  - moe_align_block_size.py:1   padding wrapper
  - fused_moe_triton_kernels.py:324  `fused_moe_kernel` (the Triton kernel itself)

Run directly:
    python dev/debug_fused_moe.py

Run under debugpy and attach from Cursor/VSCode (see dev/launch.json):
    python -m debugpy --listen 5678 --wait-for-client dev/debug_fused_moe.py
"""

from __future__ import annotations

import os
import sys

# On Blackwell-class GPUs (sm_103a/B300) the ptxas bundled with `triton` may not
# know the target arch. Point Triton at the system CUDA toolchain *before*
# importing torch/triton so the override is picked up.
_SYS_PTXAS = "/usr/local/cuda-13.2/bin/ptxas"
if os.path.exists(_SYS_PTXAS):
    os.environ.setdefault("TRITON_PTXAS_PATH", _SYS_PTXAS)

import torch  # noqa: E402

# Make sure we hit the in-repo copy of sglang.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# fused_moe_triton_config.get_moe_configs reads get_global_server_args().enable_deterministic_inference
# at import-time inside the call path, so we have to seed a ServerArgs *before* importing fused_moe.
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe  # noqa: E402
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402


def build_inputs(
    num_tokens: int = 8,
    hidden: int = 1024,
    intermediate: int = 2048,
    num_experts: int = 8,
    top_k: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 0,
):
    """Build a self-contained set of MoE tensors.

    Shapes follow the conventions in fused_moe.py / fused_moe_triton_kernels.py:

        hidden_states: (num_tokens, hidden)
        w1           : (num_experts, 2 * intermediate, hidden)   # gate/up fused
        w2           : (num_experts, hidden, intermediate)
        topk_weights : (num_tokens, top_k)
        topk_ids     : (num_tokens, top_k)        int32, values in [0, num_experts)
    """
    g = torch.Generator(device=device).manual_seed(seed)

    hidden_states = torch.randn(num_tokens, hidden, dtype=dtype, device=device, generator=g)
    w1 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype, device=device, generator=g) * 0.02
    w2 = torch.randn(num_experts, hidden, intermediate, dtype=dtype, device=device, generator=g) * 0.02

    # Fake router: uniform-ish logits, then pick top-k.
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device, generator=g)
    topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=top_k, dim=-1)
    # Normalize within the kept top-k (typical convention).
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
    return hidden_states, w1, w2, topk_output


def main():
    assert torch.cuda.is_available(), "this debug script requires a CUDA GPU"
    torch.cuda.set_device(0)

    # Small enough to step through, large enough that BLOCK_SIZE_M padding still triggers.
    num_tokens = 8
    hidden = 1024
    intermediate = 2048
    num_experts = 8
    top_k = 2

    hidden_states, w1, w2, topk_output = build_inputs(
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        top_k=top_k,
    )

    runner_cfg = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,   # single-card, no EP
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=top_k,
        activation="silu",
        is_gated=True,
        inplace=False,                   # easier to reason about while debugging
        apply_router_weight_on_input=False,
    )

    print("[debug_fused_moe] hidden_states:", hidden_states.shape, hidden_states.dtype)
    print("[debug_fused_moe] w1:", w1.shape, "w2:", w2.shape)
    print("[debug_fused_moe] topk_ids:", topk_output.topk_ids)
    print("[debug_fused_moe] topk_weights:", topk_output.topk_weights)

    # >>> Set a breakpoint on the next line and step into fused_moe. <<<
    out = fused_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_output=topk_output,
        moe_runner_config=runner_cfg,
    )

    torch.cuda.synchronize()
    print("[debug_fused_moe] output:", out.shape, out.dtype)
    print("[debug_fused_moe] output.norm:", out.float().norm().item())


if __name__ == "__main__":
    main()

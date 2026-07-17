"""dtype_unit_sizes — sglang API matches hand-verified constants (v2).

Unit test (no GPU, no server boot). The previous v1 was flagged for
having two-paths-from-same-source: spec path was a line-for-line
transliteration of `Mamba2StateShape.create`, so a bug in that
function would be mirrored in the spec path. This version pins
EXPECTED constants derived from the model architecture by walking
Mamba2 semantics by hand:

  Per Mamba2 paper: each layer holds:
    - conv state of width `conv_dim = 2·n_groups·state_size + intermediate_size`
      and height `conv_kernel - 1` (the past inputs to the gated conv)
    - SSM state of shape `(num_heads, head_dim, state_size)`
  Per-rank sharding under TP=k:
    - conv_state: divide(conv_dim, k) channels
    - SSM state: divide(num_heads, k) heads

  intermediate_size = linear_value_head_dim × linear_num_value_heads
  n_groups          = linear_num_key_heads
  num_heads         = linear_num_value_heads
  head_dim          = linear_value_head_dim
  state_size        = linear_key_head_dim
  conv_kernel       = linear_conv_kernel_dim

These were hand-computed against each model's config.json + the
Mamba2 architecture spec (Dao & Gu 2024). If sglang's
`Mamba2StateShape.create` diverges from these constants, either:
  (a) sglang regressed, OR
  (b) the constants are wrong and our understanding of Mamba2 is off

Either is actionable; the previous "two paths from same source" wasn't.

Test matrix: (model × tp × ssm_dtype), kv_dtype-aware for KV per-token.

Note on env handling: each test backs up and restores
`SGLANG_MAMBA_SSM_DTYPE` to avoid leaking state into subsequent
tests in this or other suites.
"""
import json
import os
import sys

HUB = "/scratch/yuzhou/.cache/huggingface/hub"


def _resolve(model_name):
    base = f"{HUB}/models--Qwen--{model_name}"
    snap = os.listdir(f"{base}/snapshots")[0]
    return f"{base}/snapshots/{snap}/config.json"


# Hand-verified constants (computed independently by walking Mamba2 paper +
# each model's published config.json arch fields).
#
# Derivation example (Qwen3.5-9B, tp=1, ssm=float32):
#   intermediate_size = 128 * 32        = 4096
#   n_groups          = 16
#   num_heads         = 32
#   head_dim          = 128
#   state_size        = 128
#   conv_kernel       = 4
#   conv_dim          = 2*16*128 + 4096 = 8192
#   conv_numel/layer  = 8192 * (4-1)    = 24576
#   temporal_numel    = 32*128*128      = 524288
#   per layer (fp32 SSM + bf16 conv)
#     = 24576*2 + 524288*4              = 49152 + 2097152 = 2_146_304
#   × 24 linear_attention layers        = 51_511_296
HAND_VERIFIED_MAMBA = {
    # (model, tp, ssm_dtype) → bytes
    # tp=1
    ("Qwen3.5-9B",       1, "float32"):  51_511_296,
    ("Qwen3.5-9B",       1, "bfloat16"): 26_345_472,
    ("Qwen3.5-9B",       1, "float16"):  26_345_472,   # same as bf16 (both 2 bytes)
    ("Qwen3.5-35B-A3B",  1, "float32"):  64_389_120,
    ("Qwen3.5-35B-A3B",  1, "bfloat16"): 32_931_840,
    ("Qwen3.5-122B-A10B", 1, "float32"): 153_649_152,
    ("Qwen3.5-122B-A10B", 1, "bfloat16"): 78_151_680,
    # tp=2 (production scenario for 35B/122B)
    ("Qwen3.5-35B-A3B",  2, "float32"):  32_194_560,
    ("Qwen3.5-35B-A3B",  2, "bfloat16"): 16_465_920,
    ("Qwen3.5-122B-A10B", 2, "float32"): 76_824_576,
}

# KV per-token hand-verified
# Derivation (Qwen3.5-9B, tp=1, bf16):
#   L_attn=8, num_kv_heads=4, head_dim=256, dtype=2 bytes
#   per_token = 8 * 2(K+V) * 4 * 256 * 2 = 32_768
HAND_VERIFIED_KV = {
    ("Qwen3.5-9B",       1, "bfloat16"): 32_768,
    ("Qwen3.5-9B",       1, "fp8_e4m3"): 16_384,
    ("Qwen3.5-35B-A3B",  1, "bfloat16"): 20_480,
    ("Qwen3.5-35B-A3B",  1, "fp8_e4m3"): 10_240,
    ("Qwen3.5-122B-A10B", 1, "bfloat16"): 24_576,
    # tp=2 (KV head replication: num_kv_heads=2 or 4, tp=2 → halve)
    ("Qwen3.5-35B-A3B",  2, "bfloat16"): 10_240,
}


def _sglang_mamba_per_req(text_cfg, ssm_dtype, tp):
    """Call sglang's API and return mamba_per_req."""
    # Snapshot and override env
    old = os.environ.get("SGLANG_MAMBA_SSM_DTYPE")
    os.environ["SGLANG_MAMBA_SSM_DTYPE"] = ssm_dtype
    try:
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype,
        )
        shape = Mamba2StateShape.create(
            tp_world_size=tp,
            intermediate_size=text_cfg["linear_value_head_dim"] * text_cfg["linear_num_value_heads"],
            n_groups=text_cfg["linear_num_key_heads"],
            num_heads=text_cfg["linear_num_value_heads"],
            head_dim=text_cfg["linear_value_head_dim"],
            state_size=text_cfg["linear_key_head_dim"],
            conv_kernel=text_cfg["linear_conv_kernel_dim"],
        )
        n_linear = text_cfg["layer_types"].count("linear_attention")
        params = Mamba2CacheParams(
            shape=shape, layers=list(range(n_linear)),
            dtype=mamba2_state_dtype(None),
        )
        return params.mamba_cache_per_req
    finally:
        if old is None:
            os.environ.pop("SGLANG_MAMBA_SSM_DTYPE", None)
        else:
            os.environ["SGLANG_MAMBA_SSM_DTYPE"] = old


def _spec_kv_per_token(text_cfg, kv_dtype, tp):
    dt = {"bfloat16": 2, "float16": 2, "float32": 4,
          "fp8_e4m3": 1, "fp8_e5m2": 1}[kv_dtype]
    L_attn = text_cfg["layer_types"].count("full_attention")
    n_kv = text_cfg["num_key_value_heads"]
    hd = text_cfg["head_dim"]
    per_token = L_attn * 2 * n_kv * hd * dt
    if n_kv >= tp:
        per_token //= tp
    return per_token


# ---------- sub-tests ----------

def test_1_sglang_matches_hand_verified_constants():
    """For every (model, tp, ssm_dtype) cell: sglang API output ==
    pre-computed independent constant. Byte-exact equality."""
    failures = []
    print(f"    {'model':22s} {'tp':>3s} {'ssm':>9s} "
          f"{'expected':>14s} {'sglang':>14s}  match")
    print("    " + "-" * 75)
    for (model, tp, ssm), expected in sorted(HAND_VERIFIED_MAMBA.items()):
        cfg = json.load(open(_resolve(model)))
        text_cfg = cfg.get("text_config", cfg)
        actual = _sglang_mamba_per_req(text_cfg, ssm, tp)
        ok = (actual == expected)
        mark = "✓" if ok else "✗"
        print(f"    {model:22s} {tp:>3d} {ssm:>9s} {expected:>14,} "
              f"{actual:>14,}  {mark}")
        if not ok:
            failures.append(
                f"{model} tp={tp} ssm={ssm}: expected={expected:,} "
                f"actual={actual:,} delta={actual - expected:+,}")
    assert not failures, (
        "sglang API diverged from hand-verified constants:\n  "
        + "\n  ".join(failures))


def test_2_kv_per_token_matches_hand_verified():
    """Byte-exact match for KV per-token across (model × tp × kv_dtype)."""
    failures = []
    print(f"    {'model':22s} {'tp':>3s} {'kv':>9s} "
          f"{'expected':>10s} {'spec':>10s}  match")
    print("    " + "-" * 65)
    for (model, tp, kv), expected in sorted(HAND_VERIFIED_KV.items()):
        cfg = json.load(open(_resolve(model)))
        text_cfg = cfg.get("text_config", cfg)
        actual = _spec_kv_per_token(text_cfg, kv, tp)
        ok = (actual == expected)
        mark = "✓" if ok else "✗"
        print(f"    {model:22s} {tp:>3d} {kv:>9s} {expected:>10,} "
              f"{actual:>10,}  {mark}")
        if not ok:
            failures.append(
                f"{model} tp={tp} kv={kv}: expected={expected:,} "
                f"actual={actual:,}")
    assert not failures, "\n  ".join(failures)


def test_3_env_var_takes_effect_each_call():
    """sglang reads SGLANG_MAMBA_SSM_DTYPE at call-time (not cached at
    import). Verify by flipping it twice in the same process."""
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    import torch

    old = os.environ.get("SGLANG_MAMBA_SSM_DTYPE")
    try:
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "bfloat16"
        d1 = mamba2_state_dtype(None)
        assert d1.temporal == torch.bfloat16, \
            f"first call: got {d1.temporal}, expected bfloat16"

        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "float32"
        d2 = mamba2_state_dtype(None)
        assert d2.temporal == torch.float32, \
            f"second call: got {d2.temporal}, expected float32"

        # Default (no env) → fp32
        del os.environ["SGLANG_MAMBA_SSM_DTYPE"]
        d3 = mamba2_state_dtype(None)
        assert d3.temporal == torch.float32, \
            f"unset: got {d3.temporal}, expected fp32 default"

        print(f"    env=bf16 → {d1.temporal} ✓")
        print(f"    env=fp32 → {d2.temporal} ✓")
        print(f"    env unset (default) → {d3.temporal} ✓")
    finally:
        if old is None:
            os.environ.pop("SGLANG_MAMBA_SSM_DTYPE", None)
        else:
            os.environ["SGLANG_MAMBA_SSM_DTYPE"] = old


def test_4_invalid_dtype_falls_back_to_default():
    """sglang warns and falls back to fp32 on invalid env value
    (mamba_utils.py:96-101). Test this safety net."""
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    import torch

    old = os.environ.get("SGLANG_MAMBA_SSM_DTYPE")
    try:
        os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "fp4_blackwell"  # bogus
        d = mamba2_state_dtype(None)
        assert d.temporal == torch.float32, \
            f"invalid dtype should fall back to fp32, got {d.temporal}"
        print(f"    env='fp4_blackwell' (bogus) → {d.temporal} (fp32 fallback) ✓")
    finally:
        if old is None:
            os.environ.pop("SGLANG_MAMBA_SSM_DTYPE", None)
        else:
            os.environ["SGLANG_MAMBA_SSM_DTYPE"] = old


# ---------- runner ----------

def main():
    tests = [
        ("1 sglang mamba_per_req matches hand-verified constants",
         test_1_sglang_matches_hand_verified_constants),
        ("2 KV per-token matches hand-verified constants",
         test_2_kv_per_token_matches_hand_verified),
        ("3 env var SGLANG_MAMBA_SSM_DTYPE takes effect each call",
         test_3_env_var_takes_effect_each_call),
        ("4 invalid SSM dtype falls back to fp32",
         test_4_invalid_dtype_falls_back_to_default),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\ndtype_unit_sizes: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

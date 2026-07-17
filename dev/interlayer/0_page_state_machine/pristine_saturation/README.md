# pristine_saturation — engine-level dtype-matrix ground-truth

What it tests: sglang's actual allocator output (parsed from
server.log) matches `HAND_VERIFIED_{MAMBA,KV}` constants (from
`0_page_state_machine/dtype_unit_sizes/`) across (model × kv_dtype
× ssm_dtype) cells. Catches the case where sglang silently allocates
different bytes than its API promises.

## Driver + validator

- `run_pristine.sh` — boots pristine sglang per cell
  (`no SGLANG_HIMA --radix-eviction-policy lru`), captures
  server.log + cell.json descriptor
- `validate_pristine.py` — parses logs, asserts per-cell `(mamba_per_req,
  kv_per_token)` matches HAND_VERIFIED within 0.1%

## Reproduce — 6-cell matrix

Default `CELLS` covers 4 cells on Qwen3.5-9B. Override `CELLS` to add
larger models. Run sequentially (each cell ~30-180s boot).

```bash
cd /scratch/yuzhou/projects/sglang

# 4-cell 9B sweep (default)
GPU=3 PORT=30055 OUT_DIR=/tmp/pristine_9b \
    bash dev/interlayer/0_page_state_machine/pristine_saturation/run_pristine.sh

# Add 35B cells via CELLS override (one per line in env-var)
CELLS=$'35B_auto_fp32   Qwen3.5-35B-A3B  1 auto  float32\n35B_auto_bf16   Qwen3.5-35B-A3B  1 auto  bfloat16' \
    GPU=3 PORT=30055 OUT_DIR=/tmp/pristine_35b \
    bash dev/interlayer/0_page_state_machine/pristine_saturation/run_pristine.sh
```

Re-validate any output dir:

```bash
.venv/bin/python dev/interlayer/0_page_state_machine/pristine_saturation/validate_pristine.py --out-dir /tmp/pristine_9b
```

## Result table

| cell | model | tp | kv | ssm | mamba Δ% | kv Δ% |
|---|---|---|---|---|---|---|
| 9B_auto_bf16   | Qwen3.5-9B | 1 | auto    | bfloat16 | -0.008% | -0.008% |
| 9B_auto_fp32   | Qwen3.5-9B | 1 | auto    | float32  | -0.012% | +0.008% |
| 9B_fp8_bf16    | Qwen3.5-9B | 1 | fp8_e4m3 | bfloat16 | -0.008% | -0.008% |
| 9B_fp8_fp32    | Qwen3.5-9B | 1 | fp8_e4m3 | float32  | -0.012% | +0.008% |
| 35B_auto_bf16  | Qwen3.5-35B-A3B | 1 | auto | bfloat16 | +0.018% | -0.015% |
| 35B_auto_fp32  | Qwen3.5-35B-A3B | 1 | auto | float32  | -0.025% | +0.026% |

All within 0.1% tol; residual is sglang's `%.2f GB` log-print precision
floor. Tolerance derivation in `validate_pristine.py:validate_cell` docstring.
Off-by-one root cause (sglang allocates `size + page_size` slots) is
documented in `dev/interlayer/design.md` under "Padded slot 0".

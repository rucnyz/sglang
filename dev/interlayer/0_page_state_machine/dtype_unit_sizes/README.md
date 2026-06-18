# dtype_unit_sizes — spec-only per-unit sizes match sglang API

What it tests: hand-verified per-slot byte sizes (computed independently
by walking the Mamba2 paper + each model's config.json arch fields)
match sglang's `Mamba2StateShape.create` API output byte-exact, across
(model × tp × ssm_dtype) cells. Also verifies the
`SGLANG_MAMBA_SSM_DTYPE` env-var takes effect at call-time, not
import-cached.

4 sub-tests:
- test_1: sglang API byte-exact matches `HAND_VERIFIED_MAMBA` across
  **10 cells** — a selected subset from {3 models × tp ∈ {1,2} × ssm
  ∈ {fp32, bf16, fp16}} = 18 possible; only the production-relevant
  combos are pinned (fp16 only on 9B; bf16 not present on 122B tp=2).
- test_2: `HAND_VERIFIED_KV` byte-exact across 6 cells
- test_3: env-var `SGLANG_MAMBA_SSM_DTYPE` takes effect call-time
  (not import-cached). Test backs up + restores env to avoid leak
- test_4: invalid dtype falls back to fp32 (safety net)

The `HAND_VERIFIED_*` dicts are the source-of-truth: hand-computed
from Mamba2 paper + per-model arch fields, **independent of sglang
code**. If sglang regresses, this test catches it; if our understanding
of Mamba2 is wrong, the test also catches it (both directions actionable).

**Scope**: covers only the three Qwen3.5 models in production scope.
If sglang adds a new mamba arch, the test does NOT auto-iterate
sglang's full model registry — adding a new arch requires extending
`HAND_VERIFIED_*` with hand-derived constants for that arch. This is
deliberate: the value of the test is the independent hand derivation,
which can't be automated without losing the cross-check.

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
.venv/bin/python dev/interlayer/0_page_state_machine/dtype_unit_sizes/test_dtype_unit_sizes.py
```

Pure-Python; takes ~1s; no GPU. Reads `config.json` from the HUB
snapshots in `~/.cache/huggingface/hub/`.

## Result

4/4 PASS (v2 after strict review — earlier "spec path" was a
line-for-line transliteration of `Mamba2StateShape.create`, making
it a two-paths-from-same-source check; rewritten to use independently
hand-derived constants).

## Why it matters

This folder is the spec-level check; the engine-level check lives in
[`../pristine_saturation/`](../pristine_saturation/). Together they
bracket the entire size-computation pipeline:
- spec level (here): paper math → sglang API (no engine boot)
- engine level (`../pristine_saturation/`): sglang API → actual
  allocator output (engine boot)

If sglang adds a new mamba arch or changes Mamba2 layout, this test
flags the constant divergence immediately (when the new arch is
added to `HAND_VERIFIED_*`).

## Downstream consumers

`../pristine_saturation/validate_pristine.py` imports
`HAND_VERIFIED_MAMBA` and `HAND_VERIFIED_KV` from
`test_dtype_unit_sizes.py` via `sys.path`. Renaming the file or the
two dict symbols breaks that import — loudly (ImportError), not
silently. Keep the symbol names stable; if the test file is ever
renamed again, update `../pristine_saturation/validate_pristine.py`
at the same time.

# va_reservation_hbm — VA reservation is free (HBM = sglang baseline)

What it tests: `cuMemAddressReserve` of N GB of virtual address space
costs ZERO HBM (it's pure VA bookkeeping, no physical backing). After
the interlayer wires in its arena, total HBM use should equal the
sglang baseline plus only the actually-mapped chunks — no surprise
overhead from "reserving 71 GiB of VA".

7 sub-tests:
- test_0: noise-calibrated tolerance — `TOL = max(4σ, 64 KiB)` so
  the 64 KiB literal is the floor on quiet platforms; if the GPU is
  noisier, `4σ` widens the tolerance automatically (no flake risk)
- test_1: handle lifecycle with **identity check on grow** (no
  release+recreate cycle)
- test_2: `cuMemAddressReserve` 2 MiB → 71 GiB all delta 0 MiB
  (aligned to chunk_size — the alignment bug was previously masked
  by an over-conservative "cap size" workaround that hid the root
  cause; this test forces the alignment to be right)
- test_3: 2 arenas + 1 shared pool — HBM aggregate AND aliasing `is`
  checks (no double-count, no list copy)
- test_4: map/unmap ×100 loop drift 0 MiB (per-op floor < 320 bytes)
- test_5: owned vs external paired equal
- test_6: lazy init via empty-pool + arena-shortfall path verified

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/0_page_state_machine/va_reservation_hbm/test_va_reservation_hbm.py
```

No env-vars; takes ~30s; needs a GPU. Scope is mechanism-only;
end-to-end sglang boot with arena is covered by
[`../pristine_saturation/`](../pristine_saturation/).

## Result

7/7 PASS (v3 after 2 strict reviews). Noise-calibrated TOL = 4σ =
64 KiB. The handle-identity-on-grow check
([test_1](test_va_reservation_hbm.py)) is load-bearing: it caught an
earlier "release+recreate cycle" where the pool would silently swap
handles, breaking the FirePlan invariant.

Commit: `428b3c6b91`

# fire_wall_curve — per-chunk actuator-wall cost ≤ physical floor

This is the **actuator-wall** half of design.md §"fire_wall_curve +
decode_wall — per-batch-size physical-floor budgets"; the
**decode-stream-wall** half is pinned by [`../decode_wall/`](../decode_wall/).

What it tests: the per-n curve of cross-arena transfer wall-time. The
physical floor is ~70 µs per chunk (TLB invalidation on cuMemUnmap —
NVIDIA driver level, not optimizable). This test verifies that batched
unmap/map calls scale linearly with n (not super-linearly) and stay
within the per-n budget.

5 sub-tests covering n ∈ {4, 8, 16, 30}:
- n=4: small KV transfer
- n=8: typical fire size
- n=16: large KV transfer
- n=30: mamba per-slot transfer (mamba's per-slot bytes are larger,
  so 30 chunks is realistic for a per-mamba-slot fire)
- All assert p50 wall-time ≤ `n × 70 µs × 1.4 = n × 98 µs` budget
  (per-chunk physical floor + 40% jitter slack) AND p99/p50 < 5
  (no tail blowups)

Per-chunk breakdown logged: cuMemUnmap vs cuMemMap time, plus the
1.11× ratio across n values (confirming per-chunk cost is constant,
not super-linear).

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/0_page_state_machine/fire_wall_curve/test_fire_wall_curve.py
```

No env-vars; takes ~30s; needs a GPU.

## Result

5/5 PASS. Per-chunk costs 68.9–76.5 µs (1.11× ratio across n);
mamba per-slot n=30 = 2060 µs total wall. All within budget.

Why this matters: design.md's fire-cost model assumes per-fire wall is
`O(n) × ~70µs/chunk + O(1)`. If a future arena refactor introduces
super-linear cost (e.g. forgets to batch cuMemSetAccess), this test catches it.

Commit: `428b3c6b91`

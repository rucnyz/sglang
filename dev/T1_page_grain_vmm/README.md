# T1: Switch VMM granularity 256 MB chunk → 2 MiB page

Task #73. Foundation step for ideal-mode design (paper §3.2.1).

## What changes

Default `SGLANG_ARENA_CHUNK_BYTES` goes from `64 * 1024 * 1024` (64 MiB) to
`2 * 1024 * 1024` (2 MiB). 2 MiB is CUDA VMM's native page size on H200
(returned by `cuMemGetAllocationGranularity` with `RECOMMENDED`), so this
matches the underlying hardware granularity exactly. No multi-page
coarsening, no padding.

Code change is two-line: `memory_pool.py:311` and `memory_pool.py:1188`,
both `str(64 * 1024 * 1024)` → `str(2 * 1024 * 1024)`. Everything else in
the arena code already parameterizes off `chunk_size`; no semantic changes.

## Why

The chunk-grain (256 MB) actuator commit-success rate is dominated by random
live-block pinning: with $N$ active sub-agents holding $\sim 1$ MB of live KV
each spread across the pool, the probability that some block lands in a
30-chunk over-cap range scales as $\sim 30 \times 256\,\text{MB}/\text{pool size}
\times N$. At 256 MB chunks on Qwen3.5-35B-A3B / H200 with $N=120$, this
gives $\sim 88\%$ abort rate per fire — only $\sim 1/8$ commit. Going to 2 MiB
reduces the per-page pinning probability by $128 \times$, taking commit rate
to $\sim 95\%$+ (paper §3.2.1).

## Flag

`SGLANG_ARENA_CHUNK_BYTES` continues to be the override:
- `=2097152` (default after T1): 2 MiB pages.
- `=67108864` (legacy): 64 MiB chunks. Used in A/B and as the
  baseline-compat path during the transition.

The arena-on path is itself still gated by `SGLANG_ARENA_SHARED=1`. Default
serving (no arena) is unaffected.

## Expected behavior

The numbers below are the \emph{design expectation} from paper §3.2.1, not
measurements made by T1 itself. T1 only verifies boot + smoke at 2 MiB
granularity. Real per-fire wall and commit-success rate are measured by T3
(smart over-cap selection) and T7 (M2 swarm validation).

| condition | boot time | per-fire wall (design) | commit rate (design) | tokens/chunk |
|---|---|---|---|---|
| 64 MiB chunks (legacy) | $\sim$5 s of cuMemMap | $\sim$3 s/fire (30 chunks) | $\sim$12% | $\sim$6500 (KV @ 10 KB/token) |
| 2 MiB pages (T1 default) | $\sim$8--12 s of cuMemMap | $\sim$0.1 s/fire (3840 pages) | $\sim$95% (with T2+T3) | $\sim$200 (KV @ 10 KB/token) |

The boot-time delta (~3-7 s extra cuMemMap calls) is a one-time cost; the
per-fire wall reduction (~30×) and commit-success increase (~8×) are
expected to amortize at any nontrivial fire frequency. \emph{Both deltas
are unverified by T1 alone}; T1's contribution is the granularity flip
that makes T2+T3 viable.

## A/B perf comparison

See `results/ab_smoke.txt` after running `bash reproduce.sh`. The on/off
delta is what this task contributes.

## How to run

```bash
bash dev/T1_page_grain_vmm/reproduce.sh    # smoke + A/B + capture results
bash dev/T1_page_grain_vmm/test/test_boot_smoke.sh  # quick boot + 5-prompt smoke
```

## Status

- [x] Design note
- [x] Default flipped in `memory_pool.py` (lines 311, 1188)
- [x] Boot smoke at 2 MiB on Qwen3.5-35B-A3B / H200 — boot 106 s, 5-prompt
  smoke 1.32 s, no errors
- [x] A/B perf result captured — see `notes.md` and `results/ab_smoke.txt`.
  Boot time and smoke latency identical within noise; arena-init log shows
  page-grain reaches `tokens_per_chunk=1` on the mamba pool (every slot
  independently mappable, which is exactly what T3 needs).
- [x] Known shortcoming: growth-budget shrinkage at page-grain — `T5: VA
  overcommit at boot` will resolve.

## Followups

T2 (allocator placement bias) and T3 (smart over-cap selection) build on
this; T3 specifically gets its big drain-success-rate win from the page
granularity established here.

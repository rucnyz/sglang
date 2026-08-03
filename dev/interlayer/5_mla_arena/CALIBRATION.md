# Kimi-Linear-48B cost-model calibration (2026-07-31)

## Run

```
CUDA_VISIBLE_DEVICES=2,4 EXTRA_FLAGS="--tp 2 --trust-remote-code \
  --max-mamba-cache-size 16 --disable-cuda-graph" MEM_FRACTION=0.85 REPEATS=3 \
  bash dev/eval/cost_model/calibrate.sh moonshotai/Kimi-Linear-48B-A3B-Instruct H200
```

Runs 1–2 completed full 21-length sweeps; run 3's bench wedged post-sweep in
CUDA teardown as an unkillable R-state process (SIGKILL ignored; the known
driver-unwind zombie class), so the fit uses runs 1–2.

## Pathological bench shapes (reproducible in BOTH runs)

`bench_one_batch` on this stack measures wildly non-monotonic latencies at
specific prefill lengths, identical across independent processes:

| L | run1 ms | run2 ms | neighbors |
|---|---------|---------|-----------|
| 512  | 359 | 423 | 256: 54–70, 768: 56 (run2) |
| 1536 | 37 636 | 37 973 | 1024: 48, 2048: 49 |
| 2560 | 497 | 475 | 3072: 52 |
| 16384 | 804 | 655 | 24576: 311 |

L=1536 at ~38 s (≈780× its neighbors) is a per-shape kernel/autotune
pathology of the KDA prefill path in the bench harness, NOT steady-state
serving cost (upstream CI serves ~1k-token GSM8K prompts at normal speed on
this tree; our smokes generate normally). The c_KV curve prices eviction /
re-prefill of multi-thousand-token cache segments, so calibration uses the
smooth steady-state envelope:

- keep L ≤ 8192 rows with ms < 150
- keep L > 8192 rows with ms < 0.025·L + 100
- (30 of 44 rows kept; the 4 shapes above + first-call warmups dropped)

Worth an upstream look eventually: reproduce with
`bench_one_batch --input-len 1536` alone. Tracked as a curiosity, not a
blocker — real traces replay through chunked prefill.

## Fit (clean set, runs 1–2)

```
c_KV(L) = 6.3711e-08·L² + 9.6780e-03·L + 47.743 ms
quad RMS = 21.6 ms (linear 90.6 ms → the L² term is real)
```

Cross-model sanity: α within [4.9e-08 (Nemotron), 1.3e-07 (35B)];
γ=47.7 ms between 35B (24.2) and Nemotron (70.4). c_M = 0 as for all
hybrids (recompute folded into c_KV).

Exported in `run_arm.sh`'s `*Kimi*` branch.

## lh -6.3% decomposition (2026-08-01)

lh@64 N3: base(Unified) 1510.8±4.1 / sys(HiMA,MambaTree) 1415.1±9.1.
Isolation single-rep with SGLANG_FORCE_MAMBA_RADIX_TREE=1 (tree only, no
HiMA): 1488.3 (P50 139). So Unified->Mamba tree costs −1.5%; HiMA's control
plane on the same tree costs −4.9% in this KV-tree-heavy regime. swarm and
shifting are cost-neutral or better (sys 866.2 vs 864.6; 1207.7 vs 1201.2,
P50 −6%). Follow-up engineering candidates: LPB scoring cost on deep trees,
admitter per-arrival work; not blocking the paper row.

## Deep-gate regression root cause + fix (2026-08-02)

Gates 3 (deep-long@256) and 4 (deep-shifting@128) had sys LOSING to base
(-12%/-15%). Forensics (wf_41514793): a single warmup k2m fire, priced off a
~1000x-physical R_m spike (mamba momentarily full at boot; LPB loss n_b
multiplier over-counts), shrank the KV pool 21.7% (planner asked 140 pages,
actuator granted 420 chunks — a separate planner/actuator unit-contract
mismatch, amplified by Kimi's asymmetric 7/20 subpools, lcm=140) and the
free->free-only return path could never reclaim it (mamba chunks fragmented
by snapshots; 7,770 correct m2k decisions, 99% "no free source pages").
Control-plane overhead, admitter churn, retract storms all REFUTED
(0.007% thread time, zero retractions).

Fix (minimal): clamp the mamba loss signal at planner intake to
slots_evicted x c_kv(pool_max) — the physical rebuild ceiling. Without the
mispriced spike the ruinous k2m never fires. OPEN (tracked, not blocking):
(a) planner/actuator page-unit contract (n_src multiplication), (b) k2m
irreversibility on fragmented mamba arenas, (c) mamba eviction accounting
blind spot (churn bypasses LPB tally so R_m=0 under load).

## Async-fire deadlock (2026-08-02, open item)

gate5b hung 4 min into replay: scheduler's LAST line was "[admission-cap]
grew pool.size 302 -> 378" one second after "async fire worker started";
TP workers spun at 100% GPU/CPU with zero batch output (driver-level
deadlock between the fire worker's cuMemMap/Unmap and the scheduler
thread's ReqTokenVAArena grow — SharedHandlePool is lock-free by design
and the two threads raced). Mitigation: SGLANG_HIMA_FIRE_ASYNC=0
(serialize fires on the scheduler thread) for the Kimi campaign. Proper
fix (open): arena-level mutex around cuMem ops or defer cap-grow while a
fire is in flight.

## Deep-pressure campaign wrap-up (2026-08-03)

Full gate ladder on depth-first real slices (data archived at
figures/data/kimi48b_deep in the paper repo):

| gate | config | result |
|---|---|---|
| 2 | base top100@128 | 1177.6 / P99 4074, KV 0.94, no eviction |
| 3 | A/B full-deep@256 | base 774 cache 0.66; sys(-12%) — capacity theft |
| 4 | A/B deepshift@128 | base 787 cache 0.74; sys(-15%) — same theft |
| 6 | sys@128 (fixes, no drain) | 1184.9 / P99 3602 (-12% vs base) |
| 7 | sys@128 + m2k drain | 1186.7 / P99 3256 (-20% vs gate2 r1) |
| fills | base N3 / sys N3 @128 | 1184.7±7.4 P99~3618 vs 1185.8±6.7 P99~3808 |

Net: at mid pressure the fixed sys is PAR with base (single-rep P99 wins do
not survive N3 averaging); at deep pressure sys still loses (LPB scoring on
deep trees is the leading suspect; sys@256 with LRU was killed by a boot
storm before completing). m2k drain works (first real KV growth: live
4.536M -> 4.606M) but the yield is small because MLA KV is cheap — the
mamba pool's entire donation is worth ~25% of a pool that already runs at
0.97 hit. Structural conclusion: Kimi-Linear's MLA+sparse-MoE profile
leaves little for cross-pool reallocation to win at these scales; the
row's honest story is cost-neutrality plus tail improvements at mid
pressure.

Box note: after ~36 h of gate iterations the H200 driver on GPUs 1/6 and
2/4 degraded (probe-pass no longer predicts boot survival; unwind windows
stretched to 2 h+). Reboot recommended before further GPU work.

## Gate 10: k2m tick-path starvation — the m2k twin (2026-08-03)

Mamba-bound swarm probe (swarmdense@64, `--max-mamba-cache-size 320`, the
paper's recurrent-cache sizing precedent) exposed that the **tick-path k2m
fire had never moved a page on Kimi**. Two compounding defects in
`BudgetAgent._maybe_fire`:

1. **Unit confusion**: the k2m branch sized plans as `min(n_free,
   lcm_pages)`. `lcm_pages` counts SUBPOOL-chunks (Kimi: lcm(7,20)=140),
   but the plan's unit is source lockstep pages (1 KV page = 7 subpool
   chunks), so one atomic LCM unit is 140/7 = **20 KV pages**. The
   actuator floors each fire to an LCM multiple — any plan below 20 pages
   moves zero.
2. **Free-only supply**: at steady state the radix cache owns all
   nominally-idle KV, so genuinely-free pages hover at ~4-5 — the same
   starvation m2k had before `SGLANG_XPOOL_M2K_DRAIN`, mirrored.

Observed signature (gate10 sys, first run): ~960 k2m fires per rank at
~1/s, every one `unmapped=0 granted=0` (~110-150us pure overhead each),
mamba pinned at 320 slots / usage 0.80 with KV at 0.20 — the exact
supply-side thrash the gate was built to relieve, unrelieved.

Fix (mirrors m2k):
- `SGLANG_XPOOL_K2M_DRAIN` (default on): tick k2m passes
  `allow_drain=True`; Stage-2 completes the unit from cold cached KV in
  loss-aware victim order. Free pages are still taken first (Stage-1),
  so models whose free supply already reaches one LCM are unchanged.
- Sizing: `min(max(n_free, lcm_pages // n_kv_subpools), lcm_pages)` —
  identical to the old size whenever `n_free >= one LCM unit` (all Qwen
  gates), floored up to exactly one unit otherwise.
- Sub-LCM refuse guard (both directions): after the working-set floor
  clamps, a target below one LCM unit of source pages aborts with
  `fire_abort_reason` instead of paying cap-barrier for a guaranteed
  no-op. This also ends the no-op fire storm.

Per-fire economics on Kimi: one LCM unit converts 20 KV pages = 327,680
tokens (4.2% of the 7.75M pool) into 7 mamba pages = **+126 slots (+39%
of CAP=320)** — one to three fires relieve the entire swarm-dense
snapshot thrash while the KV tree barely notices.

### Gate 10 addendum: the dst-side clamp (bug 8, same afternoon)

The agent-side fix alone still granted zero: `MambaArenaActuator.
grow_headroom_pages()` = allocator `max_size - live_size`, and the
`mamba_max_size` branch in `HybridReqToTokenPool` preferred the
req-pool-keyed formula (`req.max_size*3` REQ slots = 405) over the
arena factor formula whenever the req pool is in dynamic-cap mode —
i.e. on every HiMA run. CAP=320: headroom 85 slots = 4 dst pages < one
7-page LCM unit -> every grant floors to zero; CAP=1692 (all previous
Kimi sys runs): 405 < live, headroom 0 — **k2m was structurally dead on
Kimi in every run to date**, independent of the tick-path starvation.
Fix: when arena-backed, `mamba_max_size = max(mamba_size *
SGLANG_XPOOL_MAMBA_MAX_FACTOR, req.max_size*3)` (factor default 4; the
req-keyed value kept only as a floor). CAP=320 -> max_size 1280,
headroom 960 slots = 53 pages. conv_state (physically allocated at
max_size) grows ~0.9 GB at this shape — within MEMFRAC 0.80 slack.

Run ledger: `sys_nofix` = pre-fix storm (772.2, cache 0.474, P99 10.5s);
`sys_lcmfix` = agent-side fix only (fires 1/min, still granted=0 — an
even cleaner overhead-only control); round-3 sys = both fixes.

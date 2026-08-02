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

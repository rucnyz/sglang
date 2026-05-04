# Cross-TP-rank coordinated budgeter

## Motivation
The current design is scoped to a single TP rank's view of HBM (each rank has
its own pool VA reservations, free handles, and budgeter agent). Production
deployments at TP=2/4/8 run multiple ranks per model; under cross-pool
asymmetric workloads each rank may want to reallocate independently. If the
ranks' decisions are uncoordinated they can fire in opposite directions
(one rank shrinks KV while another grows it), or they can fire serially when
batched cross-rank reallocation would have been faster.

## Approach
- Per-rank budgeter agents continue to compute local $\hat V_i'$ and target
  allocation $m_i^\star$.
- A single coordinator (rank 0 by convention) collects per-rank targets each
  control tick, runs the global Lagrange solve $\sum_{\text{rank } r}
  V_{i,r}'(m_{i,r}^\star) = \lambda^\star$, and broadcasts the per-rank
  consensus targets back via NCCL `all_reduce` or similar.
- All ranks fire their local actuator in lockstep, gated on a barrier so
  cross-rank kernels (attention all-reduce, MoE token shuffling) don't observe
  partial reallocation.

## Challenges
- **NCCL barrier per fire**: a cross-rank `all_reduce` is ~tens of μs at
  TP=8; on top of the ~100 ms ideal-mode fire wall this is sub-1% overhead.
  But adds a hard synchronization point that disrupts overlapped compute.
- **Lockstep firing semantics**: if one rank's drain predicate fails (e.g.,
  live-block migration takes longer there), all ranks must wait. Worst-case
  wall = max over ranks, not avg.
- **Cross-rank target divergence**: if per-rank workloads differ within the
  same model (rare but possible with EP routing), forcing identical pool
  splits per rank may not be optimal. A relaxation: allow per-rank divergence
  up to ε of the consensus, fire only on max-divergence pair.
- **Failure recovery**: if one rank crashes mid-fire, the others must roll back
  their local actuator state to keep the model coherent. This introduces a
  transactional-fire path absent in single-rank.

## When to do it
Production deployments at TP ≥ 2 where the workload is severe enough that
inter-pool fires happen on the same timescale as inter-batch attention all-
reduce (i.e., minute-scale fires). For paper, single-rank already exercises
the algorithmic core; multi-rank is engineering complexity for a quantitative
extension, not a qualitative one. Worth a follow-up paper or a section in a
journal extension.

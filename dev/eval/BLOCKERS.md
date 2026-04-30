# Eval blockers — append-only log

When a setting from `SETTINGS.md` can't run, document the blocker here and move on.

## Format

Each entry:
- **Setting:** which one
- **Blocker:** what's preventing it
- **Date observed:** when
- **Workaround:** if any
- **Resolved at:** if/when

---

## Path-axis dispatcher (Settings 5.A, 5.B, A6) — **BLOCKED**

- **Setting:** 5.A (`dev/eval/09_path_dense.sh` not written), 5.B, A6
- **Blocker:** path-axis dispatcher itself is not implemented. Paper §4.5 calls for a per-batch dispatcher that detects `K≤1` batches and routes them through a prefill-only fast path that bypasses paged-attention and DeltaNet slot pool. Estimated ~600 LoC in scheduler + model-runner. Substantial integration risk; needs user discussion before autonomous start.
- **Date observed:** 2026-04-30 night
- **Workaround:** none. Setting 5 cannot run until dispatcher exists.
- **Resolved at:** —

## Setting 1 (24-h phase-shift trace) — **partially blocked**

- **Setting:** 1 (`dev/eval/01_phase_shift_trace.sh` not written)
- **Blocker:** trace requires three datasets (Phase A: alpaca+lora, Phase B: rerank-style ShareGPT, Phase C: WildChat 8-turn). The pd_exp generator at `aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py` produces (1) and approximates (2). WildChat for Phase C needs `pd_exp/multiturn/export_dataset.py` with `--dataset wildchat` — first time may need HF download.
- **Date observed:** 2026-04-30 night
- **Workaround:** start with Phase A + B only on the trace and document Phase C as "coming after WildChat export."
- **Resolved at:** —


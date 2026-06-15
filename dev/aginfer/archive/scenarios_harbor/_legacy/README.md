# _legacy — pre-N=3 era cycles (kept for history, not in git)

These cycle dirs are from before the N≥3 + B/O alternation
methodology landed (see `../_shared/methodology.md`).  Most are
N=1 single shots, smoke tests, or top-level orchestrator summary
dirs whose data is duplicated under arms/.

**Not tracked in git** — kept on disk so the original data can
be re-mined if a question arises.  Do not rely on these numbers
for paper claims; re-run under current methodology instead.

## Inventory

| dir | era | what |
|---|---|---|
| `run_A_hicache_n32` | N=1 early sanity | first HiCache-on sanity check |
| `run_C_hicache_cap256k` | N=1 | early A3 precursor — 256K pool but full max_completion |
| `run_D_nohicache_cap256k` | N=1 | 256K pool + HiCache OFF |
| `run_E_nohicache_cap512k` | N=1 | 512K pool + HiCache OFF |
| `run_F_hicache_cap512k` | N=1 | 512K pool + HiCache ON (paper §8 historical) |
| `run_F_prime_lru_same_topology` | N=1 | F-equivalent under LRU |
| `run_G_thunderagent` | N=1 | original TA single-shot (Run G in paper drafts) |
| `run_H_ours_greedy` | N=1 | original OURS_greedy single-shot (Run H) |
| `run_H_prime_ours_signfix` | N=1 | post-bugfix H' single-shot |
| `run_K_full` | N=1 | K-full single-shot (no cycle suffix) |
| `run_K_ka` | N=1 | K-a (admission OFF) single-shot |
| `run_K_kv_off` | N=1 | K-kv_off (kv_scheduler OFF) single-shot |
| `run_K_matrix_20260526_234639` | summary dir | top-level orchestrator summary for the N=3 K matrix |
| `run_H_prime_now_matrix_20260527_153233` | summary dir | top-level orchestrator summary for the H'_now matrix |
| `run_4arm_matrix_20260527_204352` | summary dir | top-level orchestrator summary for the 4-arm matrix |
| `run_K_a3_a3_tb_smoke_210826` | A3 smoke | traceback-instrumentation smoke (debug) |
| `run_K_a3_a3_tb_full_212405` | A3 smoke | traceback-instrumentation full cycle (debug) |
| `run_K_a3_a3_finegrain_231209` | failed | SGLANG_TP unbound — never ran past sglang startup |
| `run_K_kv_off_smoke_232839` | smoke | kv_off smoke |
| `run_TA_now_smoke_194520` | smoke | TA smoke |
| `run_TA_now_smoke_195601` | smoke | TA smoke |
| `run_TA_now_smoke_200631` | smoke | TA smoke |
| `SUMMARY.md` | doc | old top-level results SUMMARY (pre-scenarios/ reorg) |

## Why keep them

* If a methodological question comes up ("what was Run G actually
  showing?"), we can re-parse the original log instead of
  re-running.
* The summary dirs (run_*_matrix_<date>) carry the orchestrator's
  metadata that the per-cycle dirs don't have.
* SUMMARY.md is the canonical pre-reorg index.

## Why not in git

Each cycle is ~ 1 GB (sglang log + harbor jobs + daemon log).
23 items × 1 GB ≈ 23 GB.  Cheap on disk, expensive in git.

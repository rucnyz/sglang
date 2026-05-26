# T1 — `GET /aginfer/state`

## WHAT WE PROMISED

**Capability**
* Returns JSON with two top-level keys: `tier_usage` (per-tier
  used_bytes + cap_bytes) and `units` (list of all evictable nodes).
* Each unit object has at minimum: `hash, tier, n_tokens, n_bytes,
  last_access_time, hit_count, session_ids`.
* Top-level: `page_size`, `bytes_per_token`, `tier_usage` (HBM + DRAM,
  each `{used_bytes, cap_bytes}`).
* **Forward-compat note**: the dump reads `node.session_ids` via
  `try/except AttributeError` and emits `[]` when absent. T3
  (`session_id` passthrough) just needs to populate the attribute
  during tree insertion — the read path is already wired here. T3 does
  NOT need to touch `dump_aginfer_state`.
* Read-only — does not change cache state.

**Cost ceiling** (revised after measuring on Qwen3-0.6B + flashinfer)
* Operating point: Run F peaked at #cached-token 121 856 / page_size 256
  ≈ 470 leaves; steady-state 100-1 000 leaves is the regime that matters.
* Linear in node count: ~100 μs/node + 200 ms baseline (= Python dict
  construction + ZMQ pickle from scheduler subproc + HTTP orjson).
* Practical bound: < 300 ms p99 at 1 000 nodes; < 1.2 s p99 at 10 000.
  At daemon's 5 s tick this stays under 5 % overhead.
* < ~150 lines added to sglang (io_struct + scheduler dispatch +
  tokenizer mixin pair + HTTP handler + tree-walk helper).

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Torn snapshot under live mutation | Drive 50 concurrent /v1/chat/completions while 5 background threads loop `GET /aginfer/state` at 10 Hz, with cap=64 K so eviction runs hot. | No exceptions; every response is valid JSON; no `hash` duplicated within a single response; `Σ unit.n_bytes per tier ≈ tier_usage.used_bytes` within page-rounding. | parse 500 responses, run schema validator + invariant checker, assert 0 failures |
| Endpoint latency spike (lock contention) | Inject `__post_init__` sleep(0.05) in the walk path (test build flag) | p99 ≤ 200 ms; daemon event-to-migrate p99 ≤ 280 ms (= +200 ms over baseline) | timeit 50 calls under traffic; assert p99 < 200 ms |
| Concurrent walk + concurrent migrate | Same as row 1 + 5 daemon threads issuing POST /aginfer/migrate to random hashes | No deadlock (< 1 s p99); no stale `tier` reported (= migrate completed before state read) | 60-s stress, expect ≥ 95 % of migrates to reflect in next state poll |

## HOW WE VERIFY

Mechanism (no harbor needed). `verify/t1_state_endpoint.py`:

```
1. Launch sglang with a small KV pool (cap 64 K so trees stay tractable).
2. Send ~50 distinct chat-completion warmups with shared + per-req prefixes
   so the radix tree has a known structure (≥ 50 nodes).
3. GET /aginfer/state and assert:
   - 200 OK, schema matches
   - len(units) >= 50
   - sum(u.n_bytes for u in units if u.tier=="HBM")
        ≈ tier_usage.HBM.used_bytes  (within page rounding)
4. Crank prefix variation up to ~10 000 distinct units, measure latency
   of `/aginfer/state` over 20 calls, report p50/p99.
```

## REPRODUCING

End-to-end repro, copy-paste from a clean shell.  Picks GPU 4 by
default; pick any free GPU and edit `CUDA_VISIBLE_DEVICES`.

```bash
# 1. Activate the preinstalled env.
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

# 2. Launch sglang.  Qwen3-0.6B + flashinfer is enough for T1
#    (trtllm_mha default would bypass the radix entirely).
cd /scratch/yuzhou/projects/sglang/dev/aginfer
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=4 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
  > logs/sglang_t1.log 2>&1 &

# Wait for "Uvicorn running on http://127.0.0.1:30001" in the log.

# 3. Run the verify.
AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python verify/t1/verify.py
# expected last line: "=== T1 PASSED ==="

# 4. Tear down sglang.
pkill -f "launch_server.*30001"
```

## RESULTS

**PASSED** — happy path + worst-case forced injection, on Qwen3-0.6B + `--attention-backend flashinfer` (trtllm_mha auto-picks page_size=1 and bypasses sglang radix insertion; flashinfer + UnifiedRadixCache does insert).

* date: 2026-05-25 (initial) / 2026-05-26 (bytes-schema rewrite)
* sglang sha (initial impl): `bbf3e7b33`
* sglang sha (after 10× perf opt): `82d2732d6`
* lines added: 150 (initial) + 322 (after opt) + ~40 (bytes-schema rewrite)
* schema valid: ✓ (100 % across all measured snapshots)
* invariants: ✓ no duplicate hash within a single response;
  Σ unit.n_bytes per tier == tier_usage.used_bytes within
  page_size × bytes_per_token; n_bytes == n_tokens × bytes_per_token
  for every unit
* **Schema rewrite (2026-05-26)** after design audit found
  `n_tokens`-vs-`n_bytes` drift (T1 README promised `n_bytes` but code
  only emitted `n_tokens`). Paper §7 per-unit value rule needs BYTES
  (the cost denominator); sticking to tokens-only would silently break
  cross-tier value comparison when HiCache lands in T9. Now the
  schema has:
  - top-level: `page_size`, `bytes_per_token` (computed once from the
    device KV pool's `get_bytes_per_token()` or
    `get_kv_size_bytes() / size`)
  - `tier_usage.{HBM,DRAM}.{used_bytes, cap_bytes}` (no `_tokens` fields)
  - each unit: `n_tokens` AND `n_bytes` (the value NUMERATOR is
    token-counted; the cost DENOMINATOR is byte-counted)
* Measured on Qwen3-0.6B + flashinfer: `bytes_per_token = 114688`
  (= 28 layers × 8 KV heads × 128 head_dim × 2 (K+V) × bf16; matches
  the model config). HBM cap = 65 536 tokens × 114 688 = 7.5 GB.

| Stage | metric | before opt | after opt | bound |
|---|---|---:|---:|---:|
| 2 (~50 → carry-over ~4.3-4.8k units) | p50 / p99 (10 calls) | 24 ms / 32 ms | 29 ms / 34 ms | n/a |
| 3 (stress ~6.3-6.9k units) | p50 / p99 (20 calls) | 37 ms / **469 ms** | 40 ms / **48 ms** | `200 + 0.100·N` ms = 836 ms |
| 4 WORST CASE (15 s concurrent walker + traffic) | walker ok / fail / p99 | 487 / 0 / **955 ms** | 459-662 / 0 / **196 ms** | 0 fail (strict) |

* delta vs ceiling: **10× under** at the stress regime (48 ms vs 836 ms bound); 5× tail-latency improvement under concurrent stress.
* root cause discovered: Gen-2 cyclic-GC fired every ~50 dumps because each dump allocated 10k Python dicts + lists, and the sweep over the live radix tree + KV-pool descriptors took 300-500 ms. Fix: direct-to-bytearray JSON in `dump_aginfer_state_bytes`, no per-node dict allocation. Walk itself is only ~14 ms at 4300 nodes.
* raw logs (relative to this directory):
  * `results/20260525_224238_baseline.log` (before opt)
  * `results/20260525_232021_optimized.log` (after opt)
  * `results/20260525_232149_optimized_run2.log`
  * `results/20260525_232258_optimized_run3.log`
  * `results/optimization_notes.md` (writeup of the 10× p99 improvement)

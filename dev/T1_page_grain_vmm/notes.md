# T1 notes

## A/B run on Qwen3.5-35B-A3B / H200 GPU 2

Both configs booted successfully (106 s each, identical within noise — cold
model load dominates the cuMemMap call-count delta) and serve a 5-prompt
smoke in ~1.3 s either way.

```
[page2MiB]   boot=106s smoke(5 prompts)=1.32s arena-init-lines=5
[chunk64MiB] boot=106s smoke(5 prompts)=1.35s arena-init-lines=5
```

## Arena state at boot (page-grain default vs legacy)

Pulled from server log `MultiTensorArena initialized: ...` and `* arena: ...`
lines:

| | 2 MiB pages | 64 MiB chunks |
|---|---:|---:|
| KV `chunk_bytes` | 2 097 152 | 67 108 864 |
| KV `tokens_per_chunk` | 2 048 | 65 536 |
| KV `boot_mapped` (chunks) | 617 | 20 |
| KV `max_tokens` | 1 271 808 | 1 572 864 |
| KV `static_min` (actuator floor) | 2 048 tokens | 65 536 tokens |
| Mamba `chunk_bytes` | 2 097 152 | 67 108 864 |
| Mamba `tokens_per_chunk` | 1 | 32 |
| Mamba `boot_mapped` (slots) | 362 | 12 |
| Mamba `max_tokens` | 366 | 512 |

## Findings

### 1. Mamba's `tokens_per_chunk = 1` at page-grain — perfect for T3/T4

A single mamba slot is $\sim 2$ MiB on Qwen3.5-35B-A3B (per layer × 30 layers).
At page-grain, this means **each mamba slot is its own VMM page**, and
`drain_ready` becomes "is this specific slot live?" rather than
"is *any* slot in this 32-slot batch live?". That's exactly the granularity
T3 (smart over-cap selection) needs for ~100% commit rate.

### 2. KV per-token bytes = 1024; 2048 tokens per page

Each KV page covers 2048 tokens at 1 KB/token (10 KV heads × 256 head_dim ×
2 bytes BF16 ÷ 5 layers). 617 pages × 2048 tokens = 1.27 M tokens of KV
capacity (vs 1.57 M at 64 MiB chunks). The drop is because the growth budget
is computed in chunks, not bytes:

```
max_tokens = tot_aligned + kv_growth_chunks * tokens_per_chunk
           = 1 263 616 + 4 * 2 048      = 1 271 808   (page)
           = 1 310 720 + 4 * 65 536     = 1 572 864   (chunk)
```

So `kv_growth_chunks = 4` gives 8 MiB of growth budget at page-grain vs
256 MiB at chunk-grain. **This is a known shortcoming that T5 (VA
overcommit at boot) explicitly addresses** — the growth budget will be
re-derived from a desired byte budget rather than a fixed chunk count.

### 3. No regressions on the smoke path

Both server logs show clean init, no `cuMemMap` errors, no stride/shape
warnings. 5 generate calls each return well-formed JSON with text. The
`tokens_per_chunk=1` on mamba doesn't break any kernel path on the smoke
prompts.

## Status

T1 done. Code change is two lines (memory_pool.py line 311 and 1188); A/B
captured; mamba pool naturally lands on the per-slot granularity that
unlocks T3 / T4. Only outstanding follow-up is the growth-budget shrinkage,
which T5 will resolve.

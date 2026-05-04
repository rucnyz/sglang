# T5 notes

## A/B on Qwen3.5-35B-A3B / H200 GPU 2

Boot 110 s, 5 generates clean. The arena init log shows the new
headroom in action:

```
KV    arena init: chunk_bytes=2097152 init_tokens=1263616  max_tokens=85149696
                                                            (headroom 83.9M tokens
                                                             = 80 GiB at 2 MiB/chunk
                                                             ÷ 2048 tokens/chunk)
mamba arena init: chunk_bytes=2097152 init_tokens=362      max_tokens=41322
                                                            (headroom 40 960 slots
                                                             = 80 GiB at 2 MiB/slot)
```

Comparison vs T1-only (8 MiB headroom = 4 chunks at 2 MiB):

| | T1 (4 chunks headroom) | T5 (80 GiB headroom) | factor |
|---|---|---|---|
| KV max_tokens | 1 271 808 | **85 149 696** | **67×** |
| KV growable bytes | 8 MiB | **80 GiB** | 10 240× |
| mamba max_slots | 366 | **41 322** | **113×** |
| mamba growable bytes | 8 MiB | 80 GiB | 10 240× |

VA reservation total per pool ≈ init bytes + 80 GiB headroom ≈ 87 GiB
each. Two pools combined VA ≈ 174 GiB > 143 GiB physical, which is the
overcommit invariant paper §3.2.1 promises. **No extra physical bytes
allocated** at boot — VA reservation is virtual, costs zero physical
HBM.

## What T5 makes possible

Before T5: actuator could grow either pool by at most 8 MiB per fire.
A swarm workload wanting to move 30 chunks (~7.5 GiB) from KV into
mamba would have only 8 MiB of headroom in mamba's VA — `cuMemMap`
would fail past slot 366.

After T5: same fire moves 30 chunks (~60 KB at 2 MiB chunks) into
mamba freely, and there's headroom for thousands more such fires.
The 80 GiB ceiling matches the ideal-mode "inter-pool can grow either
pool to ~half of total HBM" promise.

## What T5 does NOT verify

- **Actual fire** with non-trivial growth. The smoke just verifies the
  VA reservation succeeded and serving works; it doesn't fire the
  cross-pool actuator and grow either pool.
- **Memory pressure across the boot reservation**: VA reservation is
  virtual but the underlying CUDA driver does maintain a per-process
  page-table-entry budget. At 2 MiB pages over ~174 GiB VA = ~89 K
  PTEs per process. Production driver versions handle this fine; the
  smoke confirms boot doesn't trip a driver limit, but heavy multi-tenant
  setups may want to lower the env vars.
- **Whether 80 GiB is the right default**. For TP > 1 or smaller
  models, a smaller default (e.g., 30 GiB) might fit better. Tunable
  via `SGLANG_ARENA_KV_HEADROOM_BYTES` / `SGLANG_ARENA_MAMBA_HEADROOM_BYTES`.

## Status

T5 done. Two env vars added. Smoke confirms 67×/113× max_tokens
expansion. Real-fire validation: T7.

## Knob reference

```
# T5 (new): bytes-of-headroom (precedence over CHUNKS).
SGLANG_ARENA_KV_HEADROOM_BYTES=80GB    # default 80 GiB
SGLANG_ARENA_MAMBA_HEADROOM_BYTES=80GB # default 80 GiB

# Legacy: chunks-of-headroom (used only if BYTES unset).
SGLANG_ARENA_KV_HEADROOM_CHUNKS=N      # explicit override
SGLANG_ARENA_MAMBA_HEADROOM_CHUNKS=N   # explicit override

# Effective headroom (chunks):
#   if KV_HEADROOM_BYTES set:    bytes / chunk_size
#   elif KV_HEADROOM_CHUNKS set: that
#   else:                        80 GiB / chunk_size  (T5 default)
```

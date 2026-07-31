"""Unit test: MLATokenToKVPool arena branch (dev/interlayer/5_mla_arena).

Run on an idle GPU:
    CUDA_VISIBLE_DEVICES=<idle> .venv/bin/python \
        dev/interlayer/5_mla_arena/test/test_mla_arena_pool.py

Covers: arena gate honors SGLANG_KV_ARENA + SGLANG_ARENA_CHUNK_BYTES; buffer
shape/rows vs the torch.zeros twin; page-0 zeroing; write/read roundtrip;
per-token byte math for the Kimi-Linear dims (576 x bf16 = 1152 B, 18 MiB
chunk -> 16384 tokens/chunk).
"""
import os
import sys

sys.path.insert(0, "/data/yuzhou/projects/sglang-rebase/python")

KIMI_CHUNK = 9 * 2 * 1024 * 1024  # lcm(2 MiB, 1152 B) = 18 MiB

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'} {name}{(' — ' + detail) if detail and not cond else ''}")


def build_pool(size, layer_num, use_arena):
    import torch
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    if use_arena:
        os.environ["SGLANG_KV_ARENA"] = "1"
        os.environ["SGLANG_ARENA_CHUNK_BYTES"] = str(KIMI_CHUNK)
    else:
        os.environ.pop("SGLANG_KV_ARENA", None)
        os.environ.pop("SGLANG_ARENA_CHUNK_BYTES", None)
    os.environ.pop("SGLANG_ARENA_SHARED", None)
    return MLATokenToKVPool(
        size=size,
        page_size=1,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
    )


def main():
    import torch

    size, layers = 32768, 2

    zeros_pool = build_pool(size, layers, use_arena=False)
    check("twin(zeros): _kv_arena is None", zeros_pool._kv_arena is None)
    zshape = zeros_pool.kv_buffer[0].shape

    pool = build_pool(size, layers, use_arena=True)
    check("arena: _kv_arena set", pool._kv_arena is not None)
    a = pool._kv_arena
    check("arena: tokens_per_chunk exact", int(a.tokens_per_chunk) == 16384,
          f"got {a.tokens_per_chunk}")
    buf = pool.kv_buffer[0]
    check("arena: n buffers == layer_num", len(pool.kv_buffer) == layers)
    check("arena: per-token shape matches zeros twin",
          tuple(buf.shape[1:]) == tuple(zshape[1:]),
          f"{tuple(buf.shape[1:])} vs {tuple(zshape[1:])}")
    check("arena: rows cover size+page (chunk-aligned up)",
          buf.shape[0] >= zshape[0] and buf.shape[0] % 16384 == 0,
          f"rows={buf.shape[0]} zeros_rows={zshape[0]}")
    check("arena: dtype", buf.dtype == torch.bfloat16, str(buf.dtype))
    check("arena: page-0 zeroed", bool((buf[:1] == 0).all().item()))

    x = torch.randn(7, 1, 576, dtype=torch.bfloat16, device="cuda")
    pool.kv_buffer[1][100:107] = x
    torch.cuda.synchronize()
    check("arena: rw roundtrip layer1",
          bool(torch.equal(pool.kv_buffer[1][100:107], x)))
    check("arena: layer0 unaffected by layer1 write",
          bool((pool.kv_buffer[0][100:107] == 0).all().item()))

    ptr_before = pool.kv_buffer[0].data_ptr()
    check("arena: data_ptrs tensor matches buffers",
          int(pool.data_ptrs[0].item()) == ptr_before)

    # get_key_buffer / get_value_buffer views work on the arena tensor
    kb = pool.get_key_buffer(0)
    vb = pool.get_value_buffer(0)
    check("arena: get_value_buffer lora slice",
          vb.shape[-1] == 512 and kb.shape[-1] == 576)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())

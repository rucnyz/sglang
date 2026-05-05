"""
Bypass PyTorch's MemPool/CUDAPluggableAllocator machinery by
constructing tensors directly from cuMemMap'd VA via `at::from_blob`.
This avoids the +6-7% TTFT regression caused by:
  - PyTorch silently disabling expandable_segments when a user MemPool
    is active (CUDACachingAllocator.cpp:1587-1591).
  - Per-malloc mutex + hash-map work in CUDAPluggableAllocator
    (CUDAPluggableAllocator.cpp:83-93,124-140).

See https://github.com/pytorch/pytorch/issues/165419 for the upstream
discussion and vAttention (arXiv 2405.04437) for the canonical
reference using this pattern.

The arena owns the underlying VA's lifetime; we install a no-op
deleter on the storage. As long as the MultiTensorArena outlives any
tensor it produces (which it always does in the engine — pools live
for the entire serving session), this is safe.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Sequence

import torch


logger = logging.getLogger(__name__)


_EXT_LOCK = threading.Lock()
_EXT = None


_CPP_SRC = r"""
#include <torch/extension.h>
#include <stdexcept>

// Build a CUDA tensor that aliases the bytes at `va` with the given
// shape/dtype/device. PyTorch never frees this storage; arena owns it.
torch::Tensor tensor_from_va(
    int64_t va,
    std::vector<int64_t> sizes,
    std::string dtype_str,
    int device_index)
{
    torch::ScalarType st;
    if (dtype_str == "bfloat16")     st = torch::kBFloat16;
    else if (dtype_str == "float16") st = torch::kFloat16;
    else if (dtype_str == "float32") st = torch::kFloat32;
    else if (dtype_str == "uint8")   st = torch::kUInt8;
    else if (dtype_str == "int8")    st = torch::kInt8;
    else if (dtype_str == "int32")   st = torch::kInt32;
    else if (dtype_str == "int64")   st = torch::kInt64;
    else throw std::runtime_error("tensor_from_va: unsupported dtype " + dtype_str);

    auto options = torch::TensorOptions()
        .dtype(st)
        .device(torch::kCUDA, device_index);

    auto deleter = [](void* /*ptr*/) {
        // No-op; arena owns the VA lifetime. The lambda is captured
        // by reference by at::from_blob's storage; as long as the
        // arena keeps the VA mapped, the tensor remains valid.
    };

    return at::from_blob(
        reinterpret_cast<void*>(static_cast<uintptr_t>(va)),
        sizes,
        deleter,
        options
    );
}
"""


_DTYPE_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
}


def _get_ext():
    global _EXT
    with _EXT_LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load_inline
            verbose = os.environ.get("SGLANG_ARENA_FROM_BLOB_VERBOSE") == "1"
            logger.info(
                "from_blob_ext: JIT-compiling at::from_blob extension "
                "(first use; cached afterwards)"
            )
            _EXT = load_inline(
                name="sglang_arena_from_blob",
                cpp_sources=[_CPP_SRC],
                functions=["tensor_from_va"],
                verbose=verbose,
            )
            logger.info("from_blob_ext: extension ready")
        return _EXT


def tensor_from_va(
    va: int,
    sizes: Sequence[int],
    dtype: torch.dtype,
    device_index: int,
) -> torch.Tensor:
    """Build a CUDA tensor that aliases the bytes at `va` with the
    given shape/dtype, on `device_index`. Caller (the arena) is
    responsible for keeping the VA mapped at least as long as any
    referenced tensor is alive.

    Implementation: `at::from_blob` with a no-op deleter. The
    resulting tensor has `is_pinned()=False`, no caching-allocator
    tracking, no MemPool participation. Identical pattern to vAttention
    and PyTorch's own `torch.from_dlpack` route in spirit.
    """
    if dtype not in _DTYPE_STR:
        raise ValueError(
            f"from_blob_ext: dtype {dtype} not supported; add it to _DTYPE_STR"
        )
    return _get_ext().tensor_from_va(
        int(va),
        list(sizes),
        _DTYPE_STR[dtype],
        int(device_index),
    )

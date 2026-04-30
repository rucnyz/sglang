"""
Phase 2e.5.6 — Process-singleton SharedHandlePool helper.

When `SGLANG_ARENA_SHARED=1`, both `MHATokenToKVPool` and `MambaPool`
build their `MultiTensorArena`s on top of one shared `SharedHandlePool`
(see `chunk_arena.py`). This module provides the lazy getter; the first
arena to ask for it triggers creation, and subsequent arenas reuse it.

Pre-requisites that the engine code must satisfy before turning the
shared mode on:
  - Both pools must be on the same CUDA device.
  - Both pools must use the same chunk_bytes (currently 64 MiB everywhere).

Cross-arena transfer of physical handles is then enabled via the
`cross_arena_transfer(...)` free function in `chunk_arena.py`.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from sglang.srt.arena.chunk_arena import SharedHandlePool


logger = logging.getLogger(__name__)


_SINGLETON_LOCK = threading.Lock()
_SINGLETON: Optional[SharedHandlePool] = None
_SINGLETON_DEVICE: Optional[int] = None
_SINGLETON_CHUNK_BYTES: Optional[int] = None


def is_shared_arena_enabled() -> bool:
    """Return True iff SGLANG_ARENA_SHARED=1 is set in the environment.

    Implies SGLANG_KV_ARENA=1 and SGLANG_MAMBA_ARENA=1; the engine pools
    consult both flags but the shared-pool path requires this top-level
    flag to actually get a shared handle pool.
    """
    return os.environ.get("SGLANG_ARENA_SHARED") == "1"


def get_or_create_shared_handle_pool(
    device_id: int,
    chunk_bytes: int,
) -> SharedHandlePool:
    """Return the process-singleton `SharedHandlePool`.

    Lazily created on first call. Subsequent calls validate that the
    requested config matches; mismatches raise (this is a process-wide
    invariant, not a per-pool override).
    """
    global _SINGLETON, _SINGLETON_DEVICE, _SINGLETON_CHUNK_BYTES
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = SharedHandlePool(
                device_id=device_id,
                chunk_size=chunk_bytes,
                n_handles=0,  # arenas grow this incrementally as they init
            )
            _SINGLETON_DEVICE = device_id
            _SINGLETON_CHUNK_BYTES = chunk_bytes
            logger.info(
                "Arena shared mode: created process-singleton SharedHandlePool "
                "device=%d, chunk_bytes=%d",
                device_id, chunk_bytes,
            )
        else:
            if _SINGLETON_DEVICE != device_id:
                raise RuntimeError(
                    f"shared SharedHandlePool already created on device "
                    f"{_SINGLETON_DEVICE}, can't add device {device_id}"
                )
            if _SINGLETON_CHUNK_BYTES != chunk_bytes:
                raise RuntimeError(
                    f"shared SharedHandlePool chunk_bytes={_SINGLETON_CHUNK_BYTES}, "
                    f"caller wants {chunk_bytes}"
                )
        return _SINGLETON


def reset_shared_handle_pool_for_test() -> None:
    """Test-only: clear the singleton so a unit test can re-create it.

    Note: does NOT release any held cuMemCreate handles; just drops the
    Python reference. Use only at process boundaries (e.g., between
    independent test functions).
    """
    global _SINGLETON, _SINGLETON_DEVICE, _SINGLETON_CHUNK_BYTES
    with _SINGLETON_LOCK:
        _SINGLETON = None
        _SINGLETON_DEVICE = None
        _SINGLETON_CHUNK_BYTES = None

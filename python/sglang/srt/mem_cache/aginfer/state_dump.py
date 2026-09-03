"""aginfer state-dump subsystem (refactor #251): the DESIGN §5 snapshot
serializer + program-state / runtime-metrics storage as free functions over a
UnifiedRadixCache.  The upstream cache keeps thin delegators; the fat dump logic
lives here (thin hook in core, fat self-contained module).

Behaviour is byte-for-byte identical to the previously-inline methods: this is a
pure mechanical move (``self`` → ``cache``), no logic change.  Instance fields
(``cache._aginfer_program_states`` / ``_aginfer_runtime_metrics`` /
``_aginfer_state_dump_metrics`` / ``_aginfer_hints`` / ``_aginfer_bpt_cache``)
are still initialised in the cache class; the functions here only read/write them.
"""
from __future__ import annotations

import shutil
import time
from typing import Optional

from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    peek_time_counter,
)

# Valid aginfer program states (DESIGN §6 round-6 H2).  Was the
# ``_AGINFER_VALID_STATES`` class attribute on UnifiedRadixCache; lives here now
# that ``set_aginfer_program_state`` does (the cache keeps a thin delegator).
_AGINFER_VALID_STATES = ("REASONING", "ACTING", "PAUSED", "ENDED")


class _StateDumpMetrics:
    """PLAN T14 — bounded ring buffer of recent ``_dump_aginfer_state_impl``
    latencies + emitted-byte counts.  Piggybacked into ``/aginfer/state``
    under the ``state_dump_metrics`` top-level key so monitoring scripts
    (and PLAN §2's "p99 > 50 ms → F3-revisit trigger") can poll on the
    same hot path the daemon already uses.

    Single-threaded by construction: only the scheduler process'
    ``_dump_aginfer_state_impl`` calls into this class, and the
    scheduler serialises requests on the event loop.  No lock.
    """

    __slots__ = (
        "_capacity", "_samples", "_first_recorded_perf_ns",
        "_total_count",
    )

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity = int(capacity)
        # (elapsed_ns, dump_bytes); dump_bytes == -1 for the dict path
        # (we never measure serialised size there).
        self._samples: list[tuple[int, int]] = []
        self._first_recorded_perf_ns: Optional[int] = None
        self._total_count = 0

    def record(self, elapsed_ns: int, dump_bytes: int) -> None:
        if self._first_recorded_perf_ns is None:
            self._first_recorded_perf_ns = time.perf_counter_ns()
        self._samples.append((int(elapsed_ns), int(dump_bytes)))
        if len(self._samples) > self._capacity:
            # Pop from the front; cheap at our sizes (1k entries, O(N)
            # once per recorded sample after wrap).  Deque would be O(1)
            # but disallows random indexing for quantile sort.
            del self._samples[0]
        self._total_count += 1

    def summary(self) -> dict:
        """Snapshot the contract field set the verify/t14 probe asserts.

        Cold start (n=0): all numeric quantiles report 0.0; the
        sentinel last_dump_bytes=-1 differentiates 'no dump yet'
        from 'dump path saw bytes=0'.
        """
        n = len(self._samples)
        if n == 0:
            return {
                "n_samples": 0,
                "n_recorded_total": 0,
                "capacity": self._capacity,
                "window_seconds": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "max_ms": 0.0,
                "mean_ms": 0.0,
                "last_dump_ms": 0.0,
                "last_dump_bytes": -1,
            }
        times = sorted(s[0] for s in self._samples)

        def _q(p: float) -> float:
            if n == 1:
                return times[0] / 1e6
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return times[idx] / 1e6

        last_ns, last_bytes = self._samples[-1]
        now_ns = time.perf_counter_ns()
        window_s = (
            (now_ns - self._first_recorded_perf_ns) / 1e9
            if self._first_recorded_perf_ns is not None else 0.0
        )
        return {
            "n_samples": n,
            "n_recorded_total": self._total_count,
            "capacity": self._capacity,
            "window_seconds": window_s,
            "p50_ms": _q(0.50),
            "p95_ms": _q(0.95),
            "p99_ms": _q(0.99),
            "max_ms": times[-1] / 1e6,
            "mean_ms": sum(times) / n / 1e6,
            "last_dump_ms": last_ns / 1e6,
            "last_dump_bytes": int(last_bytes),
        }


def set_aginfer_program_state(
    cache,
    *,
    pid: str,
    state: str,
    pre_pause_state: "Optional[str]",
) -> tuple:
    """T21 (#181, DESIGN §6 round-6 H2): daemon → sglang PUT
    ``/aginfer/program_paused`` storage.

    Returns ``(ok: bool, reason: str, applied: int)`` where
    ``applied == 0`` if the (state, pre_pause_state) for this
    pid was already at the requested value (idempotent re-apply
    per DESIGN §10 R2).

    ``state`` must be one of ``{REASONING, ACTING, PAUSED, ENDED}``.
    ``pre_pause_state`` is the same set or None.
    """
    valid = _AGINFER_VALID_STATES
    if not isinstance(pid, str) or not pid:
        return (False, f"pid must be non-empty string; got {pid!r}", 0)
    if state not in valid:
        return (
            False,
            f"state must be in {valid}; got {state!r}",
            0,
        )
    if pre_pause_state is not None and pre_pause_state not in valid:
        return (
            False,
            f"pre_pause_state must be None or in {valid}; "
            f"got {pre_pause_state!r}",
            0,
        )
    existing = cache._aginfer_program_states.get(pid)
    new_entry = {"state": state, "pre_pause_state": pre_pause_state}
    if existing is not None and existing == new_entry:
        return (True, "ok", 0)  # idempotent no-op
    cache._aginfer_program_states[pid] = new_entry
    return (True, "ok", 1)


def _aginfer_overlay_program_states(cache, per_program: dict) -> dict:
    """T21 (#181): overlay daemon-pushed program states onto the
    unit-walk-derived ``per_program`` dict, IN PLACE.

    Shared by ``_dump_aginfer_state_dict`` and
    ``_dump_aginfer_state_bytes`` so the two dump paths cannot
    diverge by construction (the #181 audit flagged that the
    previous duplicated loops were a divergence risk + that the
    verify only tested a hand-copied replica).

    Two jobs:
      1. **Overlay**: for every stored pid, set the dump entry's
         ``state`` / ``pre_pause_state`` to the daemon's view.
         Programs with no live units still get an (empty-residue)
         entry so the daemon can read its own PAUSED-with-no-
         residue bookkeeping.
      2. **Lazy GC** (#181 audit — unbounded-growth fix): an
         ENDED program with NO live units needs no daemon
         bookkeeping; drop it from ``_aginfer_program_states``
         instead of echoing it forever.  Without this, every
         program ever PUT would accumulate and pollute every
         subsequent dump's ``per_program_usage``.  ENDED programs
         that still have residual units ARE echoed (the daemon
         needs to see the terminal state while cleanup completes).
    """
    unit_pids = set(per_program.keys())  # pids with live units
    ended_no_units: list = []
    for pid, stored in cache._aginfer_program_states.items():
        if stored["state"] == "ENDED" and pid not in unit_pids:
            ended_no_units.append(pid)
            continue
        e = per_program.setdefault(pid, {
            "hbm":  {"committed": {}, "inflight": {}},
            "dram": {"committed": {}},
            "state": "REASONING",
            "pre_pause_state": None,
            "unit_hashes": [],
        })
        e["state"] = stored["state"]
        e["pre_pause_state"] = stored["pre_pause_state"]
    ended_gcd = set(ended_no_units)
    for pid in ended_no_units:
        del cache._aginfer_program_states[pid]
    # T26 (#200): overlay scheduler-pushed per-program in-flight HBM
    # bytes.  A program can have running (in-flight) bytes with NO
    # radix-committed units yet (first request still prefilling /
    # decoding), so create an entry if absent — its in-flight bytes
    # are the §8 snapshot_relief the daemon would free by pausing it.
    # Skip pids the ENDED-GC just dropped (#200 audit): an ENDED
    # program with a still-draining running req must NOT be resurrected
    # here as REASONING (the GC is authoritative about termination).
    # #217: default via getattr — runtime metrics (T26/#200) may not be
    # pushed yet (cold start before the scheduler's first
    # set_aginfer_runtime_metrics), and the overlay must still build a
    # valid program-state map rather than AttributeError.
    for pid, sp_bytes in getattr(
        cache, "_aginfer_runtime_metrics", {}
    ).get("inflight", {}).items():
        if pid in ended_gcd:
            continue
        e = per_program.setdefault(pid, {
            "hbm":  {"committed": {}, "inflight": {}},
            "dram": {"committed": {}},
            "state": "REASONING",
            "pre_pause_state": None,
            "unit_hashes": [],
        })
        e["hbm"]["inflight"] = {sp: int(b) for sp, b in sp_bytes.items()}
    return per_program


def _aginfer_node_summary(cache, node) -> dict:
    """T24 (#182): compact summary of a radix-tree node for the
    HASH_COLLISION webhook payload.  Daemon-side fatal()
    consumes this to identify which two nodes collided.

    Keep cheap — this runs inside the apply_aginfer_migrations
    DFS.  Hash collisions are < 10⁻²² probable, so we never
    amortise; pay a bit of Python here only when one actually
    happens.
    """
    # Residence: which tiers currently hold this node's KV.
    residence: list[str] = []
    cd = node.component_data[0] if node.component_data else None
    if cd is not None:
        if getattr(cd, "value", None) is not None:
            residence.append("HBM")
        if getattr(cd, "host_value", None) is not None:
            residence.append("DRAM")
    # n_tokens on whichever layer holds it.
    n_tokens = 0
    if cd is not None:
        if cd.value is not None:
            n_tokens = max(n_tokens, len(cd.value))
        if getattr(cd, "host_value", None) is not None:
            n_tokens = max(n_tokens, len(cd.host_value))
    # Hex hash-value (post-compute_node_hash_values) iff present.
    hv = node.hash_value[-1] if getattr(node, "hash_value", None) else None
    sids = []
    try:
        sids = list(node.session_ids)
    except (AttributeError, TypeError):
        pass
    return {
        "node_id": int(node.id),
        "hash_value": str(hv) if hv is not None else None,
        "residence": residence,
        "n_tokens": int(n_tokens),
        "session_ids": [str(s) for s in sids][:8],
        "hit_count": int(getattr(node, "hit_count", 0) or 0),
    }


# ---- aginfer daemon snapshot (paper §3 state s_t) ----
def _aginfer_bytes_per_token(cache) -> int:
    """Best-effort device-KV bytes-per-token.

    Used to convert n_tokens -> n_bytes in /aginfer/state.  The daemon's
    per-unit value rule (paper §7) compares units by value-per-byte, so
    we need a precise byte count.  Per-pool API:

      * DSV4 / HiSparse: ``KVCache.get_bytes_per_token()`` (precise)
      * MHA / MLA       : derive from ``get_kv_size_bytes() / size``
      * unknown         : return 0 — the daemon falls back to tokens

    Only a POSITIVE result is cached — the KV layout never changes once
    known, but an early/transient 0 (kvcache not yet wired when a hot-
    path caller asks) must NOT poison the cache.  Before #200/#206 the
    first caller was always the cold dump (kvcache ready → bpt>0); the
    T26 hot-path hooks now call this much earlier, and caching their
    transient 0 zeroed every pool_usage cap_bytes → occ_hbm≡0 → the
    daemon never saw HBM pressure (#209).
    """
    cached = getattr(cache, "_aginfer_bpt_cache", 0)
    if cached:
        return cached
    pool_alloc = cache.token_to_kv_pool_allocator
    kv = None
    if pool_alloc is not None:
        kv = getattr(pool_alloc, "_kvcache", None)
        if kv is None and hasattr(pool_alloc, "get_kvcache"):
            try:
                kv = pool_alloc.get_kvcache()
            except Exception:
                kv = None
    bpt = 0
    if kv is not None:
        if hasattr(kv, "get_bytes_per_token"):
            try:
                bpt = int(kv.get_bytes_per_token())
            except Exception:
                bpt = 0
        if bpt == 0 and hasattr(kv, "get_kv_size_bytes"):
            try:
                k_size, v_size = kv.get_kv_size_bytes()
                total_bytes = int(k_size) + int(v_size)
                size = int(getattr(kv, "size", 0) or 0)
                if size > 0:
                    bpt = total_bytes // size
            except Exception:
                bpt = 0
        if bpt == 0:
            # SWA / multi-component pools (e.g. DeepSeekV4TokenToKVPool, a
            # BaseSWAKVPool wrapper) do NOT expose get_bytes_per_token on
            # the wrapper — it lives on the per-component single pools it
            # holds (swa_kv_pool / c128_kv_pool, each a
            # DeepSeekV4SingleKVPool).  Probe them so DSV4's byte
            # accounting isn't 0 → occ_hbm≡0 → daemon-blind (#209).
            for _attr in ("full_kv_pool", "swa_kv_pool", "c128_kv_pool",
                          "kv_pool", "_pool"):
                _sub = getattr(kv, _attr, None)
                if _sub is not None and hasattr(_sub, "get_bytes_per_token"):
                    try:
                        bpt = int(_sub.get_bytes_per_token())
                    except Exception:
                        bpt = 0
                    if bpt > 0:
                        break
    if bpt > 0:  # never cache a transient 0 (#209) — recompute next call
        cache._aginfer_bpt_cache = bpt
    return bpt


def _aginfer_subpool_name(cache, ct) -> str:
    """Map a sglang ComponentType to its DESIGN §5 subpool name.

    The aginfer wire format uses architecture-determined component
    names; sglang's ``ComponentType`` enum is the authoritative
    source of which subpools a model carries.  Names are
    lower-case strings (DESIGN §12 examples: "full", "swa",
    "mamba", "draft", "attn").  We use sglang's own internal
    names verbatim so the daemon ↔ scheduler vocabularies stay
    identical and a typo on either side fails loudly.
    """
    return str(ct).rsplit(".", 1)[-1].lower()


def _aginfer_decode_bytes_per_token(cache, ct) -> int:
    """Per-token HBM byte growth during DECODE for a subpool whose
    component is ``ct`` (DESIGN §8 ``bytes_per_token_in_subpool``).

    Attention components (full / SWA) grow monotonically: each
    decoded token appends one token's worth of KV, so the per-token
    decode growth is ``_aginfer_bytes_per_token()``.  Mamba state is
    allocated ONCE per sequence at a snapshot boundary, NOT per
    decoded token, so its in-flight decode growth is 0 — the daemon's
    ``forecast_inflight_demand`` must not project Mamba growth from
    decode throughput (it would over-forecast and over-pause).  This
    is the only piece of the §8 forecast trajectory term the daemon
    cannot derive itself (decode throughput + E[remaining_tokens] are
    T26/T11); exposing it here lets the daemon assemble the product
    once those measurements land (#199).
    """
    if ct == ComponentType.MAMBA:
        return 0
    return _aginfer_bytes_per_token(cache)


def _aginfer_pool_usage(cache) -> dict:
    """Per-(tier, subpool) allocator-truth occupancy (DESIGN §5).

    Shape::

        {"HBM":  {"subpools": {sp: {used, cap, available,
                                    evictable, page_bytes}}},
         "DRAM": {"subpools": {...}},
         "DISK": {"subpools": {...}}}

    Each tier carries a ``subpools`` dict keyed by architecture-
    determined component name.  For S1 (single-stack attention,
    e.g. DeepSeek-V4-Flash) the HBM dict has exactly one entry
    keyed by the FULL component's name.  For S2 (SWA-hybrid) the
    dict has ``{"full", "swa"}``; for S3 (Mamba+attn) it has
    ``{"full", "mamba"}``.

    This is the AUTHORITATIVE pool-occupancy signal the daemon's
    admission controller and §7 value rule both consume.  The
    radix-tree-keyed sum of ``units[*].n_bytes[tier][sp]`` is a
    SUBSET of the allocator total (in-flight decode bytes are
    not in the tree), so admission must gate on this, not on
    the radix view.
    """
    pool = cache.token_to_kv_pool_allocator
    bpt = _aginfer_bytes_per_token(cache)
    page_bytes_default = int(cache.page_size) * max(1, bpt)
    # DESIGN §8 bytes_per_token_in_subpool (#199): per-token DECODE
    # growth, by component.  Attention (full / SWA) = bpt; Mamba = 0
    # (allocated per-sequence, not per-token).  DRAM/DISK carry bpt
    # for schema-key uniformity but the daemon only reads HBM's.
    dbpt_full = _aginfer_decode_bytes_per_token(cache, BASE_COMPONENT_TYPE)
    dbpt_swa = _aginfer_decode_bytes_per_token(cache, ComponentType.SWA)

    hbm_subpools: dict = {}
    dram_subpools: dict = {}

    # Determine the FULL subpool name for this cache instance.
    # When the cache supports SWA, ``pool`` exposes both
    # full_available_size() and swa_available_size(); otherwise
    # only available_size() and size.
    #
    # Semantics: ``used_bytes`` = total allocator-occupied bytes
    # (= cap − available; includes evictable radix-resident bytes).
    # ``evictable_bytes`` reports how much of `used` could be freed
    # if pressure demands it.  Admission's `theta_hi` gates on
    # `used / cap` per subpool so radix-eviction pressure registers;
    # legacy `pool_size − avail − evictable` (in-flight only) under-
    # reports occupancy when the radix tree fills the device.
    full_sp = _aginfer_subpool_name(cache, BASE_COMPONENT_TYPE)
    if pool is not None:
        is_swa = cache.supports_swa() and hasattr(pool, "full_available_size")
        if is_swa:
            full_size = int(getattr(pool, "size_full", 0)) or int(
                getattr(pool, "size", 0)
            )
            full_avail = int(pool.full_available_size())
            full_evictable = int(cache.full_evictable_size())
            full_used = max(0, full_size - full_avail)
            hbm_subpools[full_sp] = {
                "used_bytes": full_used * bpt,
                "cap_bytes": full_size * bpt,
                "available_bytes": full_avail * bpt,
                "evictable_bytes": full_evictable * bpt,
                "page_bytes": page_bytes_default,
                "decode_bytes_per_token": dbpt_full,
            }
            swa_sp = _aginfer_subpool_name(cache, ComponentType.SWA)
            swa_size = int(getattr(pool, "size_swa", 0))
            swa_avail = int(pool.swa_available_size())
            swa_evictable = int(cache.swa_evictable_size())
            swa_used = max(0, swa_size - swa_avail)
            hbm_subpools[swa_sp] = {
                "used_bytes": swa_used * bpt,
                "cap_bytes": swa_size * bpt,
                "available_bytes": swa_avail * bpt,
                "evictable_bytes": swa_evictable * bpt,
                "page_bytes": page_bytes_default,
                "decode_bytes_per_token": dbpt_swa,
            }
        else:
            pool_size = int(getattr(pool, "size", 0))
            avail = int(pool.available_size())
            evictable = int(cache.evictable_size())
            num_used = max(0, pool_size - avail)
            hbm_subpools[full_sp] = {
                "used_bytes": num_used * bpt,
                "cap_bytes": pool_size * bpt,
                "available_bytes": avail * bpt,
                "evictable_bytes": evictable * bpt,
                "page_bytes": page_bytes_default,
                "decode_bytes_per_token": dbpt_full,
            }
    else:
        hbm_subpools[full_sp] = {
            "used_bytes": 0, "cap_bytes": 0,
            "available_bytes": 0, "evictable_bytes": 0,
            "page_bytes": page_bytes_default,
            "decode_bytes_per_token": dbpt_full,
        }

    # DRAM (HiCache host pool).  Treated as a single subpool
    # keyed by the FULL component name — sglang does not split the
    # host pool by sub-pool today (SWA/Mamba KV write through to
    # the same host pool).  Subpool refinement is T26-future work.
    dram_cap = 0
    if cache.cache_controller is not None:
        # The HiCache host pool lives at cache_controller.mem_pool_host
        # (HostKVCache; cf. available_size() use elsewhere).  Earlier this
        # probed host_mem_pool / host_pool — neither exists, so DRAM
        # cap_bytes was always 0, making the daemon read DRAM as full and
        # demote value-evicted units to DROP (recompute) instead of DRAM
        # (cheap reload) — a do-no-harm regression.  ``* bpt`` (device
        # bytes/token) keeps cap consistent with the DRAM used_bytes the
        # dump walker patches in with the same bpt.
        host_pool = getattr(cache.cache_controller, "mem_pool_host", None)
        if host_pool is not None:
            dram_cap = int(getattr(host_pool, "size", 0)) * bpt
    # DRAM used bytes are filled in by the dump walker; pool_usage
    # itself doesn't have a fast aggregate.  Caller patches in.
    dram_subpools[full_sp] = {
        "used_bytes": 0,           # patched in by dump_aginfer_state_impl
        "cap_bytes": dram_cap,
        "available_bytes": dram_cap,  # patched
        "evictable_bytes": 0,         # patched
        "page_bytes": page_bytes_default,
        "decode_bytes_per_token": dbpt_full,  # schema uniformity; HBM-only signal
    }

    # DISK — P5 safe-subset: "add DISK" now really writes to sglang's
    # storage backend (write_backup_storage, cache_hooks.
    # apply_aginfer_migrations), but there is still no cross-backend
    # capacity API (HiCacheStorage.get_stats() returns None by default)
    # and no delete path, so we can only report a REAL cap_bytes for a
    # backend whose shape we recognise.  Today that's HiCacheFile (local
    # filesystem spill dir) via shutil.disk_usage() on its file_path;
    # nixl/mooncake and any other backend fall back to the 0/0 placeholder
    # (means "unknown", NOT "zero capacity") until they grow their own
    # accessor.  evictable_bytes stays 0 regardless — with no delete API
    # upstream, aginfer cannot reclaim DISK bytes even in principle.
    disk_cap = 0
    disk_avail = 0
    storage_backend = (
        getattr(cache.cache_controller, "storage_backend", None)
        if cache.cache_controller is not None
        else None
    )
    if storage_backend is not None:
        file_path = getattr(storage_backend, "file_path", None)
        if file_path is not None:
            try:
                usage = shutil.disk_usage(file_path)
            except OSError:
                pass
            else:
                disk_cap = int(usage.total)
                disk_avail = int(usage.free)
    disk_subpools = {full_sp: {
        "used_bytes": max(0, disk_cap - disk_avail),
        "cap_bytes": disk_cap,
        "available_bytes": disk_avail,
        "evictable_bytes": 0,
        "page_bytes": page_bytes_default,
        "decode_bytes_per_token": dbpt_full,  # schema uniformity; HBM-only signal
    }}

    # HBM occupancy signal the T5 watermark webhook fires on
    # (scheduler.maybe_fire reads pool_usage["HBM"]["token_usage"]).
    # Max across subpools = the bottleneck pool (the small SWA pool fills
    # first), matching the daemon's occ_hbm so sglang and daemon never
    # disagree about pressure.  WITHOUT this key the webhook's
    # .get("token_usage", 0.0) was always 0.0 -> memory_pressure NEVER
    # fired -> the daemon never woke under radix-cache pressure.  HBM
    # used_bytes here is allocator-truth (cap-available); DRAM/DISK
    # used_bytes are patched later so their token_usage is omitted.
    hbm_token_usage = 0.0
    for sp in hbm_subpools.values():
        cap = sp.get("cap_bytes", 0)
        if cap > 0:
            hbm_token_usage = max(hbm_token_usage, sp["used_bytes"] / cap)

    return {
        "HBM":  {"subpools": hbm_subpools, "token_usage": hbm_token_usage},
        "DRAM": {"subpools": dram_subpools},
        "DISK": {"subpools": disk_subpools},
    }


def _aginfer_link_stats(cache) -> dict:
    """Cold-start link_stats; T26 fills `recent_throughput_bps` /
    `time_since_last_sample_s` via HiCache + Mooncake instrumentation.

    Daemon's §7 bw_free branches on
    ``time_since_last_sample_s > LINK_IDLE_SECONDS`` to choose
    peak vs measured throughput; ``+Inf`` on cold-start so the
    peak path is taken (correct — an unused link IS idle by
    definition).

    Peak values are realistic defaults for a B300 box with PCIe
    gen5 x16 + NVMe; T26 calibration replaces them with the
    operator-provided / device-probed numbers.

    ``time_since_last_sample_s`` uses 1e9 (≈ 31 years) as the
    cold-start sentinel instead of ``math.inf`` because orjson
    rejects non-finite floats as invalid JSON.  Daemon's bw_free
    branch is ``> LINK_IDLE_SECONDS = 1.0`` so any value above
    the threshold takes the peak path.
    """
    PEAK_HBM_DRAM = 64 * 1024 * 1024 * 1024 * 8   # ~64 GB/s PCIe 5.0 x16
    PEAK_DRAM_DISK = 12 * 1024 * 1024 * 1024 * 8  # ~12 GB/s NVMe
    return {
        "HBM->DRAM": {"peak_bw_bps": PEAK_HBM_DRAM,
                      "recent_throughput_bps": 0,
                      "time_since_last_sample_s": 1.0e12},
        "DRAM->HBM": {"peak_bw_bps": PEAK_HBM_DRAM,
                      "recent_throughput_bps": 0,
                      "time_since_last_sample_s": 1.0e12},
        "DRAM->DISK": {"peak_bw_bps": PEAK_DRAM_DISK,
                       "recent_throughput_bps": 0,
                       "time_since_last_sample_s": 1.0e12},
        "DISK->DRAM": {"peak_bw_bps": PEAK_DRAM_DISK,
                       "recent_throughput_bps": 0,
                       "time_since_last_sample_s": 1.0e12},
    }


def _aginfer_tier_holding_cost(cache, pool_usage: dict) -> dict:
    """Per-(tier, subpool) h_max_per_byte_sec placeholder.

    T12 calibrates the shape; T17 just exposes the field with
    a static placeholder per (tier, subpool) declared in
    pool_usage.  Subpool key set matches pool_usage by
    construction.
    """
    # Placeholder: linear holding cost per byte per second.
    # T12 calibration replaces these.
    H = 0.0
    out: dict = {}
    for tier, entry in pool_usage.items():
        out[tier] = {
            sp: {"h_max_per_byte_sec": H}
            for sp in entry["subpools"].keys()
        }
    return out


def set_aginfer_runtime_metrics(
    cache, *, decode_per_program: dict, prefill_bps: float, inflight: dict
) -> None:
    """T26 (#200): scheduler pushes its measured throughput EMAs +
    per-program in-flight bytes here before each dump (the scheduler
    owns the running batch + forward timing; the cache only stores).
    ``decode_per_program`` = {pid: tokens/sec}; ``prefill_bps`` =
    bytes/sec; ``inflight`` = {pid: {subpool: HBM bytes}}."""
    cache._aginfer_runtime_metrics = {
        "decode_per_program": dict(decode_per_program or {}),
        "prefill_bps": float(prefill_bps or 0.0),
        "inflight": {pid: dict(sp) for pid, sp in (inflight or {}).items()},
    }


def _aginfer_throughput_ema(cache) -> dict:
    """``throughput_ema`` for the dump — the scheduler-pushed EMAs
    (T26 #200; was a hardcoded 0.0/{} placeholder).

    Daemon's §8 ``marginal_pause_cost`` (prefill_bps) and
    ``forecast_inflight_demand`` (decode_per_program) read these.
    Empty until the scheduler pushes a measurement, so the formulas
    still degenerate to their no-signal branches at cold-start.
    """
    m = getattr(cache, "_aginfer_runtime_metrics", {})   # #217: cold-start safe
    return {
        "prefill_bps": float(m.get("prefill_bps", 0.0)),
        "decode_per_program": dict(m.get("decode_per_program", {})),
    }


def dump_aginfer_state(cache) -> dict:
    """Walk the radix tree once and return the DESIGN §5 snapshot
    for the aginfer external scheduler.  Read-only, no locks held.

    Top-level shape (full schema in `dev/aginfer/DESIGN.md` §5)::

        {
          "time_counter":      int,
          "throughput_ema":    {prefill_bps, decode_per_program},
          "pool_usage":        {tier: {subpools: {sp: {...}}}},
          "per_program_usage": {pid: {hbm, dram, state,
                                      pre_pause_state, unit_hashes}},
          "units":             [{hash, residence: [tier, ...],
                                 n_tokens,
                                 n_bytes: {tier: {sp: int}},
                                 last_access_time, hit_count,
                                 session_ids}],
          "link_stats":        {"σ->τ": {...}},
          "tier_holding_cost": {tier: {sp: {h_max_per_byte_sec}}},
        }

    Residence is a SET per unit (a post-write_through unit lives in
    both HBM and DRAM simultaneously); n_bytes is nested per
    (tier, subpool) to feed the §9 multi-axis DP.

    Subpool keys are derived from the cache's component types
    (``ComponentType.FULL`` → "full", etc.).  S1 single-stack
    attention has exactly one subpool; SWA-hybrid / Mamba-hybrid
    / spec-decoding add more (DESIGN §12).

    ``link_stats`` / ``tier_holding_cost`` / ``throughput_ema``
    are emitted with cold-start defaults; T26 / T29 wire actual
    instrumentation.  See the per-helper docstrings.
    """
    return _dump_aginfer_state_impl(cache, want_bytes=False)


def dump_aginfer_state_bytes(cache) -> str:
    """Same snapshot as :meth:`dump_aginfer_state`, pre-serialised to a
    JSON **string** inside the scheduler process.

    Returns ``str`` (JSON text), not ``bytes``, so the one serialised
    payload traverses BOTH transports unchanged:
      * the native HTTP ``/aginfer/state`` route (``orjson.loads`` and
        Starlette ``Response`` both accept ``str``), and
      * Dynamo's ``call_tokenizer_manager`` passthrough, whose Rust JSON
        serializer rejects a Python ``bytes`` payload ("invalid type:
        byte array") but carries a ``str`` fine.

    The single-serialise win is unchanged: one cache walk -> one
    ``orjson.dumps`` (no 10k-element list-of-dicts pickled across the
    ZMQ control channel); only the final type is decoded to ``str``.
    The wire format (the JSON the daemon parses) is identical.
    """
    return _dump_aginfer_state_impl(cache, want_bytes=True).decode("utf-8")


def _dump_aginfer_state_impl(cache, want_bytes: bool):
    """Build the DESIGN §5 snapshot.

    Two paths to keep per-dump GC pressure bounded:

    * ``want_bytes=True`` (HTTP hot path): single walk that writes
      units directly into a ``bytearray`` and aggregates
      per_program_usage / DRAM-used into small dicts.  No per-
      node Python dict allocated for the units list, so a 10 k-node
      dump does not trip the scheduler's Gen-2 cyclic GC sweep
      (empirically ~500 ms stall — the dominant p99 tail).
      Final assembly orjson-encodes the small auxiliary dicts
      once.

    * ``want_bytes=False`` (in-process callers / tests): same walk
      but builds Python dicts per unit.  Convenient, not allocation-
      bounded; not on the HTTP hot path.

    T14: wraps the inner build with a ``perf_counter_ns`` so each
    call lands a (elapsed, dump_bytes) sample in the ring buffer.
    The summary embedded INTO this dump is from samples PRIOR to
    this call (chicken-and-egg: we can't include our own latency
    before we've finished measuring it).  Each /aginfer/state poll
    therefore advances ``n_recorded_total`` by exactly 1.
    """
    t0 = time.perf_counter_ns()
    bytes_per_token = _aginfer_bytes_per_token(cache)
    sp_full = _aginfer_subpool_name(cache, BASE_COMPONENT_TYPE)
    metrics_summary = cache._aginfer_state_dump_metrics.summary()
    if want_bytes:
        result = _dump_aginfer_state_bytes(
            cache, bytes_per_token, sp_full, metrics_summary,
        )
        dump_bytes = len(result)
    else:
        result = _dump_aginfer_state_dict(
            cache, bytes_per_token, sp_full, metrics_summary,
        )
        # Dict path: serialised size isn't measured (the call site
        # doesn't go through orjson).  Sentinel.
        dump_bytes = -1
    elapsed_ns = time.perf_counter_ns() - t0
    cache._aginfer_state_dump_metrics.record(
        elapsed_ns=elapsed_ns, dump_bytes=dump_bytes,
    )
    return result


def _dump_aginfer_state_dict(
    cache,
    bytes_per_token: int,
    sp_full: str,
    metrics_summary: dict,
) -> dict:
    """Dict-path snapshot.  Convenient for in-process callers; not
    on the HTTP hot path so per-unit Python dict allocations are
    acceptable."""
    # Walk: collect units + per-subpool DRAM aggregates.
    units: list = []
    units_append = units.append
    dram_used_by_sp: dict[str, int] = {sp_full: 0}
    root = cache.root_node
    base_ct = BASE_COMPONENT_TYPE
    stack = [root]
    stack_pop = stack.pop
    stack_extend = stack.extend
    while stack:
        node = stack_pop()
        stack_extend(node.children.values())
        if node is root:
            continue
        cd = node.component_data[base_ct]
        v = cd.value
        n_tokens_hbm = len(v) if v is not None else 0
        hv = cd.host_value
        n_tokens_dram = len(hv) if hv is not None else 0
        n_bytes: dict[str, dict[str, int]] = {}
        if n_tokens_hbm > 0:
            n_bytes["HBM"] = {sp_full: n_tokens_hbm * bytes_per_token}
        if n_tokens_dram > 0:
            n_bytes["DRAM"] = {sp_full: n_tokens_dram * bytes_per_token}
            dram_used_by_sp[sp_full] += n_tokens_dram * bytes_per_token
        if not n_bytes:
            continue
        residence = list(n_bytes.keys())
        n_tokens = max(n_tokens_hbm, n_tokens_dram)
        hv_list = node.hash_value
        if hv_list:
            unit_hash = hv_list[-1]
            if type(unit_hash) is not str:
                unit_hash = str(unit_hash)
        else:
            unit_hash = f"node-{node.id}"
        try:
            sids = node.session_ids
        except AttributeError:
            sids = None
        units_append({
            "hash": unit_hash,
            "residence": residence,
            "n_tokens": n_tokens,
            "n_bytes": n_bytes,
            "last_access_time": int(node.last_access_time),
            "hit_count": int(node.hit_count),
            "session_ids": sorted(sids) if sids else [],
            # #210: the three structural leaf predicates the daemon's
            # migrate_candidates needs to mirror sglang's apply-site
            # guards (2673/2684/2687) — else reject storms under
            # pressure (remove_not_leaf / remove_hbm_not_device_leaf /
            # remove_dram_not_host_leaf).  is_host_leaf ⟹ is_tree_leaf,
            # but is_device_leaf does NOT, so all three are dumped.
            "is_device_leaf": cache._is_device_leaf(node),
            "is_host_leaf": cache._is_host_leaf(node),
            "is_tree_leaf": len(node.children) == 0,
        })

    pool_usage = _aginfer_pool_usage(cache)
    _aginfer_patch_dram_used(cache, pool_usage, dram_used_by_sp)

    # Per-program post-walk aggregation (1/holders attribution).
    per_program: dict[str, dict] = {}
    for u in units:
        sids = u["session_ids"]
        if not sids:
            continue
        n_holders = len(sids)
        for pid in sids:
            e = per_program.setdefault(pid, {
                "hbm":  {"committed": {}, "inflight": {}},
                "dram": {"committed": {}},
                "state": "REASONING",
                "pre_pause_state": None,
                "unit_hashes": [],
            })
            e["unit_hashes"].append(u["hash"])
            for tier, sp_dict in u["n_bytes"].items():
                if tier == "DISK":
                    continue
                side = "hbm" if tier == "HBM" else "dram"
                bucket = e[side]["committed"]
                for sp, bytes_total in sp_dict.items():
                    bucket[sp] = bucket.get(sp, 0) + bytes_total // n_holders

    # T21 (#181): overlay daemon-pushed program states + GC
    # terminal entries.  Shared helper so dict-path and bytes-
    # path can NEVER diverge (#181 audit).
    _aginfer_overlay_program_states(cache, per_program)

    return {
        "time_counter": int(peek_time_counter()),
        "throughput_ema": _aginfer_throughput_ema(cache),
        "pool_usage": pool_usage,
        "per_program_usage": per_program,
        "units": units,
        "link_stats": _aginfer_link_stats(cache),
        "tier_holding_cost": _aginfer_tier_holding_cost(cache, pool_usage),
        # T40 (#184): count of live daemon-pushed hint entries.  A
        # COUNT, not the table itself — the daemon keeps no shadow
        # cache and never reads hints back (it re-scores from
        # state), so echoing the whole table would only bloat the
        # dump.  Exposed for observability + e2e verification that
        # PUT /aginfer/hints landed.
        "n_aginfer_hints": len(cache._aginfer_hints),
        # T14 — piggybacked observability; pre-this-call summary.
        "state_dump_metrics": metrics_summary,
    }


def _aginfer_patch_dram_used(cache, pool_usage: dict,
                             dram_used_by_sp: dict) -> None:
    """Post-walk DRAM-used patch on pool_usage."""
    for sp, used in dram_used_by_sp.items():
        if sp in pool_usage["DRAM"]["subpools"]:
            e = pool_usage["DRAM"]["subpools"][sp]
            e["used_bytes"] = used
            e["available_bytes"] = max(0, e["cap_bytes"] - used)
            e["evictable_bytes"] = used


def _dump_aginfer_state_bytes(
    cache,
    bytes_per_token: int,
    sp_full: str,
    metrics_summary: dict,
) -> bytes:
    """Allocation-light bytes-path snapshot.

    Hot loop writes each unit's JSON directly into a ``bytearray``
    instead of building a per-unit Python dict.  Per-program
    accumulators are small (one dict-of-dicts per program) so
    their post-walk orjson encode is cheap.

    Wire JSON is byte-equivalent to the dict path's
    ``orjson.dumps(_dump_aginfer_state_dict(...))`` output up to
    key ordering — daemon parses by key, not by position.
    """
    # Pre-bind locals for inner-loop speed.
    base_ct = BASE_COMPONENT_TYPE
    root = cache.root_node

    # Per-program accumulators (built during the walk so we don't
    # have to re-iterate units after).
    # pid → {"hbm_committed": {sp: int}, "dram_committed": {sp: int},
    #        "unit_hashes": [hash, ...]}
    pp: dict[str, dict] = {}
    dram_used_by_sp: dict[str, int] = {sp_full: 0}

    units_buf = bytearray()
    first = True

    stack = [root]
    stack_pop = stack.pop
    stack_extend = stack.extend
    while stack:
        node = stack_pop()
        stack_extend(node.children.values())
        if node is root:
            continue
        cd = node.component_data[base_ct]
        v = cd.value
        n_tokens_hbm = len(v) if v is not None else 0
        hv = cd.host_value
        n_tokens_dram = len(hv) if hv is not None else 0
        if n_tokens_hbm == 0 and n_tokens_dram == 0:
            continue
        hbm_bytes = n_tokens_hbm * bytes_per_token if n_tokens_hbm else 0
        dram_bytes = (n_tokens_dram * bytes_per_token
                      if n_tokens_dram else 0)
        if dram_bytes:
            dram_used_by_sp[sp_full] += dram_bytes

        hv_list = node.hash_value
        if hv_list:
            unit_hash = hv_list[-1]
            if type(unit_hash) is not str:
                unit_hash = str(unit_hash)
        else:
            unit_hash = f"node-{node.id}"
        try:
            sids = node.session_ids
        except AttributeError:
            sids = None
        sids_sorted = sorted(sids) if sids else None

        # ---- write the unit JSON directly into units_buf ----
        if first:
            first = False
        else:
            units_buf.append(0x2C)  # ','
        # {"hash":"<hash>","residence":[...],"n_tokens":N,
        #  "n_bytes":{...},"last_access_time":N,"hit_count":N,
        #  "session_ids":[...]}
        units_buf.extend(b'{"hash":"')
        units_buf.extend(unit_hash.encode("ascii", "backslashreplace"))
        units_buf.extend(b'","residence":[')
        if hbm_bytes and dram_bytes:
            units_buf.extend(b'"HBM","DRAM"')
        elif hbm_bytes:
            units_buf.extend(b'"HBM"')
        else:
            units_buf.extend(b'"DRAM"')
        units_buf.extend(b'],"n_tokens":')
        units_buf.extend(str(max(n_tokens_hbm, n_tokens_dram))
                         .encode("ascii"))
        units_buf.extend(b',"n_bytes":{')
        n_bytes_pieces = []
        if hbm_bytes:
            n_bytes_pieces.append(
                b'"HBM":{"' + sp_full.encode("ascii") + b'":'
                + str(hbm_bytes).encode("ascii") + b'}')
        if dram_bytes:
            n_bytes_pieces.append(
                b'"DRAM":{"' + sp_full.encode("ascii") + b'":'
                + str(dram_bytes).encode("ascii") + b'}')
        units_buf.extend(b",".join(n_bytes_pieces))
        units_buf.extend(b'},"last_access_time":')
        units_buf.extend(str(int(node.last_access_time)).encode("ascii"))
        units_buf.extend(b',"hit_count":')
        units_buf.extend(str(int(node.hit_count)).encode("ascii"))
        if sids_sorted:
            # orjson on the rare non-empty branch (~free vs json.dumps).
            import orjson as _o
            units_buf.extend(b',"session_ids":')
            units_buf.extend(_o.dumps(sids_sorted))
        else:
            units_buf.extend(b',"session_ids":[]')
        # #210: the three structural leaf predicates (see dict-path
        # dump) — migrate_candidates mirrors sglang's apply-site guards
        # 2673/2684/2687 so it never proposes a reject-guaranteed migrate.
        units_buf.extend(
            b',"is_device_leaf":'
            + (b"true" if cache._is_device_leaf(node) else b"false"))
        units_buf.extend(
            b',"is_host_leaf":'
            + (b"true" if cache._is_host_leaf(node) else b"false"))
        units_buf.extend(
            b',"is_tree_leaf":'
            + (b"true" if len(node.children) == 0 else b"false"))
        units_buf.extend(b"}")

        # ---- per-program accumulator (single dict-of-dicts per pid) ----
        if sids_sorted:
            n_holders = len(sids_sorted)
            hbm_share = hbm_bytes // n_holders if hbm_bytes else 0
            dram_share = dram_bytes // n_holders if dram_bytes else 0
            for pid in sids_sorted:
                e = pp.get(pid)
                if e is None:
                    e = {
                        "hbm_committed": {},
                        "dram_committed": {},
                        "unit_hashes": [],
                    }
                    pp[pid] = e
                e["unit_hashes"].append(unit_hash)
                if hbm_share:
                    c = e["hbm_committed"]
                    c[sp_full] = c.get(sp_full, 0) + hbm_share
                if dram_share:
                    c = e["dram_committed"]
                    c[sp_full] = c.get(sp_full, 0) + dram_share

    # ---- finalise pool_usage / aux fields (small dicts) ----
    pool_usage = _aginfer_pool_usage(cache)
    _aginfer_patch_dram_used(cache, pool_usage, dram_used_by_sp)
    link_stats = _aginfer_link_stats(cache)
    tier_holding_cost = _aginfer_tier_holding_cost(cache, pool_usage)
    throughput_ema = _aginfer_throughput_ema(cache)

    # Reshape per_program accumulators into DESIGN §5 form.
    per_program = {
        pid: {
            "hbm":  {"committed": e["hbm_committed"], "inflight": {}},
            "dram": {"committed": e["dram_committed"]},
            "state": "REASONING",
            "pre_pause_state": None,
            "unit_hashes": e["unit_hashes"],
        }
        for pid, e in pp.items()
    }
    # T21 (#181): overlay daemon-pushed program states + GC
    # terminal entries.  SAME helper the dict-path calls — the
    # two cannot diverge by construction (#181 audit).
    _aginfer_overlay_program_states(cache, per_program)

    # ---- assemble final wire JSON ----
    import orjson
    out = bytearray()
    out.extend(b'{"time_counter":')
    out.extend(str(int(peek_time_counter())).encode("ascii"))
    out.extend(b',"throughput_ema":')
    out.extend(orjson.dumps(throughput_ema))
    out.extend(b',"pool_usage":')
    out.extend(orjson.dumps(pool_usage))
    out.extend(b',"per_program_usage":')
    out.extend(orjson.dumps(per_program))
    out.extend(b',"units":[')
    out.extend(units_buf)
    out.extend(b'],"link_stats":')
    out.extend(orjson.dumps(link_stats))
    out.extend(b',"tier_holding_cost":')
    out.extend(orjson.dumps(tier_holding_cost))
    # T40 (#184): live hint-entry count (see dict-path note).  MUST
    # match the dict path's key so the two dumps stay schema-equal.
    out.extend(b',"n_aginfer_hints":')
    out.extend(str(len(cache._aginfer_hints)).encode("ascii"))
    # T14 — piggybacked state-dump cost observability.
    out.extend(b',"state_dump_metrics":')
    out.extend(orjson.dumps(metrics_summary))
    out.extend(b'}')
    return bytes(out)

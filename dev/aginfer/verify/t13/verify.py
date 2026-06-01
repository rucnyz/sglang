"""T13 — bw_free EMA validation (PLAN §1, DESIGN §5 link_stats / §7 bw_free).

T26 ("HiCache + Mooncake throughput EMA") is the task that fills
``recent_throughput_bps`` + ``time_since_last_sample_s`` from real
measurement.  Until T26 lands, sglang emits cold-start placeholders:

  * ``recent_throughput_bps = 0``
  * ``time_since_last_sample_s = 1e12``  (≈ 31 years — "never measured")
  * ``peak_bw_bps`` = realistic device peak

T13 scope (this verify) covers what we CAN validate today:

  A. Sglang emission contract: link_stats shape + cold-start values
  B. Daemon ``bw_free`` branch logic: idle / busy / saturated / fatal

The "compare EMA against ground-truth wall-clock per migrate" stage
named in PLAN §1 T13 is deferred to T26 — there's no ground truth
to compare against until HiCache/Mooncake instrumentation reports.

Usage:
    python dev/aginfer/verify/t13/verify.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import Tier  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon.events import Event, EventKind  # noqa: E402
from daemon.program_tracker import ProgramTracker  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- contract constants (must match kv_scheduler.py + sglang) ----

_LINK_IDLE_SECONDS = 1.0  # daemon's branch threshold
_REQUIRED_LINKS = ("HBM->DRAM", "DRAM->HBM", "DRAM->DISK", "DISK->DRAM")
_REQUIRED_KEYS_PER_LINK = {
    "peak_bw_bps",
    "recent_throughput_bps",
    "time_since_last_sample_s",
}


# ---- fixtures ----


def _state_json_with_links(
    link_stats: Dict[str, Dict[str, Any]],
    *,
    subpool: str = "kv",
) -> Dict[str, Any]:
    """Minimal state JSON exercising only the link_stats branch."""
    def _pool() -> Dict[str, Any]:
        return {"subpools": {
            subpool: {
                "used_bytes": 0, "cap_bytes": 10 * 1024**3,
                "available_bytes": 10 * 1024**3, "evictable_bytes": 0,
                "page_bytes": 64 * 1024,
            }
        }}
    return {
        "time_counter": 0,
        "throughput_ema": {"prefill_bps": 0.0, "decode_per_program": {}},
        "pool_usage": {t: _pool() for t in ("HBM", "DRAM", "DISK")},
        "per_program_usage": {},
        "units": [],
        "link_stats": link_stats,
        "tier_holding_cost": {
            t: {subpool: {"h_max_per_byte_sec": 0.0}}
            for t in ("HBM", "DRAM", "DISK")
        },
    }


def _build(state_json: Dict[str, Any]):
    return kvs.build_paper_state(
        state_json,
        event=Event(EventKind.LLM_PREFILL, session=None),
        tracker=ProgramTracker(),
        unknown_tier_log=set(),
    )


# ============================================================ A. emission


def stage_a0_sglang_emits_four_directions() -> None:
    """Sglang's ``_aginfer_link_stats`` MUST emit all 4 transfer
    directions the daemon expects.  We invoke the method directly
    via the module rather than spinning a full sglang instance —
    the daemon's contract is "these 4 keys with these 3 fields".

    The method lives on ``UnifiedRadixCache`` (sglang) but is
    a pure dict constructor with no instance state used; we can
    call it as an unbound method on a synthetic ``self``."""
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
    )
    # _aginfer_link_stats is a pure dict constructor; class-level call.
    stats = UnifiedRadixCache._aginfer_link_stats(None)  # type: ignore[arg-type]
    missing = set(_REQUIRED_LINKS) - set(stats.keys())
    if missing:
        raise StageFail(
            f"sglang link_stats missing directions: {missing}"
        )
    for link in _REQUIRED_LINKS:
        keys = set(stats[link].keys())
        if not _REQUIRED_KEYS_PER_LINK.issubset(keys):
            raise StageFail(
                f"link {link} missing keys: "
                f"{_REQUIRED_KEYS_PER_LINK - keys}"
            )


def stage_a1_cold_start_recent_throughput_is_zero() -> None:
    """Cold-start contract: until T26 wires measurement, every link's
    ``recent_throughput_bps`` is 0.  A nonzero pre-T26 value would
    mean someone added bogus placeholder data."""
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
    )
    stats = UnifiedRadixCache._aginfer_link_stats(None)  # type: ignore[arg-type]
    for link, entry in stats.items():
        if entry["recent_throughput_bps"] != 0:
            raise StageFail(
                f"link {link}: pre-T26 recent_throughput_bps should be "
                f"0; got {entry['recent_throughput_bps']}"
            )


def stage_a2_cold_start_idle_path_taken() -> None:
    """Cold-start ``time_since_last_sample_s`` must be ABOVE
    LINK_IDLE_SECONDS (= 1.0) so the daemon takes the idle-path
    (``bw_free = peak``).  Pre-T26 emission uses 1e12 as sentinel
    (orjson can't encode ``math.inf``)."""
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
    )
    stats = UnifiedRadixCache._aginfer_link_stats(None)  # type: ignore[arg-type]
    for link, entry in stats.items():
        t_idle = float(entry["time_since_last_sample_s"])
        if t_idle <= _LINK_IDLE_SECONDS:
            raise StageFail(
                f"link {link}: cold-start time_since_last_sample_s "
                f"({t_idle}) must be > LINK_IDLE_SECONDS "
                f"({_LINK_IDLE_SECONDS}); daemon's bw_free will not "
                f"take the idle path"
            )


def stage_a3_peak_bw_positive() -> None:
    """Every direction must have a positive ``peak_bw_bps`` — zero
    or negative is a deployment bug that causes the daemon to
    fatal at handler entry (see B3)."""
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
    )
    stats = UnifiedRadixCache._aginfer_link_stats(None)  # type: ignore[arg-type]
    for link, entry in stats.items():
        peak = entry["peak_bw_bps"]
        if peak <= 0:
            raise StageFail(
                f"link {link}: peak_bw_bps must be > 0; got {peak}"
            )


# ============================================================ B. daemon bw_free


def stage_b0_idle_link_bw_equals_peak() -> None:
    """``time_since_last_sample_s > LINK_IDLE_SECONDS`` (idle) →
    ``bw_free = peak_bw_bps`` regardless of ``recent_throughput_bps``
    (a stale measurement on a now-idle link is meaningless)."""
    link_stats = {
        link: {
            "peak_bw_bps": 64 * 1024**3,
            "recent_throughput_bps": 10 * 1024**3,  # stale; should ignore
            "time_since_last_sample_s": 5.0,  # idle (> 1.0)
        } for link in _REQUIRED_LINKS
    }
    s = _build(_state_json_with_links(link_stats))
    for (src, dst), bw in s.tier_usage.bw_free.items():
        if bw != float(64 * 1024**3):
            raise StageFail(
                f"idle path: bw_free({src.name}->{dst.name}) should "
                f"equal peak (={64 * 1024**3}); got {bw}"
            )


def stage_b1_busy_link_bw_equals_peak_minus_recent() -> None:
    """``time_since_last_sample_s <= LINK_IDLE_SECONDS`` and
    ``recent < peak`` → ``bw_free = peak - recent`` (headroom on a
    contended link)."""
    peak = 64 * 1024**3
    recent = 20 * 1024**3
    link_stats = {
        link: {
            "peak_bw_bps": peak,
            "recent_throughput_bps": recent,
            "time_since_last_sample_s": 0.5,  # busy (< 1.0)
        } for link in _REQUIRED_LINKS
    }
    s = _build(_state_json_with_links(link_stats))
    expected = float(peak - recent)
    for (src, dst), bw in s.tier_usage.bw_free.items():
        if bw != expected:
            raise StageFail(
                f"busy path: bw_free({src.name}->{dst.name}) should "
                f"equal peak-recent (={expected}); got {bw}"
            )


def stage_b2_saturated_link_bw_clamps_to_zero() -> None:
    """``recent >= peak`` (link saturated, possibly negative
    headroom) → ``bw_free = max(0, peak - recent) = 0``.  The clamp
    prevents a negative value from flipping signs in V_u's
    migration-cost term."""
    peak = 64 * 1024**3
    recent = peak + 5 * 1024**3  # over-saturated by 5 GB/s
    link_stats = {
        link: {
            "peak_bw_bps": peak,
            "recent_throughput_bps": recent,
            "time_since_last_sample_s": 0.1,  # busy
        } for link in _REQUIRED_LINKS
    }
    s = _build(_state_json_with_links(link_stats))
    for (src, dst), bw in s.tier_usage.bw_free.items():
        if bw != 0.0:
            raise StageFail(
                f"saturated link must clamp bw_free to 0; got "
                f"bw_free({src.name}->{dst.name}) = {bw}"
            )


def stage_b3_peak_zero_fatals() -> None:
    """``peak_bw_bps <= 0`` on ANY direction is a deployment bug —
    either sglang hasn't measured the link or the operator
    misconfigured.  Daemon fatals with reason
    ``peak_bw_bps_non_positive``."""
    with tempfile.TemporaryDirectory(prefix="t13_b3_") as td:
        script = f"""
import sys, os
sys.path.insert(0, {str(_AGINFER_ROOT)!r})
os.environ['AGINFER_DATA_DIR'] = {td!r}
from daemon import kv_scheduler as kvs
from daemon.events import Event, EventKind
from daemon.program_tracker import ProgramTracker

state = {{
    "time_counter": 0,
    "throughput_ema": {{"prefill_bps": 0.0, "decode_per_program": {{}}}},
    "pool_usage": {{
        t: {{"subpools": {{"kv": {{
            "used_bytes": 0, "cap_bytes": 10*1024**3,
            "available_bytes": 10*1024**3, "evictable_bytes": 0,
            "page_bytes": 64*1024,
        }}}}}} for t in ("HBM","DRAM","DISK")
    }},
    "per_program_usage": {{}}, "units": [],
    "link_stats": {{
        "HBM->DRAM": {{"peak_bw_bps": 0,  # bad
                        "recent_throughput_bps": 0,
                        "time_since_last_sample_s": 5.0}},
        "DRAM->HBM": {{"peak_bw_bps": 64*1024**3,
                        "recent_throughput_bps": 0,
                        "time_since_last_sample_s": 5.0}},
        "DRAM->DISK": {{"peak_bw_bps": 64*1024**3,
                        "recent_throughput_bps": 0,
                        "time_since_last_sample_s": 5.0}},
        "DISK->DRAM": {{"peak_bw_bps": 64*1024**3,
                        "recent_throughput_bps": 0,
                        "time_since_last_sample_s": 5.0}},
    }},
    "tier_holding_cost": {{
        t: {{"kv": {{"h_max_per_byte_sec": 0.0}}}}
        for t in ("HBM","DRAM","DISK")
    }},
}}
kvs.build_paper_state(
    state, event=Event(EventKind.LLM_PREFILL, session=None),
    tracker=ProgramTracker(), unknown_tier_log=set(),
)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": str(_AGINFER_ROOT),
                 "AGINFER_DATA_DIR": td},
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected fatal exit=1; got {result.returncode}; "
                f"stderr={result.stderr[-400:]!r}"
            )
        if "peak_bw_bps_non_positive" not in result.stderr:
            raise StageFail(
                f"expected reason 'peak_bw_bps_non_positive' in "
                f"stderr; got {result.stderr[-400:]!r}"
            )


def stage_b4_idle_threshold_boundary() -> None:
    """Boundary: ``time_since_last_sample_s == LINK_IDLE_SECONDS``
    (exactly 1.0) is NOT idle — the predicate is ``> 1.0``, strict.
    Test the just-above and just-below cases to lock the boundary."""
    peak = 64 * 1024**3
    recent = 20 * 1024**3
    # Just at the boundary (1.0) — busy path.
    link_stats = {
        link: {
            "peak_bw_bps": peak,
            "recent_throughput_bps": recent,
            "time_since_last_sample_s": 1.0,  # not > 1.0
        } for link in _REQUIRED_LINKS
    }
    s = _build(_state_json_with_links(link_stats))
    for (src, dst), bw in s.tier_usage.bw_free.items():
        if bw != float(peak - recent):
            raise StageFail(
                f"boundary t_idle == 1.0: busy path expected "
                f"(={peak - recent}); got bw_free({src.name}->"
                f"{dst.name}) = {bw}"
            )
    # Just above (1.0001) — idle path.
    link_stats2 = {
        link: {
            "peak_bw_bps": peak,
            "recent_throughput_bps": recent,
            "time_since_last_sample_s": 1.0001,  # > 1.0
        } for link in _REQUIRED_LINKS
    }
    s2 = _build(_state_json_with_links(link_stats2))
    for (src, dst), bw in s2.tier_usage.bw_free.items():
        if bw != float(peak):
            raise StageFail(
                f"boundary t_idle = 1.0001: idle path expected "
                f"(={peak}); got bw_free({src.name}->{dst.name}) = {bw}"
            )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 sglang emits 4 directions + 3 keys each",
                              stage_a0_sglang_emits_four_directions),
    ("A1 cold-start recent_throughput_bps == 0 (pre-T26)",
                              stage_a1_cold_start_recent_throughput_is_zero),
    ("A2 cold-start time_since > LINK_IDLE_SECONDS (idle path)",
                              stage_a2_cold_start_idle_path_taken),
    ("A3 peak_bw_bps > 0 for every direction",
                              stage_a3_peak_bw_positive),
    ("B0 idle link → bw_free = peak (ignores stale recent)",
                              stage_b0_idle_link_bw_equals_peak),
    ("B1 busy link → bw_free = peak − recent",
                              stage_b1_busy_link_bw_equals_peak_minus_recent),
    ("B2 saturated link → bw_free clamps to 0 (no negative)",
                              stage_b2_saturated_link_bw_clamps_to_zero),
    ("B3 peak_bw_bps <= 0 → fatal(peak_bw_bps_non_positive)",
                              stage_b3_peak_zero_fatals),
    ("B4 idle threshold boundary: t_idle = 1.0 vs 1.0001",
                              stage_b4_idle_threshold_boundary),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT13 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT13 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

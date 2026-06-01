"""Aginfer daemon entry point.

CLI:

    python -m daemon.main \\
        --sglang-base-url=http://127.0.0.1:30000 \\
        --port=9100 \\
        --kv-scheduler=enabled \\
        --admission-controller=enabled \\
        --theta-hi=0.85 \\
        --theta-lo=0.70

Composes: proxy + EventRouter + (optionally) KvScheduler + AdmissionController.

Emits the T9 startup invariants the run_k.sh grep depends on:
* ``kv_scheduler=<enabled|disabled>``
* ``admission_controller=<enabled|disabled>``
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .admission_controller import AdmissionController, attach_admission_controller
from .event_router import attach_apply_failed_handler
from .kv_scheduler import KvScheduler, attach_kv_scheduler
from .outbound import OutboundQueue
from .proxy import create_app

logger = logging.getLogger("aginfer.daemon")


def _parse_bool_flag(s: str, name: str) -> bool:
    """Accept ``enabled`` / ``disabled`` (T9 README's spelling).
    Other forms raise — we don't want ``true``/``1``/``yes`` ambiguity
    leaking into Run K's startup grep.
    """
    if s == "enabled":
        return True
    if s == "disabled":
        return False
    raise SystemExit(
        f"--{name} must be 'enabled' or 'disabled'; got {s!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aginfer-daemon")
    p.add_argument(
        "--sglang-base-url", required=True,
        help="sglang server base URL (e.g. http://127.0.0.1:30000)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument(
        "--kv-scheduler", default="enabled",
        help="enabled|disabled — gate T7 paper §4 migrations",
    )
    p.add_argument(
        "--admission-controller", default="enabled",
        help="enabled|disabled — gate T8 program-level pause/resume",
    )
    p.add_argument(
        "--theta-hi", type=float, default=0.85,
        help="admission pause-trigger watermark (default 0.85)",
    )
    p.add_argument(
        "--theta-lo", type=float, default=0.70,
        help="admission resume-gate watermark (default 0.70)",
    )
    p.add_argument(
        "--max-pauses-per-event", type=int, default=16,
        help="admission per-event pause/resume cap (default 16)",
    )
    p.add_argument(
        "--observability-summary-every-n", type=int, default=200,
        help=(
            "T42: emit one daemon_obs_summary line per N handled events "
            "(default 200).  Set lower (e.g. 20) for short load demos / "
            "stress probes so the summary cadence is visible in modest "
            "traffic."
        ),
    )
    p.add_argument(
        "--log-level", default="info",
        help="uvicorn / daemon log level",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    enable_kv = _parse_bool_flag(args.kv_scheduler, "kv-scheduler")
    enable_admission = _parse_bool_flag(
        args.admission_controller, "admission-controller"
    )

    # Configure root logging so the T9 startup-invariant grep finds
    # our marker on stdout (uvicorn forwards stdout to its own log
    # handler when started programmatically).
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    app = create_app(
        sglang_base_url=args.sglang_base_url,
        observability_summary_every_n=args.observability_summary_every_n,
    )
    bus = app.state.event_bus
    tracker = app.state.program_tracker
    router = app.state.event_router

    # Attach layers per flags.  Order matters (T8 R2-M2 guard enforces
    # this): kv_scheduler MUST attach BEFORE admission_controller.
    sched = None
    admission = None
    # T36: shared outbound queue for all fire-and-forget dispatches.
    # Lifecycle is tied to the FastAPI startup / shutdown hooks below
    # so the worker is alive whenever the daemon is.
    outbound = OutboundQueue(
        sglang_base_url=args.sglang_base_url,
        observability=router.observability,
    )
    app.state.outbound = outbound

    if enable_kv:
        sched = KvScheduler(
            tracker=tracker, sglang_base_url=args.sglang_base_url,
            # T42 — share the router's T42 aggregator so kv_scheduler's
            # per-skip reason counts land in the same observability
            # summary the router emits.
            observability=router.observability,
            # T36 — fire-and-forget POST.  KvScheduler._dispatch_migrate
            # enqueues onto this queue and returns immediately;
            # OutboundWorker pops + POSTs in the background.
            outbound=outbound,
        )
        attach_kv_scheduler(router, sched)
        app.state.kv_scheduler = sched
    if enable_admission:
        if not enable_kv:
            raise SystemExit(
                "--admission-controller=enabled requires "
                "--kv-scheduler=enabled (admission's composite wraps "
                "kv_scheduler's handler)"
            )
        admission = AdmissionController(
            tracker=tracker,
            theta_hi=args.theta_hi,
            theta_lo=args.theta_lo,
            max_pauses_per_event=args.max_pauses_per_event,
        )
        attach_admission_controller(router, admission)
        app.state.admission_controller = admission

    # T37 — APPLY_FAILED handler is unconditional (no kv_scheduler /
    # admission flag gates it; the webhook arrives regardless of which
    # layers are attached, and the daemon's observability counter needs
    # to track it).  Attach AFTER admission so we don't double-wrap
    # the composite for MEMORY_PRESSURE/PRESSURE_RESOLVED.
    attach_apply_failed_handler(router)

    # T9 startup-invariant markers — single grep-friendly line.
    logger.info(
        "[aginfer] kv_scheduler=%s admission_controller=%s "
        "theta_hi=%s theta_lo=%s sglang_base_url=%s port=%d",
        "enabled" if enable_kv else "disabled",
        "enabled" if enable_admission else "disabled",
        args.theta_hi, args.theta_lo,
        args.sglang_base_url, args.port,
    )

    # Register a shutdown handler that dumps cumulative counters as a
    # single structured-metric line.  parse_daemon_events.py looks
    # for `event=cycle_summary`.  Wall-clock side: this fires once per
    # daemon lifetime, on SIGTERM/SIGINT; cost is negligible.
    @app.on_event("shutdown")
    async def _emit_cycle_summary():  # type: ignore[unused-function]
        from ._metrics import m as _m
        kv_calls = sched.migrate_calls if sched is not None else 0
        kv_decisions = sched.decisions if sched is not None else 0
        adm_pauses = admission.pause_decisions if admission is not None else 0
        adm_resumes = admission.resume_decisions if admission is not None else 0
        _m(
            "cycle_summary",
            events_received=router.events_received,
            events_handled=router.events_handled,
            handler_failures=router.handler_failures,
            kv_decisions=kv_decisions,
            kv_migrate_calls=kv_calls,
            adm_pauses=adm_pauses,
            adm_resumes=adm_resumes,
            theta_hi=args.theta_hi,
            theta_lo=args.theta_lo,
        )
        # T42 — final observability summary so the last partial window
        # (events since the last summary_every_n cadence emission) is
        # not lost on shutdown.  Same line format as the periodic
        # `daemon_obs_summary`; operator's grep pipeline gets both.
        router.observability.emit_summary()

    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()

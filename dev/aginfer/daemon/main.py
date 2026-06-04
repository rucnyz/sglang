"""Aginfer daemon entry point.

CLI:

    python -m daemon.main \\
        --sglang-base-url=http://127.0.0.1:30000 \\
        --port=9100 \\
        --kv-scheduler=enabled \\
        --admission-controller=enabled \\
        --theta-hi=0.85 \\
        --theta-lo=0.70

Composes: proxy + EventRouter + (optionally) KvScheduler.  Admission is
the §8 candidate generator inside KvScheduler's joint_decide (#194), not
a separate composed layer; the flag toggles KvScheduler.admission_enabled.

Emits the T9 startup invariants the run_k.sh grep depends on:
* ``kv_scheduler=<enabled|disabled>``
* ``admission_controller=<enabled|disabled>``
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

# #194: admission is now the §8 candidate generator consumed by
# kv_scheduler's joint_decide, not a separate composed handler — no
# AdmissionController / attach_admission_controller import needed.
from .event_router import (
    attach_apply_failed_handler,
    attach_hash_collision_handler,
    attach_session_end_handler,
)
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
        "--theta-crit", type=float, default=0.90,
        help="critical-pressure threshold (default 0.90)",
    )
    p.add_argument(
        "--heartbeat-s", type=float, default=5.0,
        help="seconds between still_high heartbeats (default 5.0)",
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
        "--sustained-escalate-fails", type=int, default=100,
        help=(
            "T36/F3 #164: outbound consecutive POST failures threshold "
            "for sustained-escalation fatal.  Default 100.  Daemon "
            "self-kills (os._exit 1; #166) when this AND --sustained-"
            "escalate-age-s are both crossed simultaneously (DESIGN §10 "
            "sustained tier).  Supervisor (systemd/k8s) restarts; if "
            "sglang is still down, the daemon CrashLoopBackOffs visibly "
            "to ops."
        ),
    )
    p.add_argument(
        "--sustained-escalate-age-s", type=float, default=300.0,
        help=(
            "T36/F3 #164: outbound oldest-pending-batch age (seconds) "
            "threshold for sustained-escalation fatal.  Default 300 (5 "
            "minutes).  Both this AND --sustained-escalate-fails must "
            "trip for fatal — low-traffic dead-sglang doesn't escalate."
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
        theta_hi=args.theta_hi,
        theta_lo=args.theta_lo,
        theta_crit=args.theta_crit,
        heartbeat_s=args.heartbeat_s,
        observability_summary_every_n=args.observability_summary_every_n,
    )
    bus = app.state.event_bus
    tracker = app.state.program_tracker
    router = app.state.event_router

    # Attach layers per flags.  Order matters (T8 R2-M2 guard enforces
    # this): kv_scheduler MUST attach BEFORE admission_controller.
    sched = None
    # T36 (DESIGN §6 B4): shared outbound queue for all fire-and-forget
    # dispatches.  Mandatory — KvScheduler._dispatch_migrate raises if
    # outbound is None.  Lifecycle tied to FastAPI startup / shutdown
    # via app.state.outbound (see proxy.py).  T36/F3 (#164): sustained-
    # escalation thresholds plumbed from CLI for ops-tunable crash-
    # only-software backstop.
    outbound = OutboundQueue(
        sglang_base_url=args.sglang_base_url,
        observability=router.observability,
        escalate_failures=args.sustained_escalate_fails,
        escalate_oldest_age_s=args.sustained_escalate_age_s,
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
                "--kv-scheduler=enabled (admission is the program-level "
                "candidate generator inside kv_scheduler's joint_decide)"
            )
        # #194 / DESIGN §9: admission is no longer a separate handler
        # composed on top of kv_scheduler — it is the program-level
        # candidate generator (§8) consumed by the SINGLE joint_decide
        # the kv_scheduler handler runs.  Enabling it just turns on the
        # Pause/Resume levers in that joint decision (the kv-only arm
        # leaves them off).  Thresholds come from the router (§10).
        sched.admission_enabled = True

    # T37 — APPLY_FAILED handler is unconditional (no kv_scheduler /
    # admission flag gates it; the webhook arrives regardless of which
    # layers are attached, and the daemon's observability counter needs
    # to track it).  Attach AFTER admission so we don't double-wrap
    # the composite for MEMORY_PRESSURE/PRESSURE_RESOLVED.
    attach_apply_failed_handler(router)

    # T24 (#182) — HASH_COLLISION handler.  Also unconditional;
    # detection lives in sglang's apply_aginfer_migrations DFS and
    # fires regardless of which daemon layers are attached.
    # Handler calls fatal('hash_collision') so the supervisor
    # restarts the daemon (sglang's _aginfer_collision_seen
    # dedupes pair-by-pair, so a persistent collision fires exactly
    # one daemon fatal per (node_a, node_b) pair).
    attach_hash_collision_handler(router)

    # T41 (#185) + T187 (#187) — SESSION_END handler.  Attached AFTER
    # kv_scheduler so this composite owns SESSION_END: it transitions
    # ENDED + releases the proxy gate with 499 for a parked PAUSED
    # request (F5), then runs the migrate D_t = session_scoped_units(p)
    # via kv_scheduler.handle (T187), then enqueues PUT
    # /aginfer/program_paused {ENDED}.  Passing ``sched`` (None when
    # --kv-scheduler=disabled) wires the migrate step; without it the
    # handler is pure F5.
    attach_session_end_handler(router, tracker, outbound, sched)

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
        # #194: admission is folded into joint_decide (no AdmissionController);
        # pause/resume counters now live on the KvScheduler that dispatches
        # the joint plan.  (Pre-#194 these read a deleted `admission` var —
        # a NameError that aborted the whole shutdown handler, dropping the
        # cycle_summary metric + the final observability summary.)
        adm_pauses = sched.pause_calls if sched is not None else 0
        adm_resumes = sched.resume_calls if sched is not None else 0
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

#!/usr/bin/env python3
"""aginfer_bridge — additive HTTP <-> /engine/call_tokenizer_manager translator.

The aginfer daemon (dev/aginfer/daemon) speaks plain HTTP ``/aginfer/*`` to an
sglang engine.  On Dynamo the engine runs IN-PROCESS inside the dynamo.sglang
worker, with no sglang HTTP server — but with ``--enable-rl`` + ``DYN_SYSTEM_PORT``
the worker exposes a generic passthrough at
``POST http://localhost:<DYN_SYSTEM_PORT>/engine/call_tokenizer_manager``
(body ``{"method","args","kwargs"}``; typed args as ``{"io_struct.ClassName":{...}}``).

This bridge is a tiny, separate process that translates each daemon
``/aginfer/<route>`` call into a ``call_tokenizer_manager`` invocation of the
matching ``tokenizer_manager`` method, then reproduces the SAME flatten the
native sglang ``/aginfer/*`` HTTP routes apply.  Point the daemon's
``--sglang-base-url`` at THIS bridge.

ZERO changes to Dynamo core and ZERO changes to the daemon.

Direction of traffic (do not confuse):
  daemon  --(GET state / POST migrate / PUT hints / PUT program_paused)-->  bridge --> engine
  engine  --(--aginfer-notify-url webhook: memory_pressure / still_high)-->  daemon :9100/aginfer/event   (NOT this bridge)
  engine  --(GET /aginfer/thresholds pull)-->                               daemon :9100                  (NOT this bridge)

So this bridge hosts EXACTLY the four daemon-outbound routes below.  /aginfer/event,
/aginfer/thresholds and /aginfer/session_prefix are hosted BY the daemon, not here.

Contract verified against sglang http_server.py + tokenizer_control_mixin.py +
the daemon's outbound.py / kv_scheduler.py (see dev/dynamo/README.md §5).
"""

import argparse
import asyncio
import json
import logging
import os

import aiohttp
from aiohttp import web

logger = logging.getLogger("aginfer_bridge")

# ----------------------------------------------------------------------------
# call_tokenizer_manager plumbing
# ----------------------------------------------------------------------------


async def _ctm(app, method, args=None):
    """Invoke a tokenizer_manager method via /engine/call_tokenizer_manager.

    Returns the parsed JSON reply (``_normalize_result`` shape — for the aginfer
    methods, which all return ``List[ReqOutput]``, that is ``{"result":[...]}``).
    Serialised through a lock so concurrent daemon calls never interleave two
    ZMQ control-plane fan-outs on the engine.
    """
    body = {"method": method}
    if args is not None:
        body["args"] = args
    async with app["ctm_lock"]:
        async with app["http"].post(app["ctm_url"], json=body) as r:
            text = await r.text()
            if r.status != 200:
                # A CTM 500/4xx must NOT be masked as a daemon 200 — surface it
                # so the daemon counts a failure rather than fatal()ing on a
                # malformed state body.
                raise web.HTTPBadGateway(
                    text=f"call_tokenizer_manager {method} -> {r.status}: {text[:512]}"
                )
            return json.loads(text)


def _ranks(ctm_reply):
    """Extract the per-DP-rank list from a _normalize_result reply.

    aginfer methods return List[ReqOutput] -> _normalize_result wraps as
    {"result":[<per-rank dict>, ...]}.  Tolerate a bare list defensively.
    """
    if isinstance(ctm_reply, dict) and isinstance(ctm_reply.get("result"), list):
        return ctm_reply["result"]
    if isinstance(ctm_reply, list):
        return ctm_reply
    return [ctm_reply]


def _state_dict(rank):
    """Recover one rank's state dict from asdict(GetAginferStateReqOutput).

    Primary path (sglang fork emits ``state_bytes`` as a JSON *str*): json.loads
    the string.  Defensive fallbacks cover a stock engine that still ships raw
    ``bytes`` (which the Rust pythonize->serde_json path corrupts into an array
    of byte-ints) or the ``state`` dict form (unsupported-cache placeholder).
    """
    st = rank.get("state")
    if st is not None:
        return st
    sb = rank.get("state_bytes")
    if sb is None:
        raise web.HTTPBadGateway(text="state rank carries neither state nor state_bytes")
    if isinstance(sb, str):
        return json.loads(sb)
    if isinstance(sb, list):  # pythonize int-array corruption (stock bytes path)
        return json.loads(bytes(sb))
    if isinstance(sb, (bytes, bytearray)):
        return json.loads(sb)
    raise web.HTTPBadGateway(text=f"unparseable state_bytes type {type(sb).__name__}")


# ----------------------------------------------------------------------------
# hint normalization — replicates http_server.py _validate_hints_body, which the
# bridge bypasses.  Without this the S2 holder-count lever can be silently
# neutralised (n_holders defaulting to absent rather than 0).
# ----------------------------------------------------------------------------


def _norm_num(v, lo=None, hi=None, as_int=False, field=""):
    if isinstance(v, bool):
        raise ValueError(f"{field}: bool not allowed")
    x = int(v) if as_int else float(v)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{field}: non-finite")
    if lo is not None and x < lo:
        raise ValueError(f"{field}: < {lo}")
    if hi is not None and x > hi:
        raise ValueError(f"{field}: > {hi}")
    return x


def _norm_hint(h):
    hh = str(h["hash"])
    if not hh:
        raise ValueError("hash: empty")
    return {
        "hash": hh,
        "p_hat": _norm_num(h["p_hat"], 0.0, 1.0, field="p_hat"),
        "lambda": _norm_num(h["lambda"], 0.0, None, field="lambda"),
        "stamp": _norm_num(h["stamp"], 0, None, as_int=True, field="stamp"),
        "n_holders": _norm_num(
            h.get("n_holders", 0), 0, None, as_int=True, field="n_holders"
        ),  # ALWAYS emit — the S2 lever
    }


# ----------------------------------------------------------------------------
# routes — exactly the four the daemon calls outbound
# ----------------------------------------------------------------------------


async def get_state(request):
    """GET /aginfer/state -> tokenizer_manager.get_aginfer_state().

    Single-rank: the inner state dict AT TOP LEVEL (no envelope) — the daemon
    json.loads this and fatal()s on any missing required key, so never wrap.
    Multi-rank: {"per_rank":[<each rank's state dict>]} for the daemon's
    _flatten_per_rank to aggregate.
    """
    reply = await _ctm(request.app, "get_aginfer_state")
    states = [_state_dict(r) for r in _ranks(reply)]
    body = states[0] if len(states) == 1 else {"per_rank": states}
    return web.json_response(body)


async def migrate(request):
    """POST /aginfer/migrate {"actions":[...], "batch_id":...} -> migrate_aginfer.

    Drop the daemon's top-level batch_id (MigrateAginferReq has no such field);
    forward each action verbatim (hash/add_tiers/remove_tiers/action_id — the
    daemon already emits all four).  The daemon discards the response body and
    only checks status, but mirror the native flatten anyway.
    """
    payload = await request.json()
    args = [{"io_struct.MigrateAginferReq": {"actions": payload["actions"]}}]
    ranks = _ranks(await _ctm(request.app, "migrate_aginfer", args))
    if len(ranks) == 1:
        r0 = ranks[0]
        out = {
            "applied": r0["applied"],
            "applied_hashes": r0["applied_hashes"],
            "skipped": r0["skipped"],
        }
    else:
        out = {
            "per_rank": [
                {
                    "applied": r["applied"],
                    "applied_hashes": r["applied_hashes"],
                    "skipped": r["skipped"],
                }
                for r in ranks
            ]
        }
    return web.json_response(out)


async def hints(request):
    """PUT /aginfer/hints {"hints":[...], "batch_id":...} -> update_aginfer_hints.

    Replicate _validate_hints_body normalization (esp. always-emit n_holders),
    drop batch_id, AND the per-rank ok, SUM applied, raise 400 on any rank
    failure (so the daemon learns a rejection rather than seeing a false 200).
    """
    payload = await request.json()
    try:
        norm = [_norm_hint(h) for h in payload["hints"]]
    except (KeyError, ValueError, TypeError) as exc:
        return web.json_response({"detail": f"validation: {exc}"}, status=400)
    args = [{"io_struct.UpdateAginferHintsReq": {"hints": norm}}]
    ranks = _ranks(await _ctm(request.app, "update_aginfer_hints", args))
    if not all(r["ok"] for r in ranks):
        first = next(r for r in ranks if not r["ok"])
        return web.json_response({"detail": f"validation: {first['reason']}"}, status=400)
    return web.json_response(
        {"ok": True, "ranks": len(ranks), "applied": sum(int(r["applied"]) for r in ranks)}
    )


async def program_paused(request):
    """PUT /aginfer/program_paused {"pid","state","pre_pause_state","batch_id"}
    -> update_aginfer_program_paused.  Drop batch_id; AND ok / SUM applied / 400."""
    p = await request.json()
    kw = {
        "pid": p["pid"],
        "state": p["state"],
        "pre_pause_state": p.get("pre_pause_state"),
    }
    args = [{"io_struct.UpdateAginferProgramPausedReq": kw}]
    ranks = _ranks(await _ctm(request.app, "update_aginfer_program_paused", args))
    if not all(r["ok"] for r in ranks):
        first = next(r for r in ranks if not r["ok"])
        return web.json_response({"detail": f"validation: {first['reason']}"}, status=400)
    return web.json_response(
        {"ok": True, "ranks": len(ranks), "applied": sum(int(r["applied"]) for r in ranks)}
    )


async def health(request):
    """GET /health — bridge liveness + a probe of the CTM endpoint."""
    try:
        ranks = _ranks(await _ctm(request.app, "get_aginfer_state"))
        return web.json_response({"ok": True, "ranks": len(ranks)})
    except Exception as exc:  # noqa: BLE001
        return web.json_response({"ok": False, "error": str(exc)}, status=502)


# ----------------------------------------------------------------------------
# app wiring
# ----------------------------------------------------------------------------


async def _on_start(app):
    app["http"] = aiohttp.ClientSession()
    app["ctm_lock"] = asyncio.Lock()


async def _on_stop(app):
    await app["http"].close()


def build_app(ctm_url):
    app = web.Application()
    app["ctm_url"] = ctm_url
    app.on_startup.append(_on_start)
    app.on_cleanup.append(_on_stop)
    app.add_routes(
        [
            web.get("/aginfer/state", get_state),
            web.post("/aginfer/migrate", migrate),
            web.put("/aginfer/hints", hints),
            web.put("/aginfer/program_paused", program_paused),
            web.get("/health", health),
        ]
    )
    return app


def main():
    ap = argparse.ArgumentParser(description="aginfer daemon<->engine HTTP bridge")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AGINFER_BRIDGE_PORT", "9200")))
    default_ctm = os.environ.get(
        "AGINFER_CTM_URL",
        f"http://127.0.0.1:{os.environ.get('DYN_SYSTEM_PORT', '8081')}/engine/call_tokenizer_manager",
    )
    ap.add_argument("--ctm-url", default=default_ctm)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logger.info("aginfer_bridge on %s:%d -> CTM %s", args.host, args.port, args.ctm_url)
    web.run_app(build_app(args.ctm_url), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

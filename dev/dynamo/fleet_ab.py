#!/usr/bin/env python3
"""Fleet A/B driver (async) for the Dynamo aginfer experiment.

High-concurrency multi-trajectory agent fleet against a Dynamo frontend
(thunderagent_router baseline vs aginfer_router) with a 4-tier HiCache sglang backend
under a capped HBM pool. Tags every request with nvext.agent_context.

Rewritten on aiohttp: one event loop, a single pooled ClientSession with a high
connector limit -> survives hundreds of concurrent streaming connections (urllib +
threads exhausted sockets and timed out, producing spurious fails).

Workload: SIZE != VALUE, the axis a value-gated router must beat a size-gated one on:
  Class A (high-value, SMALL): short shared prefix, MANY turns (high reuse frequency).
      ThunderAgent pauses these first (smallest token_total) -> evicted -> re-prefill
      every turn. aginfer keeps them resident.
  Class B (low-value, LARGE): long prefix, FEW turns (low reuse). ThunderAgent keeps
      these; aginfer pauses them (low V_u).

Pressure: ThunderAgent only pauses when worker KV util >= 0.95 (pause_threshold), checked
every 5s. With HiCache demoting cold prefixes to DRAM, GPU pressure comes from IN-FLIGHT
(active, non-demotable) KV -> need many requests decoding simultaneously: large
--max-tokens + tight --gap + high trajectory count so active KV approaches the pool.

Metrics (per arm):
  client: per-request TTFT (streaming first token), makespan, fails/timeouts.
  server: parse the sglang backend log's report_prefill_stats over the run window ->
          aggregate #new-token (re-prefill, lower=better) vs #cached-token.

Usage (in the container):
  python fleet_ab.py --base http://localhost:8100 --model Qwen/Qwen3-0.6B \
     --classA 40 --turnsA 8 --tokA 600 --classB 24 --turnsB 2 --tokB 2200 \
     --gap 0.25 --max-tokens 160 --tag run1
"""
import argparse, json, time, asyncio, sys
import aiohttp

FILLER = ("This is shared system context that the agent must keep in mind throughout the "
          "entire trajectory. It defines tools, constraints, and the operating procedure. ")

def make_prefix(approx_tokens, salt, run_salt=""):
    words_needed = max(4, int(approx_tokens / 0.75))
    base = (FILLER * (words_needed // len(FILLER.split()) + 1)).split()
    return f"[ctx {run_salt}/{salt}] " + " ".join(base[:words_needed])

class Stats:
    def __init__(self):
        self.ttfts = []
        self.resume_ttfts = []
        self.first_ttfts = []
        self.fails = 0
        self.timeouts = 0
        self.requests = 0
        self.err_samples = []
        self.inflight = 0
    def add(self, ttft, ok, turn_idx, err=None, timed_out=False):
        self.requests += 1
        if not ok or ttft is None:
            self.fails += 1
            if timed_out:
                self.timeouts += 1
            if err and len(self.err_samples) < 6:
                self.err_samples.append(str(err)[:120])
            return
        self.ttfts.append(ttft)
        (self.first_ttfts if turn_idx == 0 else self.resume_ttfts).append(ttft)

async def stream_request(session, base, model, traj_id, session_id, messages, max_tokens,
                         req_timeout, final=False, priority=None):
    url = base.rstrip("/") + "/v1/chat/completions"
    ctx = {"trajectory_id": traj_id, "session_id": session_id, "session_type_id": "agent"}
    if final:
        ctx["trajectory_final"] = True
    nvext = {"agent_context": ctx}
    if priority is not None:
        nvext["agent_hints"] = {"priority": int(priority)}
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "stream": True, "temperature": 0.0, "seed": 0,
        "stream_options": {"include_usage": True},
        "nvext": nvext,
    }
    t0 = time.monotonic()
    ttft = None
    parts = []
    try:
        timeout = aiohttp.ClientTimeout(total=req_timeout, sock_connect=10)
        async with session.post(url, json=body, timeout=timeout) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                ch = obj.get("choices") or []
                if ch:
                    piece = (ch[0].get("delta", {}) or {}).get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        parts.append(piece)
        total = time.monotonic() - t0
        if ttft is None:
            ttft = total
        return ttft, True, "".join(parts), None, False
    except asyncio.TimeoutError:
        return None, False, "", "timeout", True
    except Exception as e:
        return None, False, "", f"{type(e).__name__}:{e}", False

async def run_trajectory(session, args, traj_id, prefix, turns, stats, priority=None):
    session_id = f"sess-{traj_id}"
    history = [{"role": "system", "content": prefix}]
    for t in range(turns):
        history.append({"role": "user",
                        "content": f"Step {t+1}: continue the task. One-sentence update."})
        stats.inflight += 1
        ttft, ok, text, err, to = await stream_request(
            session, args.base, args.model, traj_id, session_id, history,
            args.max_tokens, args.req_timeout, priority=priority)
        stats.inflight -= 1
        stats.add(ttft, ok, t, err=err, timed_out=to)
        # FIXED reply (NOT the real decode) so the conversation history — and thus every
        # turn's prefill prompt — is byte-identical across arms regardless of decode
        # non-determinism (batch composition). The S2 metric is prefill; decode output is
        # irrelevant to it. This is the token-identity guarantee for the fair A/B.
        history.append({"role": "assistant",
                        "content": "Acknowledged. Continuing the task as instructed."})
        if t < turns - 1 and args.gap > 0:
            await asyncio.sleep(args.gap)
    # close-ping: release the program from the router accounting (frees its util).
    # Sent as a dedicated trajectory_final request (not counted in TTFT stats).
    try:
        await stream_request(session, args.base, args.model, traj_id, session_id,
                             [{"role": "user", "content": "done"}], 1,
                             args.req_timeout, final=True)
    except Exception:
        pass

def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p/100.0)*(len(s)-1))))
    return round(s[i], 4)

async def main_async(args):
    stats = Stats()
    jobs = []
    rs = args.run_salt
    pA = args.priorityA if args.priorityA >= 0 else None
    pB = args.priorityB if args.priorityB >= 0 else None
    for i in range(args.classA):
        jobs.append((f"{rs}A{i}", make_prefix(args.tokA, f"A{i % max(1,args.groupsA)}", rs), args.turnsA, pA))
    for i in range(args.classB):
        jobs.append((f"{rs}B{i}", make_prefix(args.tokB, f"B{i}", rs), args.turnsB, pB))

    print(f"[{args.tag}] launching {len(jobs)} trajectories "
          f"(A={args.classA}x{args.turnsA}@{args.tokA}tok/{args.groupsA}grp, "
          f"B={args.classB}x{args.turnsB}@{args.tokB}tok, gap={args.gap}s, "
          f"maxtok={args.max_tokens})", flush=True)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    t_start = time.monotonic()

    async def progress():
        try:
            while True:
                await asyncio.sleep(4)
                print(f"  [+{time.monotonic()-t_start:5.1f}s] done={stats.requests} "
                      f"ok={len(stats.ttfts)} fail={stats.fails} to={stats.timeouts} "
                      f"inflight={stats.inflight}", flush=True)
        except asyncio.CancelledError:
            return

    async with aiohttp.ClientSession(connector=connector) as session:
        prog = asyncio.ensure_future(progress())
        await asyncio.gather(*[
            run_trajectory(session, args, tid, pfx, turns, stats, priority=prio)
            for (tid, pfx, turns, prio) in jobs])
        prog.cancel()
    makespan = time.monotonic() - t_start

    out = {
        "tag": args.tag, "makespan_s": round(makespan, 3),
        "requests": stats.requests, "fails": stats.fails, "timeouts": stats.timeouts,
        "ttft_mean": round(sum(stats.ttfts)/len(stats.ttfts), 4) if stats.ttfts else None,
        "ttft_p50": pct(stats.ttfts, 50), "ttft_p95": pct(stats.ttfts, 95),
        "ttft_p99": pct(stats.ttfts, 99),
        "resume_ttft_mean": round(sum(stats.resume_ttfts)/len(stats.resume_ttfts), 4) if stats.resume_ttfts else None,
        "resume_ttft_p95": pct(stats.resume_ttfts, 95),
        "first_ttft_mean": round(sum(stats.first_ttfts)/len(stats.first_ttfts), 4) if stats.first_ttfts else None,
        "err_samples": stats.err_samples,
        "t_start_wall": round(time.time() - makespan, 3),
        "t_end_wall": round(time.time(), 3),
    }
    print("RESULT " + json.dumps(out), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8100")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--classA", type=int, default=40)
    ap.add_argument("--turnsA", type=int, default=8)
    ap.add_argument("--tokA", type=int, default=600)
    ap.add_argument("--groupsA", type=int, default=8)   # shared-prefix groups within A
    ap.add_argument("--classB", type=int, default=24)
    ap.add_argument("--turnsB", type=int, default=2)
    ap.add_argument("--tokB", type=int, default=2200)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=160)
    ap.add_argument("--req-timeout", dest="req_timeout", type=float, default=90.0)
    ap.add_argument("--run-salt", dest="run_salt", default="r0")
    ap.add_argument("--priorityA", type=int, default=-1)  # -1 = omit (no priority hint)
    ap.add_argument("--priorityB", type=int, default=-1)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()

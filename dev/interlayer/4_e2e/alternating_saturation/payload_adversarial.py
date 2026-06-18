"""alternating_saturation — adversarial alternating-saturation workload dispatcher.

design.md §alternating_saturation:
> Conjecture: a workload that alternates between KV-saturated and
> mamba-saturated phases at the tick-boundary period forces the
> Admitter to fire constantly back-and-forth. Throughput should not
> regress vs `inter=off` baseline, even though actuator work is high.
>
> Test: synthetic R1+M3 alternating sweep, 5-minute total, period
> ≈ 2× Budgeter tick (e.g., 2s phases at 1 Hz tick).
> Pass: output_throughput_inter ≥ output_throughput_off · 0.95

Implementation:
- Phase KV: 50 reqs/sec, input=1024 (long context → KV-heavy)
- Phase MAMBA: 50 reqs/sec, input=64 (short context → mamba-heavy)
- Switch phase every 2.0s
- 5-minute total

Each request: short output (64 tokens) so per-req lifetime ≈ phase
duration. Mid-phase, the "wrong" pool fills; planner sees pressure
shift and (if budgeter on) may fire. Fire direction should ALSO
alternate. alternating_saturation passes if alternation doesn't cause runaway thrash.

Dispatches via sglang's /v1/completions HTTP endpoint. Records:
- per-req: send_ts, recv_ts, prompt_len, output_len
- aggregate: completed count, total output tokens, mean TPOT

Usage:
  python payload_adversarial.py --port 30077 --duration 300 --out /tmp/d8c
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp


async def _send(session, url, prompt_tokens, output_tokens, ts0):
    """Send one /v1/completions request. Returns dict with timing."""
    payload = {
        "model": "default",
        "prompt": " ".join(["lorem"] * prompt_tokens),
        "max_tokens": output_tokens,
        "stream": False,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    t_send = time.perf_counter() - ts0
    try:
        async with session.post(url, json=payload, timeout=120) as resp:
            data = await resp.json()
            t_recv = time.perf_counter() - ts0
            ok = resp.status == 200
            output_text = ""
            if ok and "choices" in data and data["choices"]:
                output_text = data["choices"][0].get("text", "")
            return {
                "send_t": t_send,
                "recv_t": t_recv,
                "ok": ok,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "output_chars": len(output_text),
            }
    except Exception as e:
        return {
            "send_t": t_send,
            "recv_t": time.perf_counter() - ts0,
            "ok": False,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "error": str(e),
        }


async def driver(args, out_records):
    url = f"http://{args.host}:{args.port}/v1/completions"
    ts0 = time.perf_counter()
    tasks = []
    rng = random.Random(args.seed)

    async with aiohttp.ClientSession() as session:
        # Phase generator: alternates KV-heavy / MAMBA-heavy every
        # `args.phase_s` seconds.
        end = ts0 + args.duration
        next_send = ts0
        phase_dt = 1.0 / args.rps  # interval between request sends
        while time.perf_counter() < end:
            # Sleep until next send
            now = time.perf_counter()
            if now < next_send:
                await asyncio.sleep(next_send - now)
            # Determine current phase
            phase_idx = int((time.perf_counter() - ts0) / args.phase_s)
            is_kv_phase = (phase_idx % 2 == 0)
            if is_kv_phase:
                in_len = args.kv_input_len
            else:
                in_len = args.mamba_input_len
            out_len = args.output_len
            tasks.append(asyncio.create_task(
                _send(session, url, in_len, out_len, ts0)
            ))
            next_send += phase_dt

        # Wait for all in-flight requests
        print(f"[d8c] driving done, waiting on {len(tasks)} in-flight reqs...")
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for r in results:
            out_records.append(r)


def summarize(records, total_duration):
    completed = [r for r in records if r.get("ok")]
    failed = len(records) - len(completed)
    total_output = sum(r["output_tokens"] for r in completed)
    rps = len(completed) / total_duration if total_duration > 0 else 0
    tps = total_output / total_duration if total_duration > 0 else 0
    # TPOT = (recv - send) / output_tokens for each req, then mean.
    tpots = []
    for r in completed:
        if r["output_tokens"] > 0:
            tpots.append((r["recv_t"] - r["send_t"]) * 1000 / r["output_tokens"])
    tpots.sort()
    tpot_mean = sum(tpots) / len(tpots) if tpots else 0
    tpot_p50 = tpots[len(tpots) // 2] if tpots else 0
    tpot_p99 = tpots[int(len(tpots) * 0.99)] if tpots else 0
    return {
        "completed": len(completed),
        "failed": failed,
        "total_output_tokens": total_output,
        "duration_s": total_duration,
        "request_throughput": rps,
        "output_throughput": tps,
        "mean_tpot_ms": tpot_mean,
        "p50_tpot_ms": tpot_p50,
        "p99_tpot_ms": tpot_p99,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30077)
    p.add_argument("--duration", type=int, default=300,
                   help="total seconds to drive workload")
    p.add_argument("--phase_s", type=float, default=2.0,
                   help="seconds per phase (KV/mamba), default 2s")
    p.add_argument("--rps", type=int, default=50,
                   help="requests per second")
    p.add_argument("--kv_input_len", type=int, default=1024,
                   help="prompt tokens during KV phase (long → KV-heavy)")
    p.add_argument("--mamba_input_len", type=int, default=64,
                   help="prompt tokens during mamba phase (short → mamba-heavy)")
    p.add_argument("--output_len", type=int, default=64,
                   help="output tokens per req")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True,
                   help="output JSON path (summary)")
    args = p.parse_args()

    records = []
    t0 = time.perf_counter()
    asyncio.run(driver(args, records))
    total_duration = time.perf_counter() - t0

    summary = summarize(records, total_duration)
    summary["records_count"] = len(records)
    summary["args"] = vars(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[d8c] completed={summary['completed']} "
          f"failed={summary['failed']} "
          f"rps={summary['request_throughput']:.2f} "
          f"tps={summary['output_throughput']:.0f} "
          f"tpot={summary['mean_tpot_ms']:.2f}ms "
          f"dur={total_duration:.1f}s")
    print(f"[d8c] wrote summary to {args.out}")


if __name__ == "__main__":
    main()

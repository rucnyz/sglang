"""Phase 1 validation: 3-phase mixed workload.

Drives one running SGLang server through three workload regimes back-to-back so
the dashboard can show pool-pressure transitions:

  Phase A (KV-bound):    random short prompts, no LoRA           -> KV pool builds up
  Phase B (LoRA-bound):  random prompts + skewed LoRA selection  -> LoRA pool saturates
  Phase C (prefix-locality): multi-turn shared-prefix traffic     -> kv_evictable + cache_hit_rate rise

Each phase runs for --phase-seconds seconds. The server must already be running
with --enable-lora and a set of LoRA paths. The sampling client (sample_metrics.py)
should be running concurrently against the same server.

Usage:
    python dev/1/run_validation.py \\
        --host 127.0.0.1 --port 30000 \\
        --tokenizer Qwen/Qwen3-4B \\
        --lora-names lora_0 lora_1 lora_2 lora_3 lora_4 lora_5 lora_6 lora_7 \\
        --phase-seconds 60 \\
        --out dev/1/validation.timeline.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import aiohttp


async def fire_request(session, host, port, prompt, lora_path=None, max_tokens=64):
    body = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0.7},
    }
    if lora_path:
        body["lora_path"] = lora_path
    try:
        async with session.post(f"http://{host}:{port}/generate", json=body, timeout=60) as r:
            await r.read()
            return r.status
    except Exception:
        return -1


async def driver(host, port, phase, kwargs, deadline, timeline_f):
    """Issue requests for `phase` until `deadline`. Logs to timeline."""
    rng = random.Random(kwargs.get("seed", 1))
    rate = kwargs["rate"]
    interval = 1.0 / rate
    sent = 0
    fail = 0
    async with aiohttp.ClientSession() as session:
        next_send = time.time()
        pending = []
        while time.time() < deadline:
            now = time.time()
            if now < next_send:
                await asyncio.sleep(min(0.05, next_send - now))
                continue

            # Build the request payload depending on phase
            if phase == "A":
                # random prompts, no LoRA
                p = " ".join(rng.choices(["the", "quick", "brown", "fox", "jumps", "over",
                                          "lazy", "dog", "and", "runs", "fast"], k=rng.randint(8, 32)))
                lora = None
            elif phase == "B":
                # skewed LoRA: 40% lora_0, 30% lora_1, then uniform
                names = kwargs["lora_names"]
                if not names:
                    lora = None
                else:
                    r = rng.random()
                    if r < 0.4:
                        lora = names[0]
                    elif r < 0.7:
                        lora = names[1]
                    else:
                        lora = rng.choice(names)
                p = " ".join(rng.choices(["llm", "serving", "memory", "manage", "test"], k=rng.randint(8, 32)))
            elif phase == "C":
                # shared-prefix multi-turn: 8 distinct system prompts, each used by many requests
                gid = rng.randint(0, 7)
                prefix = (f"You are assistant {gid}. Always answer concisely. "
                          + "Background: " * 30) [:1024]   # ~1K tokens shared per group
                question = " ".join(rng.choices(["explain", "summarize", "describe", "compare"], k=rng.randint(4, 12)))
                p = f"{prefix}\nUser: {question}"
                lora = None
            else:
                raise ValueError(phase)

            task = asyncio.create_task(fire_request(session, host, port, p, lora,
                                                    max_tokens=kwargs.get("max_tokens", 32)))
            pending.append(task)
            sent += 1
            timeline_f.write(json.dumps({"ts": time.time(), "phase": phase,
                                          "event": "send", "sent": sent}) + "\n")
            timeline_f.flush()
            next_send += interval
            # Keep pending list bounded
            if len(pending) > 256:
                done, pending_set = await asyncio.wait(pending, timeout=0.001,
                                                        return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.result() != 200:
                        fail += 1
                pending = list(pending_set)

        # Drain
        if pending:
            results = await asyncio.gather(*pending, return_exceptions=True)
            for r in results:
                if r != 200:
                    fail += 1
    timeline_f.write(json.dumps({"ts": time.time(), "phase": phase,
                                  "event": "phase_end", "sent": sent, "fail": fail}) + "\n")
    timeline_f.flush()
    print(f"[{time.strftime('%H:%M:%S')}] phase {phase} done: sent={sent} fail={fail}")


async def main_async(args):
    out = open(args.out, "w", buffering=1)

    # Wait for server up (best-effort)
    print(f"[{time.strftime('%H:%M:%S')}] checking {args.host}:{args.port}/health ...")
    async with aiohttp.ClientSession() as s:
        for _ in range(30):
            try:
                async with s.get(f"http://{args.host}:{args.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)

    # Run phases sequentially
    t0 = time.time()
    out.write(json.dumps({"ts": t0, "event": "start"}) + "\n")
    out.flush()

    for phase, rate in (("A", args.rate_a), ("B", args.rate_b), ("C", args.rate_c)):
        deadline = time.time() + args.phase_seconds
        print(f"[{time.strftime('%H:%M:%S')}] phase {phase} for {args.phase_seconds}s @ {rate} rps")
        kwargs = {
            "rate": rate,
            "lora_names": args.lora_names,
            "max_tokens": 32 if phase != "C" else 64,
            "seed": 1,
        }
        await driver(args.host, args.port, phase, kwargs, deadline, out)

    out.write(json.dumps({"ts": time.time(), "event": "end"}) + "\n")
    out.close()
    print(f"\nValidation done. timeline: {args.out}")
    print(f"phase boundaries (relative to t0={t0}):")
    print(f"  A: 0 - {args.phase_seconds}")
    print(f"  B: {args.phase_seconds} - {2*args.phase_seconds}")
    print(f"  C: {2*args.phase_seconds} - {3*args.phase_seconds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--tokenizer", required=False, help="(unused; kept for symmetry)")
    ap.add_argument("--lora-names", nargs="*", default=[],
                    help="LoRA adapter names registered in the server")
    ap.add_argument("--phase-seconds", type=int, default=60)
    ap.add_argument("--rate-a", type=float, default=8, help="phase A request rate (rps)")
    ap.add_argument("--rate-b", type=float, default=8)
    ap.add_argument("--rate-c", type=float, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

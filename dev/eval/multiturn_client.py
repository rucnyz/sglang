#!/usr/bin/env python3
"""Multi-turn agent benchmark client.

Models the long-horizon agent regime from paper §motivation §76:
- N concurrent agent sessions, each accumulating multi-turn history
- Each session: turn-by-turn append messages, send chat completion,
  receive response, append assistant message, advance to next turn
- Session resets when accumulated context exceeds --session-cap-tokens
  (simulates a fresh agent starting)

The client maintains the OpenAI-compatible `messages: [...]` array
per-session, so SGLang's prefix cache hits across turns within a
session and KV usage grows monotonically (= the regime where L2's
mamba_to_kv transfer should help: KV pool fills while mamba pool sits
at one slot per session).

Output: JSONL of per-request metrics + a final summary JSON.
"""

import argparse
import asyncio
import json
import os
import random
import string
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List

import aiohttp


# ---- token helpers ---------------------------------------------------------

def make_user_msg(n_tokens: int, rng: random.Random) -> str:
    """Generate a synthetic user message of approximately n_tokens.

    A typical English/code token is ~4 chars on the Qwen3.5 tokenizer
    (close enough — exact tokenization happens server-side). Use
    randomized lowercase words separated by spaces so the prompt is not
    cacheable across users (each user's session has unique content,
    while within a session the prefix cache hits).
    """
    n_words = max(1, int(n_tokens * 0.75))  # ~0.75 word per token
    words = []
    for _ in range(n_words):
        wlen = rng.randint(3, 8)
        w = "".join(rng.choices(string.ascii_lowercase, k=wlen))
        words.append(w)
    return " ".join(words)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


# ---- per-session state -----------------------------------------------------

@dataclass
class SessionState:
    user_id: int
    history: List[dict] = field(default_factory=list)
    total_tokens: int = 0
    turn_count: int = 0
    sessions_completed: int = 0  # how many session-resets so far


@dataclass
class RequestMetric:
    user_id: int
    turn: int
    total_history_tokens: int
    new_user_tokens: int
    output_tokens: int
    ttft_ms: float
    e2e_ms: float
    error: str = ""


# ---- one chat-completion call ----------------------------------------------

async def chat_call(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    messages: List[dict],
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, float, float, int, str]:
    """Send /v1/chat/completions in streaming mode.

    Returns (response_text, ttft_ms, e2e_ms, output_tokens, error_msg).
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    output_tokens = 0
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return ("", 0.0, 0.0, 0, f"HTTP {resp.status}: {body[:120]}")
            async for line in resp.content:
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload_bytes = line[5:].lstrip()
                if payload_bytes == b"[DONE]":
                    break
                try:
                    data = json.loads(payload_bytes)
                except Exception:
                    continue
                if data.get("error"):
                    return ("", 0.0, 0.0, 0, str(data["error"])[:120])
                # Capture content tokens
                choices = data.get("choices") or []
                if choices and "delta" in choices[0]:
                    delta = choices[0]["delta"]
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000.0
                        text_parts.append(content)
                # Final usage chunk
                if not choices and data.get("usage"):
                    output_tokens = data["usage"].get("completion_tokens", 0)
        e2e = (time.perf_counter() - t0) * 1000.0
        return ("".join(text_parts), ttft or e2e, e2e, output_tokens, "")
    except Exception as e:
        return ("", 0.0, (time.perf_counter() - t0) * 1000.0, 0, str(e)[:120])


# ---- per-user driver --------------------------------------------------------

async def run_user(
    user_id: int,
    api_base: str,
    model: str,
    new_user_tokens_per_turn: int,
    max_output_tokens_per_turn: int,
    session_cap_tokens: int,
    deadline_ts: float,
    metrics: list,
    rng: random.Random,
    session_label: str,
) -> SessionState:
    state = SessionState(user_id=user_id)
    timeout_s = 600.0  # tolerant of long prefills
    async with aiohttp.ClientSession() as session:
        # Optional system prompt — kept simple to avoid skewing prefix cache.
        sys_prompt = (
            f"You are agent {user_id} in long-horizon session "
            f"'{session_label}'. Continue the conversation."
        )
        state.history = [{"role": "system", "content": sys_prompt}]
        state.total_tokens = estimate_tokens(sys_prompt)

        while time.time() < deadline_ts:
            user_msg = make_user_msg(new_user_tokens_per_turn, rng)
            user_tok = estimate_tokens(user_msg)
            messages = list(state.history) + [
                {"role": "user", "content": user_msg}
            ]

            text, ttft, e2e, out_tok, err = await chat_call(
                session, api_base, model, messages,
                max_output_tokens_per_turn, timeout_s,
            )
            metrics.append(RequestMetric(
                user_id=user_id,
                turn=state.turn_count,
                total_history_tokens=state.total_tokens,
                new_user_tokens=user_tok,
                output_tokens=out_tok,
                ttft_ms=ttft,
                e2e_ms=e2e,
                error=err,
            ))
            if err:
                # Brief backoff and continue (server may be admission-pacing)
                await asyncio.sleep(0.5)
                continue

            # Append turn to history
            state.history.append({"role": "user", "content": user_msg})
            state.history.append({"role": "assistant", "content": text})
            state.total_tokens += user_tok + out_tok
            state.turn_count += 1

            # Reset session at cap (simulates a fresh agent starting)
            if state.total_tokens >= session_cap_tokens:
                state.sessions_completed += 1
                state.history = [{"role": "system", "content": sys_prompt}]
                state.total_tokens = estimate_tokens(sys_prompt)
                state.turn_count = 0
    return state


# ---- main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://127.0.0.1:30099")
    p.add_argument("--model", required=True)
    p.add_argument("--num-concurrency", type=int, default=16,
                   help="Number of concurrent agent sessions.")
    p.add_argument("--turn-input-tokens", type=int, default=4096,
                   help="New user-message tokens per turn (synthetic).")
    p.add_argument("--turn-output-tokens", type=int, default=4096,
                   help="Max output tokens per turn (server caps with max_tokens).")
    p.add_argument("--session-cap-tokens", type=int, default=300000,
                   help="Reset session when accumulated context exceeds this.")
    p.add_argument("--max-time-s", type=int, default=300,
                   help="Total benchmark wall-clock seconds.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics: List[RequestMetric] = []
    deadline_ts = time.time() + args.max_time_s
    base_rng = random.Random(args.seed)

    async def driver():
        users = [
            run_user(
                user_id=i,
                api_base=args.api_base,
                model=args.model,
                new_user_tokens_per_turn=args.turn_input_tokens,
                max_output_tokens_per_turn=args.turn_output_tokens,
                session_cap_tokens=args.session_cap_tokens,
                deadline_ts=deadline_ts,
                metrics=metrics,
                rng=random.Random(base_rng.randint(0, 2**31)),
                session_label=f"agent-{i}",
            )
            for i in range(args.num_concurrency)
        ]
        return await asyncio.gather(*users, return_exceptions=False)

    print(f"[multiturn] launching {args.num_concurrency} concurrent agents, "
          f"target session cap {args.session_cap_tokens} tokens, "
          f"duration {args.max_time_s}s", flush=True)
    t0 = time.time()
    states = asyncio.run(driver())
    wall = time.time() - t0

    # Persist metrics
    metrics_path = os.path.join(args.output_dir, "multiturn_metrics.jsonl")
    with open(metrics_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(asdict(m)) + "\n")

    # Compute summary
    valid = [m for m in metrics if not m.error]
    n_total = len(metrics)
    n_valid = len(valid)
    n_errors = n_total - n_valid
    if valid:
        ttfts = sorted(m.ttft_ms for m in valid)
        e2es = sorted(m.e2e_ms for m in valid)
        out_toks = sum(m.output_tokens for m in valid)
        in_toks = sum(m.total_history_tokens + m.new_user_tokens for m in valid)
        def pct(a, p): return a[max(0, min(len(a)-1, int(len(a)*p)))]
        summary = {
            "wall_s": wall,
            "num_concurrency": args.num_concurrency,
            "session_cap_tokens": args.session_cap_tokens,
            "num_requests_total": n_total,
            "num_requests_valid": n_valid,
            "num_errors": n_errors,
            "mean_ttft_ms": sum(ttfts) / len(ttfts),
            "p50_ttft_ms": pct(ttfts, 0.50),
            "p99_ttft_ms": pct(ttfts, 0.99),
            "mean_e2e_ms": sum(e2es) / len(e2es),
            "p50_e2e_ms": pct(e2es, 0.50),
            "p99_e2e_ms": pct(e2es, 0.99),
            "input_tps": in_toks / wall,
            "output_tps": out_toks / wall,
            "sessions_completed_total": sum(s.sessions_completed for s in states),
            "max_session_tokens_observed": max(
                (s.total_tokens for s in states), default=0,
            ),
        }
    else:
        summary = {
            "wall_s": wall,
            "num_requests_total": n_total,
            "num_errors": n_errors,
            "error": "no valid responses",
        }
    summary_path = os.path.join(args.output_dir, "multiturn_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

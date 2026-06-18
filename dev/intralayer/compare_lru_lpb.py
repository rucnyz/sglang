# SPDX-License-Identifier: Apache-2.0
"""LRU vs LPB end-to-end comparison on sglang — mirror of vLLM's
dev/compare_lru_lpb.py.

Runs the SAME phase pipeline (A → B → G → E → F → H → C) against
sglang's offline Engine, with the `--radix-eviction-policy` boot
flag (passed to Engine()) toggling between the recency-LRU baseline
(`--mode lru`) and the hits-per-byte LPB eviction (`--mode lpb`).

Outputs are written to
    dev/intralayer/runs/compare_{mode}{tag}_t{trial}.jsonl
with the same per-row schema as vLLM's driver so a single plotter
can ingest both.

Invocation (mirrors vLLM):
    .venv/bin/python -u dev/intralayer/compare_lru_lpb.py --mode lru \\
        --trial 1 --tag _pathA --util 0.9 --tp 2 --phase-f-scale 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"
# Intel OpenMP (which torch pulls in via MKL) auto-pins the process to a
# single CPU on hosts with hot CPU contention. Disable BEFORE any torch
# import.
os.environ.setdefault("KMP_AFFINITY", "disabled")
# Extend LPB's hits-in-window so the anchor's hit count survives a
# multi-minute run instead of expiring after the 60s default.
os.environ.setdefault("SGLANG_LPB_WINDOW_S", "3600.0")

import torch
from transformers import AutoTokenizer

# Dataset is bundled in the vLLM-side dev/ tree; reuse it directly so
# both engines run on identical cc-burst content.
DATA_VLLM = Path("/scratch/yuzhou/projects/vllm-songyang/dev/cc_long_traces.jsonl")
DATA_LOCAL = Path(__file__).resolve().parent / "cc_long_traces.jsonl"
if DATA_VLLM.exists():
    DATA = DATA_VLLM
elif DATA_LOCAL.exists():
    DATA = DATA_LOCAL
else:
    raise FileNotFoundError("cc_long_traces.jsonl not found")

MODEL = "Qwen/Qwen3.5-35B-A3B"
N_ANCHOR_WARM = 500
N_BURST_SESSIONS = 10
N_TPOT_TOKENS = 20  # decode tokens for the throughput pass


def flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p))
            continue
        t = p.get("type", "")
        if t == "text":
            parts.append(p.get("text", ""))
        elif t == "tool_use":
            inner = p.get("input", "")
            if not isinstance(inner, str):
                inner = json.dumps(inner, ensure_ascii=False)
            parts.append(
                f"<tool_use name={p.get('name', '')} "
                f"id={p.get('id', '')}>{inner}</tool_use>"
            )
        elif t == "tool_result":
            inner = p.get("content", "")
            if isinstance(inner, list):
                inner = "\n".join(
                    x.get("text", str(x)) if isinstance(x, dict) else str(x)
                    for x in inner
                )
            elif not isinstance(inner, str):
                inner = str(inner)
            parts.append(
                f"<tool_result id={p.get('tool_use_id', '')}>"
                f"{inner}</tool_result>"
            )
        else:
            parts.append(json.dumps(p, ensure_ascii=False))
    return "\n".join(parts)


def msg_to_chunk(m: dict) -> str:
    role = m.get("role", "user")
    return f"<|im_start|>{role}\n{flatten_content(m.get('content'))}<|im_end|>\n"


def _result_cached_tokens(r) -> int:
    """sglang's generate() returns either a dict or a list; both expose
    `meta_info["cached_tokens"]` per request."""
    if isinstance(r, list):
        return sum(_result_cached_tokens(x) for x in r)
    meta = r.get("meta_info", {})
    return int(meta.get("cached_tokens", 0) or 0)


def _result_output_tokens(r) -> int:
    if isinstance(r, list):
        return sum(_result_output_tokens(x) for x in r)
    meta = r.get("meta_info", {})
    n = meta.get("completion_tokens")
    if n is None:
        n = meta.get("output_token_count", 0)
    return int(n or 0)


def _result_iter(r):
    """Iterate per-request results regardless of single-vs-batch shape."""
    if isinstance(r, list):
        return r
    return [r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lru", "lpb"], required=True)
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--util", type=float, default=0.9,
                    help="mem_fraction_static. sglang default 0.85-0.9.")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--phase-f-scale", type=int, default=10)
    ap.add_argument("--skip-phase-g", action="store_true",
                    help="Skip Phase G (pre-pressure swarm). Without G, "
                         "the anchor's tree-node last_access_time stays "
                         "anchored at Phase A's warmup, so sglang's "
                         "radix-tree LRU sees it as old after Phase F's "
                         "churn — gives LPB a chance to actually protect "
                         "the anchor while LRU evicts it.")
    ap.add_argument("--two-anchor", action="store_true",
                    help="Phase A warms TWO anchors (A and B). "
                         "Mid-pipeline only re-touches B (Phase B's "
                         "cc-burst gets a B-prefix variant injected; "
                         "Phase G/H optionally only swarm B). At Phase "
                         "H entry, anchor A has high hit_count from "
                         "Phase A but old last_access_time. Under "
                         "sglang LRU: A is evicted by Phase F's "
                         "recency pressure. Under LPB: A is protected "
                         "by its hit count. Phase H queries anchor A "
                         "(NOT the one that was bumped) so the swarm "
                         "TTFT reveals the LPB win.")
    ap.add_argument("--out-dir",
                    default="/scratch/yuzhou/projects/vllm-songyang/dev/intralayer/runs/sglang",
                    help="Directory for jsonl files. Default writes "
                         "into the vllm-songyang dev/intralayer hub.")
    args = ap.parse_args()
    mode = args.mode
    lpb_on = mode == "lpb"
    trial = args.trial
    model_id = args.model
    tag = args.tag
    util = args.util
    tp = args.tp
    pf_scale = args.phase_f_scale
    rng = random.Random(1000 + trial)

    # LPB on/off is the `--radix-eviction-policy` boot flag, passed
    # to Engine() below (#181). Single source of truth across plain /
    # hybrid / hierarchical caches; the old SGLANG_LPB_LRU env toggle
    # was removed from production.
    eviction_policy = "lpb" if lpb_on else "lru"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"compare_{mode}{tag}_t{trial}.jsonl"
    out_jsonl.unlink(missing_ok=True)
    fout = out_jsonl.open("w")
    log = lambda **kw: (fout.write(json.dumps(kw) + "\n"), fout.flush())  # noqa: E731
    log(kind="meta", engine="sglang", mode=mode, trial=trial,
        model=model_id, tag=tag, util=util, tp=tp,
        phase_f_scale=pf_scale,
        radix_eviction_policy=eviction_policy,
        sglang_lpb_window_s=os.environ.get("SGLANG_LPB_WINDOW_S"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    sessions: list[list[dict]] = [
        json.loads(l)["messages"]
        for l in DATA.read_text().splitlines() if l.strip()
    ][:N_BURST_SESSIONS + 1]

    first_user = next(m for m in sessions[0] if m["role"] == "user")
    anchor_ids = tokenizer.encode(
        msg_to_chunk(first_user), add_special_tokens=False
    )
    anchor_len = len(anchor_ids)
    print(f"[{mode}] Anchor A: {anchor_len} tokens")

    # Optional anchor B (--two-anchor mode): same length, totally
    # different content, so it's a separate radix-tree path. Phase A
    # warms both; mid-pipeline only re-touches B; Phase H queries A.
    anchor_b_ids: Optional[list[int]] = None
    if args.two_anchor:
        # Use a separate cc session's first user message as anchor B
        # (session 1 instead of session 0).
        first_user_b = next(m for m in sessions[1] if m["role"] == "user")
        anchor_b_ids = tokenizer.encode(
            msg_to_chunk(first_user_b), add_special_tokens=False
        )
        print(f"[{mode}] Anchor B: {len(anchor_b_ids)} tokens (two-anchor mode)")

    print(f"[{mode}] Loading sglang Engine "
          f"model={model_id} tp={tp} mem_fraction_static={util} "
          f"radix_eviction_policy={eviction_policy}...")

    from sglang import Engine
    engine = Engine(
        model_path=model_id,
        tp_size=tp,
        mem_fraction_static=util,
        max_total_tokens=None,
        trust_remote_code=True,
        log_level="warning",
        radix_eviction_policy=eviction_policy,
    )

    def issue(token_ids, max_new_tokens: int):
        sp = {"max_new_tokens": max_new_tokens, "temperature": 0.0}
        t0 = time.monotonic()
        r = engine.generate(input_ids=list(token_ids), sampling_params=sp)
        wall = time.monotonic() - t0
        return {
            "wall_s": wall,
            "cached": _result_cached_tokens(r),
            "prompt_len": len(token_ids),
            "output_tokens": _result_output_tokens(r),
        }

    def issue_batch(batch_token_ids, max_new_tokens: int):
        sp = {"max_new_tokens": max_new_tokens, "temperature": 0.0}
        t0 = time.monotonic()
        r = engine.generate(
            input_ids=[list(x) for x in batch_token_ids],
            sampling_params=sp,
        )
        wall = time.monotonic() - t0
        rs = _result_iter(r)
        return {
            "wall_s": wall,
            "per_req_cached": [_result_cached_tokens(x) for x in rs],
            "per_req_output_tokens": [_result_output_tokens(x) for x in rs],
            "n_requests": len(rs),
        }

    t_start = time.monotonic()

    # ----- Phase A: warm anchor N_ANCHOR_WARM times ----- #
    # In --two-anchor mode, interleave A and B (alternating) so they
    # both accumulate hits but B's last_access_time ends up newer
    # (B is touched last in the interleave). Phase G (if not skipped)
    # additionally swarms only B, pushing B's recency further ahead.
    print(f"\n[{mode}] Phase A: warming anchor {N_ANCHOR_WARM}x "
          f"{'(both A and B interleaved)' if args.two_anchor else ''} ...")
    for i in range(N_ANCHOR_WARM):
        r = issue(anchor_ids, max_new_tokens=1)
        if anchor_b_ids is not None:
            r_b = issue(anchor_b_ids, max_new_tokens=1)
        if i in (0, 1, N_ANCHOR_WARM // 2, N_ANCHOR_WARM - 1):
            print(f"  warm[{i:>3}] A cached={r['cached']}/{anchor_len}  "
                  f"wall={r['wall_s']*1000:.0f}ms"
                  + (f"  B cached={r_b['cached']}" if anchor_b_ids else ""))
    log(kind="phase", phase="A_done", elapsed_s=time.monotonic() - t_start)

    r = issue(anchor_ids, max_new_tokens=1)
    print(f"\n[{mode}] BASELINE anchor probe: "
          f"cached={r['cached']}/{anchor_len} "
          f"({100*r['cached']/anchor_len:.1f}%)")
    log(kind="anchor_probe", label="baseline",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    # ----- Phase B: cc cold burst ----- #
    print(f"\n[{mode}] Phase B: cc burst over {N_BURST_SESSIONS} sessions; "
          f"each turn issued max_tokens=1 then max_tokens={N_TPOT_TOKENS+1}.")
    for s_idx in range(1, 1 + N_BURST_SESSIONS):
        msgs = sessions[s_idx]
        running_ids: list[int] = []
        prev_len = 0
        turn = 0
        for m in msgs:
            piece = tokenizer.encode(msg_to_chunk(m), add_special_tokens=False)
            running_ids.extend(piece)
            if m.get("role") != "assistant":
                continue
            if len(running_ids) > 60_000:
                break
            r_t = issue(running_ids, max_new_tokens=1)
            r_d = issue(running_ids, max_new_tokens=N_TPOT_TOKENS + 1)
            log(
                kind="cc_turn", session_idx=s_idx, turn=turn,
                prompt_len=len(running_ids),
                ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
                full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
                output_tokens=r_d["output_tokens"],
                new_content_tokens=len(running_ids) - prev_len,
                elapsed_s=time.monotonic() - t_start,
            )
            prev_len = len(running_ids)
            turn += 1
        print(f"  [{mode}] session {s_idx:>2}: {turn} turns done "
              f"(elapsed {time.monotonic() - t_start:.0f}s)")
    log(kind="phase", phase="B_done", elapsed_s=time.monotonic() - t_start)

    # ----- Phase G: PRE-pressure concurrent SWARM ----- #
    # Build the swarm prompts unconditionally so Phase H can use them.
    # In --two-anchor mode, Phase G touches B (the "young" anchor)
    # while Phase H always queries A (the "old" anchor) — so under
    # LRU's recency view A is the stale one that gets evicted, while
    # under LPB A is still protected by its hit count.
    N_SWARM = 30
    g_swarm_anchor = anchor_b_ids if (args.two_anchor and anchor_b_ids is not None) else anchor_ids
    h_swarm_anchor = anchor_ids  # Always A in two-anchor mode
    swarm_prompts: list[list[int]] = []  # Phase G prompts
    h_swarm_prompts: list[list[int]] = []  # Phase H prompts (== G in single-anchor mode)
    for j in range(N_SWARM):
        tail_text = f"\n<|im_start|>user\n[swarm-{j:03d}] continue\n<|im_end|>\n"
        tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
        swarm_prompts.append(g_swarm_anchor + tail_ids)
        h_swarm_prompts.append(h_swarm_anchor + tail_ids)
    swarm_total_prompt = sum(len(p) for p in swarm_prompts)
    if args.skip_phase_g:
        print(f"\n[{mode}] Phase G: SKIPPED (--skip-phase-g). The anchor's "
              "tree-node access time will stay at Phase A's warmup, so "
              "after Phase F it's the oldest non-trivial node.")
        log(kind="phase", phase="G_skipped",
            elapsed_s=time.monotonic() - t_start)
    else:
        print(f"\n[{mode}] Phase G: concurrent swarm "
              f"({N_SWARM} anchored requests, batched).")

        g_ttft = issue_batch(swarm_prompts, max_new_tokens=1)
        g_ttft_cached_total = sum(g_ttft["per_req_cached"])
        print(f"  swarm TTFT batch_wall={g_ttft['wall_s']*1000:.0f}ms  "
              f"sum_cached={g_ttft_cached_total}/{swarm_total_prompt} "
              f"({100*g_ttft_cached_total/swarm_total_prompt:.1f}%)")

        g_full = issue_batch(swarm_prompts, max_new_tokens=N_TPOT_TOKENS + 1)
        g_full_cached_total = sum(g_full["per_req_cached"])
        g_full_output_tokens = sum(g_full["per_req_output_tokens"])
        print(f"  swarm full batch_wall={g_full['wall_s']*1000:.0f}ms  "
              f"throughput={g_full_output_tokens / g_full['wall_s']:.1f} tok/s")

        for j in range(N_SWARM):
            log(
                kind="swarm_turn", j=j,
                prompt_len=len(swarm_prompts[j]),
                ttft_cached=g_ttft["per_req_cached"][j],
                full_cached=g_full["per_req_cached"][j],
                full_output_tokens=g_full["per_req_output_tokens"][j],
                elapsed_s=time.monotonic() - t_start,
            )
        log(
            kind="swarm_batch",
            n_requests=N_SWARM,
            total_prompt_tokens=swarm_total_prompt,
            ttft_batch_wall_s=g_ttft["wall_s"],
            ttft_batch_cached_total=g_ttft_cached_total,
            full_batch_wall_s=g_full["wall_s"],
            full_batch_cached_total=g_full_cached_total,
            full_total_output_tokens=g_full_output_tokens,
            elapsed_s=time.monotonic() - t_start,
        )

    # ----- Phase E: cold-unique random ----- #
    N_COLD = 50
    N_DECOY = 50 * pf_scale
    PROMPT_LEN_COLD = 2048
    print(f"\n[{mode}] Phase E: cold flow ({N_COLD} unique 2K random prompts; "
          f"trial seed = {1000 + trial}).")
    total_tokens = (N_COLD + N_DECOY) * PROMPT_LEN_COLD
    cold_ids = [rng.randint(10, 50_000) for _ in range(total_tokens)]
    for k in range(N_COLD):
        prompt = cold_ids[k * PROMPT_LEN_COLD: (k + 1) * PROMPT_LEN_COLD]
        r_t = issue(prompt, max_new_tokens=1)
        r_d = issue(prompt, max_new_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="cold_turn", k=k,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if k in (0, N_COLD // 2, N_COLD - 1):
            print(f"  cold[{k:>2}] cached={r_t['cached']:>3} "
                  f"wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase F: decoy-warming adversarial ----- #
    N_DECOYS = 5 * pf_scale
    DECOY_LEN_TARGET = 10_000 + 20_000 * (pf_scale > 1)
    N_DECOY_WARM = 100 if pf_scale == 1 else 5
    print(f"\n[{mode}] Phase F: decoy-warming "
          f"({N_DECOYS} decoys × {DECOY_LEN_TARGET}-tok × {N_DECOY_WARM} hits, "
          f"then {N_DECOY} cold prompts).")
    base_phrases = [
        "Alpha vector indices traversal: ",
        "Beta sentinel coordinator dispatch: ",
        "Gamma reduction pipeline metadata: ",
        "Delta consensus quorum tracker: ",
        "Epsilon backpressure buffer manifold: ",
        "Zeta hyperscale ingress shuttle: ",
        "Eta stochastic gradient resonance: ",
        "Theta meridian arbitration loop: ",
        "Iota holographic dispatch fabric: ",
        "Kappa coalesced retrieval pipeline: ",
    ]
    decoys_ids: list[list[int]] = []
    for d_idx in range(N_DECOYS):
        phrase = base_phrases[d_idx % len(base_phrases)]
        decoy_text = phrase + (
            f"decoy-{d_idx}-payload word{rng.randint(0, 999_999)} "
            * max(1500, DECOY_LEN_TARGET // 6 + 100)
        )
        d_ids = tokenizer.encode(decoy_text, add_special_tokens=False)
        d_ids = d_ids[:DECOY_LEN_TARGET]
        decoys_ids.append(d_ids)
        if d_idx in (0, N_DECOYS // 2, N_DECOYS - 1):
            print(f"  decoy[{d_idx}]: {len(d_ids)} tokens")
    for hit in range(N_DECOY_WARM):
        for d_idx in range(N_DECOYS):
            r = issue(decoys_ids[d_idx], max_new_tokens=1)
        if hit in (0, 1, N_DECOY_WARM // 2, N_DECOY_WARM - 1):
            print(f"  decoy_warm[{hit:>3}] last cached={r['cached']}/"
                  f"{len(decoys_ids[-1])} wall={r['wall_s']*1000:.0f}ms")
    log(kind="phase", phase="F_warm_done",
        elapsed_s=time.monotonic() - t_start)
    for k in range(N_DECOY):
        prompt = cold_ids[
            (N_COLD + k) * PROMPT_LEN_COLD: (N_COLD + k + 1) * PROMPT_LEN_COLD
        ]
        r_t = issue(prompt, max_new_tokens=1)
        r_d = issue(prompt, max_new_tokens=N_TPOT_TOKENS + 1)
        log(
            kind="decoy_turn", k=k,
            prompt_len=len(prompt),
            ttft_cached=r_t["cached"], ttft_wall_s=r_t["wall_s"],
            full_cached=r_d["cached"], full_wall_s=r_d["wall_s"],
            output_tokens=r_d["output_tokens"],
            elapsed_s=time.monotonic() - t_start,
        )
        if k in (0, N_DECOY // 2, N_DECOY - 1):
            print(f"  decoy[{k:>2}] cached={r_t['cached']:>3} "
                  f"wall={r_t['wall_s']*1000:.0f}ms")

    # ----- Phase H: POST-pressure SWARM ----- #
    # In --two-anchor mode, swarms anchor A (the one that was NOT
    # bumped by Phase G), so it has old recency under sglang's
    # tree-LRU. LRU should evict A's tree node by recency; LPB
    # should keep it by hit count.
    h_swarm_total_prompt = sum(len(p) for p in h_swarm_prompts)
    print(f"\n[{mode}] Phase H: POST-pressure swarm "
          f"({N_SWARM} anchored requests, batched after Phase F's churn"
          f"{', anchor A' if args.two_anchor else ''}).")

    h_ttft = issue_batch(h_swarm_prompts, max_new_tokens=1)
    h_ttft_cached_total = sum(h_ttft["per_req_cached"])
    print(f"  H swarm TTFT batch_wall={h_ttft['wall_s']*1000:.0f}ms  "
          f"sum_cached={h_ttft_cached_total}/{h_swarm_total_prompt} "
          f"({100*h_ttft_cached_total/h_swarm_total_prompt:.1f}%)")

    h_full = issue_batch(h_swarm_prompts, max_new_tokens=N_TPOT_TOKENS + 1)
    h_full_cached_total = sum(h_full["per_req_cached"])
    h_full_output_tokens = sum(h_full["per_req_output_tokens"])
    print(f"  H swarm full batch_wall={h_full['wall_s']*1000:.0f}ms  "
          f"throughput={h_full_output_tokens / h_full['wall_s']:.1f} tok/s")

    for j in range(N_SWARM):
        log(
            kind="swarm2_turn", j=j,
            prompt_len=len(h_swarm_prompts[j]),
            ttft_cached=h_ttft["per_req_cached"][j],
            full_cached=h_full["per_req_cached"][j],
            full_output_tokens=h_full["per_req_output_tokens"][j],
            elapsed_s=time.monotonic() - t_start,
        )
    log(
        kind="swarm2_batch",
        n_requests=N_SWARM,
        total_prompt_tokens=h_swarm_total_prompt,
        ttft_batch_wall_s=h_ttft["wall_s"],
        ttft_batch_cached_total=h_ttft_cached_total,
        full_batch_wall_s=h_full["wall_s"],
        full_batch_cached_total=h_full_cached_total,
        full_total_output_tokens=h_full_output_tokens,
        elapsed_s=time.monotonic() - t_start,
    )

    # ----- Phase C: FINAL anchor probe ----- #
    r = issue(anchor_ids, max_new_tokens=1)
    pct = 100 * r["cached"] / anchor_len
    print(f"\n[{mode}] FINAL anchor probe (post-E/F/H): "
          f"cached={r['cached']}/{anchor_len} ({pct:.1f}%)")
    log(kind="anchor_probe", label="final",
        cached=r["cached"], anchor_len=anchor_len, wall_s=r["wall_s"],
        elapsed_s=time.monotonic() - t_start)

    fout.close()
    print(f"\n[{mode}] Done (trial {trial}). "
          f"{time.monotonic() - t_start:.0f}s total. Log: {out_jsonl}")

    try:
        engine.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()

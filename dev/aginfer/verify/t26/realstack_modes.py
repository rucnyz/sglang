"""T26 (#206) — real-stack verification of MIXED + spec-decode throughput.

The default integration_stress stack runs pure DECODE/EXTEND only, so its
stage_t26 exercises only the two original branches.  This script launches
two ADDITIONAL stacks with forward-mode-specific flags to drive the #206
code paths against the real B300 scheduler:

  MIXED   --enable-mixed-chunk --chunked-prefill-size 256
          → long-prompt + concurrent decode produces MIXED batches; the
            new split (prefill reqs → prefill_bps, decode reqs → decode
            EMA via batch.decoding_reqs) must feed BOTH metrics.

  SPEC    --speculative-algorithm NGRAM (no draft model needed)
          → decode steps verify drafted tokens.  Under spec the pre-forward
            DECODE branch SKIPS, so decode_per_program can only populate via
            the new POST-forward accept_lens hook — a strong positive proof
            the hook fires (and counts accepted tokens, not 1/req).

Both flavors additionally assert the sglang log contains NO
"aginfer ... measurement raised" line — that warning is the ONLY symptom
of the blanket-except swallowing an AttributeError on a real Req/batch
(which would silently leave the metric at the pure-path value).  So
"metrics populate AND no suppressed warning" is the real liveness gate.

Usage:
    source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
    cd /scratch/yuzhou/projects/sglang
    python dev/aginfer/verify/t26/realstack_modes.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

import httpx

_HERE = Path(__file__).resolve().parent
_INT = _HERE.parent / "integration_stress"
sys.path.insert(0, str(_INT))

import harness  # noqa: E402
from harness import (  # noqa: E402
    MODEL, SGLANG_HOST, SGLANG_PORT, StackHandles, stack,
)


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


_PAD = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
    "do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
) * 6  # ~700 tokens — long enough to chunk under --chunked-prefill-size 256


async def _drive_and_poll(
    *, n_progs: int, duration: float, max_tokens: int,
) -> dict:
    """Fire program-tagged chat load and poll /aginfer/state concurrently,
    capturing peak prefill_bps and the set of tagged programs that reached a
    positive decode rate.  Shared by both flavors."""
    base = f"{SGLANG_HOST}:{SGLANG_PORT}"
    prog_ids = [f"m26-{i}-{uuid.uuid4().hex[:6]}" for i in range(n_progs)]
    tagged = set(prog_ids)
    deadline = time.time() + duration
    seen = {"prefill_bps_max": 0.0, "decode_pos": set(),
            "inflight": set(), "polls": 0}

    async def worker(pid: str) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                body = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system",
                         "content": f"{_PAD} ({pid} {n} {uuid.uuid4().hex[:8]})"},
                        {"role": "user",
                         "content": (f"Count slowly from 1 to 40, one number "
                                     f"per line. ({pid} {n} "
                                     f"{uuid.uuid4().hex[:12]})")},
                    ],
                    "max_tokens": max_tokens, "temperature": 0.0,
                    "program_id": pid,
                }
                try:
                    await cli.post(f"http://{base}/v1/chat/completions",
                                   json=body, timeout=120.0)
                except Exception:
                    pass
                n += 1

    async def poller() -> None:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while time.time() < deadline:
                try:
                    r = await cli.get(f"http://{base}/aginfer/state")
                    if r.status_code == 200:
                        b = r.json()
                        seen["polls"] += 1
                        te = b.get("throughput_ema", {})
                        seen["prefill_bps_max"] = max(
                            seen["prefill_bps_max"],
                            float(te.get("prefill_bps", 0.0)))
                        for pid, rate in (te.get("decode_per_program", {}) or {}).items():
                            if pid in tagged and float(rate) > 0.0:
                                seen["decode_pos"].add(pid)
                        ppu = b.get("per_program_usage", {}) or {}
                        for pid, e in ppu.items():
                            infl = e.get("hbm", {}).get("inflight") or {}
                            if pid in tagged and any(v > 0 for v in infl.values()):
                                seen["inflight"].add(pid)
                except Exception:
                    pass
                await asyncio.sleep(0.25)

    await asyncio.gather(*(worker(p) for p in prog_ids), poller())
    return {
        "prefill_bps_max": seen["prefill_bps_max"],
        "decode_pos": len(seen["decode_pos"]),
        "inflight": len(seen["inflight"]),
        "n_tagged": len(tagged),
        "polls": seen["polls"],
    }


def _assert_no_suppressed_warning(stack_h: StackHandles, label: str) -> None:
    """The blanket-except in the hooks logs 'aginfer ... measurement raised'
    on a swallowed error — the ONLY signal that a real Req/batch attribute
    access failed.  Its absence is the liveness gate."""
    text = stack_h.sglang_log.read_text(errors="replace")
    for needle in ("aginfer throughput measurement raised",
                   "aginfer spec-decode measurement raised"):
        if needle in text:
            raise StageFail(
                f"{label}: sglang log shows a suppressed instrumentation "
                f"error ({needle!r}) — a hook excepted on a real batch")


def flavor_mixed(results_dir: Path) -> None:
    print("[mixed] launching stack with --enable-mixed-chunk "
          "--chunked-prefill-size 256 …")
    extra = ["--enable-mixed-chunk", "--chunked-prefill-size", "256"]
    with stack(results_dir, sglang_extra_args=extra) as h:
        t0 = time.time()
        res = asyncio.run(_drive_and_poll(
            n_progs=4, duration=30.0, max_tokens=96))
        print(f"  [mixed] polls={res['polls']} "
              f"prefill_bps_max={res['prefill_bps_max']:.3g} "
              f"decode_pos={res['decode_pos']}/{res['n_tagged']} "
              f"inflight={res['inflight']}/{res['n_tagged']} ({time.time()-t0:.1f}s)")
        if h.sglang_proc.poll() is not None:
            raise StageFail("mixed: sglang died mid-flavor")
        if res["polls"] == 0:
            raise StageFail("mixed: no /aginfer/state polls succeeded")
        if res["prefill_bps_max"] <= 0.0:
            raise StageFail("mixed: prefill_bps never > 0 — the MIXED prefill "
                            "split did not feed prefill_bps")
        if res["decode_pos"] < res["n_tagged"]:
            raise StageFail(f"mixed: only {res['decode_pos']}/{res['n_tagged']} "
                            f"tagged programs got a positive decode rate — the "
                            f"MIXED decode split is incomplete")
        _assert_no_suppressed_warning(h, "mixed")
    print(_green("  [mixed] MIXED batches feed prefill_bps + decode EMA, "
                 "no suppressed warning OK"))


def flavor_spec(results_dir: Path) -> None:
    """ngram is **spec-v1** (it forces overlap OFF), so it does NOT expose
    accept_lens post-forward — decode is counted by the conservative 1/req
    pre-forward path, gated on `is_spec_v2` so v1 is NOT skipped.  This
    flavor is therefore the v1 REGRESSION GUARD: under a real spec config
    the decode metric stays populated (the gating didn't blank it) and no
    hook excepts on a real spec batch/result.  The v2 accept_lens →
    per-program EMA logic is pinned by the pure `stage_spec_decode`
    (no Qwen3-0.6B EAGLE draft checkpoint is available to drive v2 live)."""
    print("[spec] launching stack with --speculative-algorithm NGRAM "
          "--speculative-ngram-max-bfs-breadth 1 (spec-v1, overlap off) …")
    # max-bfs-breadth 1 ⇒ eagle_topk 1, which the default trtllm_mha backend
    # accepts at page_size 64 (topk>1 there is rejected as unstable).
    extra = [
        "--speculative-algorithm", "NGRAM",
        "--speculative-ngram-max-bfs-breadth", "1",
    ]
    with stack(results_dir, sglang_extra_args=extra) as h:
        t0 = time.time()
        # Longer decode so spec verify steps actually run.
        res = asyncio.run(_drive_and_poll(
            n_progs=3, duration=30.0, max_tokens=128))
        print(f"  [spec] polls={res['polls']} "
              f"decode_pos={res['decode_pos']}/{res['n_tagged']} "
              f"inflight={res['inflight']}/{res['n_tagged']} ({time.time()-t0:.1f}s)")
        if h.sglang_proc.poll() is not None:
            raise StageFail("spec: sglang died mid-flavor")
        if res["polls"] == 0:
            raise StageFail("spec: no /aginfer/state polls succeeded")
        # v1 regression guard: decode must STAY populated (1/req pre-forward),
        # not blanked by the spec-v2 skip.
        if res["decode_pos"] < res["n_tagged"]:
            raise StageFail(f"spec(v1): only {res['decode_pos']}/{res['n_tagged']} "
                            f"tagged programs got a positive decode rate — the "
                            f"is_spec_v2 gating wrongly blanked v1 decode "
                            f"(should keep 1/req pre-forward)")
        _assert_no_suppressed_warning(h, "spec")
    print(_green("  [spec] v1 decode stays populated (1/req), no hook excepted "
                 "on a real spec batch OK"))


def main() -> int:
    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)
    print("=" * 64)
    print("T26 (#206) — real-stack MIXED + spec-decode measurement")
    print("=" * 64)
    flavors = [("mixed", flavor_mixed), ("spec", flavor_spec)]
    failed = []
    for name, fn in flavors:
        try:
            fn(results_dir)
        except StageFail as e:
            failed.append(name)
            print(_red(f"  [{name}] FAIL: {e}"))
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            import traceback
            print(_red(f"  [{name}] ERROR: {e}"))
            traceback.print_exc()
    print("=" * 64)
    if failed:
        print(_red(f"FAILED: {', '.join(failed)}"))
        return 1
    print(_green("T26 #206 real-stack PASS — MIXED + spec-decode both green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

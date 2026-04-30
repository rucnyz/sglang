"""Job manifest for the Layer 2 regression+benefit suite.

5 workloads × 2 arms = 10 jobs. Each job runs a workload script that
boots its own server, runs the workload, and writes a metrics.json.
"""
from __future__ import annotations
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run import Job  # type: ignore


# Common env vars for the prelude arm. SGLANG_XPOOL_EDGE_TRIGGER=1
# forces edge-triggered planning so the suite tests THAT specific
# implementation. We also enable adaptive K_BIG (proven not to regress).
PRELUDE_ENV = {
    "SGLANG_HPB_LRU": "1",
    "SGLANG_HPB_WINDOW_S": "120.0",
    "SGLANG_K_BIG": "8192",
    "SGLANG_K_BIG_AUTO_THRESHOLD": "0.5",
    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
    "SGLANG_ARENA_SHARED": "1",
    "SGLANG_ARENA_FROM_BLOB": "1",
    "SGLANG_ARENA_CHUNK_BYTES": str(1024**3),
    "SGLANG_BUDGETER": "1",
    "SGLANG_BUDGETER_XPOOL_PLANNER": "1",
    "SGLANG_BUDGETER_XPOOL_COORDINATED": "1",
    "SGLANG_BUDGETER_TICK_S": "2.0",
    "SGLANG_XPOOL_KV_HIGH": "0.85",
    "SGLANG_XPOOL_KV_LOW": "0.40",
    "SGLANG_XPOOL_MAMBA_HIGH": "0.80",
    "SGLANG_XPOOL_MAMBA_LOW": "0.40",
    "SGLANG_XPOOL_COOLDOWN": "2",
    "SGLANG_XPOOL_EDGE_TRIGGER": "1",  # the property under test
    # With arena memory transparency (path A) landed in
    # model_runner_kv_cache_mixin._profile_available_bytes, mem_fraction=0.8
    # is now safe with the full Layer 2 stack — the arena headroom is taken
    # from the (1-mem_fraction)·pre reserve band, so KV+mamba get the same
    # budget as baseline at the same mem_fraction.
    "MEM_FRACTION": "0.8",
}

BASELINE_ENV = {
    "MEM_FRACTION": "0.8",
}


def build_manifest() -> list[Job]:
    suite_dir = SCRIPT_DIR
    workloads_dir = suite_dir / "workloads"
    jobs: list[Job] = []

    # R1: steady-state random workload at fixed mamba_full_memory_ratio.
    # Tests: Layer 2 must NOT touch a workload that doesn't shift binding pool.
    # Pass: prelude TPS in [97%, 103%] of baseline.
    base_port = 32000
    def make(name, arm, runner, env, port):
        return Job(name=name, workload=name.split("_")[0],
                   arm=arm, runner=str(workloads_dir / runner),
                   port=port, extra_env=env,
                   pass_metric="input_tps",
                   pass_min_relative=0.95, pass_max_relative=1.05)

    jobs.append(make("R1_steady_random", "baseline", "r1_steady_random.sh",
                     BASELINE_ENV, base_port + 0))
    jobs.append(make("R1_steady_random", "prelude", "r1_steady_random.sh",
                     PRELUDE_ENV, base_port + 1))

    jobs.append(make("R2_steady_gsp", "baseline", "r2_steady_gsp.sh",
                     BASELINE_ENV, base_port + 2))
    jobs.append(make("R2_steady_gsp", "prelude", "r2_steady_gsp.sh",
                     PRELUDE_ENV, base_port + 3))

    # R3 LoRA: re-added with MAMBA_FLAGS="" override (qwen3-mamba-only flags
    # break the LoRA Triton dispatch on Qwen3-4B). Tests that L2 stays
    # silent on non-mamba workloads (mamba_pool doesn't exist → 0 transfers).
    jobs.append(make("R3_lora", "baseline", "r3_lora.sh",
                     BASELINE_ENV, base_port + 4))
    jobs.append(make("R3_lora", "prelude", "r3_lora.sh",
                     PRELUDE_ENV, base_port + 5))

    # B1: phase-shift cyclic (mamba-heavy ↔ KV-heavy). Continuous traffic, no drains.
    # Pass: prelude TPS ≥ baseline + 5% OR median E2E ≤ 95% of baseline (whichever).
    j_b1_bl = Job(name="B1_phase_shift", workload="B1_phase_shift", arm="baseline",
                   runner=str(workloads_dir / "b1_phase_shift.sh"),
                   port=base_port + 6, extra_env=BASELINE_ENV,
                   pass_metric="median_e2e_ms",
                   pass_min_relative=0.0, pass_max_relative=2.0)  # baseline always passes
    j_b1_pr = Job(name="B1_phase_shift", workload="B1_phase_shift", arm="prelude",
                   runner=str(workloads_dir / "b1_phase_shift.sh"),
                   port=base_port + 7, extra_env=PRELUDE_ENV,
                   pass_metric="median_e2e_ms",
                   pass_min_relative=0.0, pass_max_relative=0.95)  # ≤ 95% of baseline
    jobs.append(j_b1_bl); jobs.append(j_b1_pr)

    # B2: cold-burst — already a Q3.B-style workload but here we measure
    # full-system Recovery TTFT must be ≤ 105% of baseline (i.e., L2 doesn't
    # break the Q3.B HPB recovery win when stacked).
    j_b2_bl = Job(name="B2_cold_burst", workload="B2_cold_burst", arm="baseline",
                   runner=str(workloads_dir / "b2_cold_burst.sh"),
                   port=base_port + 8, extra_env=BASELINE_ENV)
    j_b2_pr = Job(name="B2_cold_burst", workload="B2_cold_burst", arm="prelude",
                   runner=str(workloads_dir / "b2_cold_burst.sh"),
                   port=base_port + 9, extra_env=PRELUDE_ENV,
                   pass_metric="mean_ttft_ms", pass_max_relative=1.05)
    jobs.append(j_b2_bl); jobs.append(j_b2_pr)
    return jobs

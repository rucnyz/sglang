# case1: long-horizon agents (KV-bound)

The 40 longest-context cc sessions (input_ids up to ~276k), replayed at
concurrency 64. Long prompts plus accumulating multi-turn context saturate KV,
while each session needs only one recurrent slot, so mamba sits idle. The waste
is the idle mamba while KV is the wall.

```bash
bash ../serve.sh 0 30098            # baseline sglang on GPU 0, port 30098
bash ../replay.sh 30098 case1 results
../../../.venv/bin/python ../parse_waste.py results/server.log --out results --label "case1 baseline"
# or: bash ../run_case.sh case1 0 30098
```

Split: default (mamba pool over-provisioned, so long-horizon wastes it).
Data: `data/trace.jsonl` (from `../make_slices.py`).

Measured (baseline): backlogged 98% of ticks, KV the wall, **mamba 92% idle =
wasted** (`wasted_pool: mamba`), queue p99 55. The idle mamba is borrowable to KV.

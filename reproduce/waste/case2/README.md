# case2: large-concurrency short Q&A (mamba-bound)

Short requests built by truncating each cc session's first turn to a 512-token
question + 128-token forced answer, all arriving at t=0, replayed at concurrency
256. Served on sglang's DEFAULT split (no `--max-mamba-cache-size`; the optional
`--mamba-full-memory-ratio` arg sweeps the split). The swarm starves on mamba
SLOTS: `max_running = mamba_pool/3` caps the running batch, so requests queue
while their tiny contexts leave KV idle. The bound is mamba slots (via
max_running), not mamba byte-usage, so mamba peaks at the 1/3 cap and KV sits idle.

```bash
bash ../serve.sh 0 30098                      # baseline sglang, default split
bash ../replay.sh 30098 data/trace.jsonl 256 0 results case2
../../../.venv/bin/python ../parse_waste.py results/server.log --out results --label "case2 baseline"
# or: bash ../run_case.sh case2 0 30098
```

Measured (baseline, default split): KV peak 2%, mamba peak 33% (the max_running
cap), so **KV is the wasted pool** (`wasted_pool: kv`, 23 GB idle). The idle KV is
borrowable to mamba (raising max_running drains the queue). Sweeping the split
toward mamba shrinks KV idle (10.8 GB at ratio 3.0, 7.0 GB at 5.0) but it never
bottoms out: the swarm binds on concurrency, not bytes. Data: `data/trace.jsonl`
(from `../make_slices.py`).

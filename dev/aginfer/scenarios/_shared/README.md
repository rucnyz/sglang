# _shared — scripts and parsers used by multiple scenarios

| file | purpose |
|---|---|
| `run_k.sh` | core variant runner (`full / ka / J / kv_off / a3`); used by repro.sh in workload scenarios |
| `run_thunderagent.sh` | launches sglang (LRU) + TA proxy + harbor → :9200 |
| `run_lru.sh` | launches sglang (LRU) + harbor → :30000 |
| `run_matrix.sh` | N=3 baseline matrix orchestrator (B/O alternation) |
| `run_extend_2cycle.sh` | extends an existing matrix by 2 cycles |
| `run_instrument_chain.sh` | chains a sequence of variants for structured-log evidence |
| `parse_daemon_events.py` | aggregate kv_decide / migrate / admission events from daemon.log |
| `parse_matrix.py` | per-cycle wall-time aggregation + Welch t-test |
| `parse_4arm.py` | 4-arm matrix aggregation |
| `parse_ttft.py` | per-request TTFT distribution |

Methodology (N≥3 protocol, B/O alternation, acceptance rule) lives in
[`../../PLAN.md`](../../PLAN.md) §5.

# Audit: cost-model state vs Admitter needs

## 1. `c^xfer(X)` EWMA — scaffold present, NOT wired

- Class `RuntimeActuatorCost` at `cost_model.py:218-272`
- α default **0.3** (matches design), half-life ≈ 1.6 fires
- Init `3000 us/chunk`; env: `SGLANG_XPOOL_NB_CHUNK_COST_INIT_US`,
  `SGLANG_XPOOL_NB_CHUNK_COST_EWMA_ALPHA`
- Consumer wired: `xpool_planner.py:380-385` reads
  `runtime_cost.current_us` for the gate threshold
- **Producer side missing**: `RuntimeActuatorCost.update()` is never
  called from production code (only from D9b tests)
- Fire path captures `result.total_us`, `result.granted_pages` at
  `agent.py:627,824` but never feeds them to the EWMA
- **Admitter must add**: a single
  `get_runtime_actuator_cost().update(result.total_us, result.granted_pages)`
  after `execute_async` returns (agent.py:~824)

## 2. `c^evict_i(X)` — missing entirely

Cannot be implemented today without new code. See
`audit_radix_evict.md` for the radix-cache surface and proposed
incremental sorted-index design.

## 3. `c_i(s)` re-prefill wall — interface exists, no Stage-0 probe

- `CostCurves` dataclass at `cost_model.py:26-80` provides
  `c_kv_ms(L)` and `c_m_ms(L)` (closed-form `α·L² + β·L + γ` for
  KV, `α·L + β` for mamba)
- Loaders at lines 96-152 read from `SGLANG_CSIGMA_*` env vars or
  `SGLANG_CSIGMA_JSON`
- **No Stage-0 boot probe code exists** (`dev/eval/cost_model/`
  directory does not exist)
- Fallback: `BUILTIN_DEFAULT` constants (lines 85-93,
  Qwen3.5-35B-A3B / H200 hardcoded)
- → On other deployments this is effectively a constant

## 4. `w_q` queue penalty — hardcoded constant

- `pressure_adapter.py:167-172`: `queue_wait_us = 100us/req` default
- Env override: `SGLANG_XPOOL_QUEUE_WAIT_US`
- Not Stage-0 calibrated

## 5. Cost-model interface — does not exist in Admitter shape

`cost_model.py` exports only `get_cost_curves()` and
`get_runtime_actuator_cost()`. Proposed Admitter facade:

```python
class CostModel:
    def c_xfer(self, n_bytes: int) -> float
    def c_evict(self, pool: str, n_bytes: int) -> float
    def c(self, pool: str, s_tokens: int) -> float
    def w_q_us(self) -> float
```

## 6. Hardcoded constants to flag

- `pressure_adapter.py:120-124` `_DEFAULT_*_RECOVER_L=2048/6144`
- `pressure_adapter.py:165-178` `pause_penalty_us=1000`,
  `queue_wait_us=100`, `persist_tick_us=5000`, `edge_us=100000`
- `cost_model.py:85-93` BUILTIN_DEFAULT cost curves (KV-only, one HW)
- `cost_model.py:300` `initial=3000us/chunk` cold-start cost
- `xpool_planner.py:82` `nb_chunk_cost_us=5000.0`, `nb_margin=1.5`
- `agent.py:131-132` `_n_pages_per_fire=4`

## 7. Conservative boot gate — half-wired

- Gate exists: `cost_model.py:245-249` `is_calibrated` returns
  `self._n_observations >= 3`
- Consumer: `xpool_planner.py:382-385` uses
  `max(runtime_cost.current_us, c.nb_chunk_cost_us)` when not
  calibrated
- **Since `update()` is never called, `_n_observations` stays at 0
  forever** — the gate is decorative until producer-side wiring (§1)
  is added

## Summary for Admitter

| Need | Status | File:line |
|---|---|---|
| `c^xfer` EWMA storage/α | working (read-only) | cost_model.py:218-272 |
| `c^xfer` producer | **missing — Admitter must add** | agent.py:~824 |
| `c^evict_i(X)` | **missing entirely** | (see audit_radix_evict) |
| `c_i(s)` curves struct | exists | cost_model.py:26-80 |
| `c_i(s)` Stage-0 probe | **missing** | — |
| `w_q` | hardcoded `100 us/req` | pressure_adapter.py:171 |
| `cost_model.c_*` facade | **missing** | — |
| `≥3 obs` boot gate | scaffold exists, never trips | cost_model.py:245-249 |
| `admitter.py`, `pressure_planner.py` | **do not exist** | — |

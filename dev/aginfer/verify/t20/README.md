# T20 — `POST /aginfer/migrate` residence-set payload (DESIGN §6)

Impl_PLAN.md §3 T20.  Replaces the legacy `{hash, target_tier}` action with
the residence-set form `{hash, add_tiers, remove_tiers, action_id}` per
DESIGN §6 + the 6 transitions in §7 transfer-window semantics.

## WHAT WE PROMISED

**Wire payload.**

Daemon sends:

```json
[
  {
    "hash": "<unit hash>",
    "add_tiers":    ["HBM" | "DRAM" | "DISK", ...],
    "remove_tiers": ["HBM" | "DRAM" | "DISK", ...],
    "action_id":    "<opaque correlator from daemon, echoed in skip>"
  },
  ...
]
```

Sglang returns:

```json
{
  "applied":         <int>,
  "applied_hashes":  [<hash>, ...],
  "skipped":         [{"hash": ..., "action_id": ..., "reason": ...}]
}
```

**Per-tier semantics** (each tier in `add_tiers` / `remove_tiers`
maps to one sglang primitive):

| Tier in `add_tiers`    | Operation |
|---|---|
| `DRAM`  | `write_backup(node)` — synthesize host copy from device |
| `HBM`   | `load_back(node)` — promote host copy to device |
| `DISK`  | not yet wired; skip with `disk_tier_not_yet_wired` (Mooncake L3 lands in a future T) |

| Tier in `remove_tiers` | Operation |
|---|---|
| `HBM`   | `evict_component(target=DEVICE)` |
| `DRAM`  | `evict_component(target=HOST)` |
| `DISK`  | noop (DISK not yet populated) |

If `remove_tiers` would empty the residence (post-add) the unit is
DROPped: `evict_component(target=ALL)` + `_remove_leaf_from_parent`
+ `_iteratively_delete_tombstone_leaf`.  Non-leaf nodes can't be
DROPped (would break prefix matching); skip with `remove_not_leaf`.

**Order of operations within one action**: `add_tiers` applied first,
then `remove_tiers`.  Rationale: synthesise the new copy BEFORE
freeing the old one, else data loss on the `{HBM} → {DRAM}` path
(would need write_backup before evict device).  Even on `{DRAM} →
{HBM}` the order is harmless (load_back is read-only on the host
side).

**Skip-reason classes** (DESIGN §6 round-9):

| Reason | Trigger |
|---|---|
| `not_in_tree` | hash not in the `{hash → node}` DFS map |
| `already_acted_this_batch` | duplicate hash within this POST |
| `add_already_present:<tiers>` | a tier in `add_tiers` is already in residence |
| `remove_already_absent:<tiers>` | a tier in `remove_tiers` is not in residence |
| `remove_not_leaf` | the remove would DROP a non-leaf node |
| `write_through_declined:<sub_reason>` | `write_backup` returned 0 / raised |
| `promote_load_back_declined:<category>` | `load_back` returned False |
| `promote_raised:<Exc>:<loc>:<msg>` | `load_back` raised |
| `disk_remove_unsupported_upstream` | `DISK` in `remove_tiers` (no delete API on any storage backend) |
| `disk_add_declined:no_storage_backend` \| `:not_host_backed` | `DISK` in `add_tiers`, storage unavailable / node not DRAM-backed |
| `disk_backup_raised:<Exc>:<loc>:<msg>` | `write_backup_storage` raised |
| `disk_add_conflicts_with_dram_add` | `DISK` and `DRAM` both in `add_tiers` of the SAME action (would race `write_backup`'s async D→H copy against `write_backup_storage`'s read of the same host buffer) |
| `disk_add_conflicts_with_dram_remove` | `DISK` in `add_tiers` and `DRAM` in `remove_tiers` of the SAME action (would race the DRAM removal freeing the host buffer against `write_backup_storage`'s in-flight read of it) |

Every skip entry carries the action's `action_id` so the daemon's
`APPLY_FAILED` event handler (T37) can correlate to the originating
joint_decide.

**Idempotency** (DESIGN §10 R2).  Re-applying the same action
returns 200 with `applied=0` and the per-action skip reason
(`add_already_present` or `remove_already_absent`).  No state change.

## WORST CASE

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Race: tree mutated mid-batch | Send a hash, then `flush_cache`, then send another action against same hash | second action → `not_in_tree` | Stage 7 |
| Duplicate hash in same batch | Same hash appears twice in actions[] | first applies, second → `already_acted_this_batch` | Stage 5 |
| DISK in add_tiers | `add_tiers=["DISK"]` | applies, or `disk_add_declined:<sub_reason>` | Stage 8 |
| DISK in remove_tiers | `remove_tiers=["DISK"]` | `disk_remove_unsupported_upstream` | Stage 12 |
| DISK+DRAM added together | `add_tiers=["DRAM","DISK"]` | `disk_add_conflicts_with_dram_add` | Stage 13 |
| DISK added, DRAM removed together | `add=["DISK"], remove=["DRAM"]` | `disk_add_conflicts_with_dram_remove` | Stage 13 |
| Idempotency | Apply `add=[DRAM]` twice on the same node | second → `add_already_present:DRAM` | Stage 6 |
| DROP a non-leaf | `remove=[HBM, DRAM]` on internal node | `remove_not_leaf` | Stage 3b |
| Add HBM on host-only node | `add=[HBM]` on `{DRAM}` residence | calls `load_back`; applies or returns `promote_load_back_declined:<cat>` (no crash) | Stage 4 |

## HOW WE VERIFY

`verify/t20/verify.py` runs against a live sglang launched with
HiCache write_through, walks the §7 transitions, and asserts the
post-state residence at each step via `GET /aginfer/state`:

```
Stage 0  Schema sanity
         POST with a single tagged prefill action returns the
         envelope {applied, applied_hashes, skipped[*]} with
         every skip carrying action_id.

Stage 1  add=[DRAM] (write_through)
         Drive a tagged prefill on a long unique prefix.  Force
         the unit to HBM-only via /flush_cache.  POST
         add=[DRAM]; verify residence == [HBM, DRAM] after.

Stage 2  remove=[HBM] from {HBM, DRAM}
         Stage 1 leaves the unit on both tiers; POST
         remove=[HBM]; verify residence == [DRAM] after.

Stage 3  remove=[HBM, DRAM] (DROP)
         Drive a fresh tagged unit; POST remove=[HBM, DRAM];
         verify unit absent from `units[]`.  3b: try the same on
         the system-prompt prefix (non-leaf) → expect
         `remove_not_leaf`.

Stage 4  add=[HBM] (load_back)
         From Stage 2 the unit is DRAM-only.  POST add=[HBM];
         verify residence ⊇ {HBM}.  load_back may decline at
         small mem-fraction; assert pass OR
         `promote_load_back_declined:<cat>` (not a crash).

Stage 5  Duplicate hash in batch
         POST [{h, add=[DRAM]}, {h, add=[DRAM]}]; response:
         applied=1, second skip = `already_acted_this_batch`.

Stage 6  Idempotency (DESIGN §10 R2)
         Replay Stage 1's add=[DRAM]; expect `applied=0` +
         `add_already_present:DRAM`.

Stage 7  Unknown hash
         POST {hash: "node-99999999", add=[DRAM]}; expect
         applied=0 + `not_in_tree`.

Stage 8  DISK in add_tiers
         POST {hash: <real>, add=[DISK]}; expect
         `disk_tier_not_yet_wired`.

Stage 9  action_id echo
         Send 3 actions with distinct action_ids; assert each
         skip's action_id == the originating action's.

Stage 12 remove=[DISK]
         POST remove=[DISK] on a real unit; expect
         `disk_remove_unsupported_upstream` and the unit
         untouched (still present, same residence).

Stage 13 DISK add-side conflicts
         POST add=[DRAM,DISK] on a nonexistent hash; expect
         `disk_add_conflicts_with_dram_add`.  POST
         add=[DISK],remove=[DRAM] on a nonexistent hash; expect
         `disk_add_conflicts_with_dram_remove`.  Both reject on
         tier-set shape alone, before hash resolution (review
         PR #4, discussion_r3921269467).
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang

SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=5 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
  > dev/aginfer/logs/sglang_t20.log 2>&1 &
SGLANG_PID=$!
until grep -q "Uvicorn running" dev/aginfer/logs/sglang_t20.log; do sleep 3; done
sleep 18  # JIT settle

AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python dev/aginfer/verify/t20/verify.py

kill "$SGLANG_PID"
```

## RESULTS

**PASSED** — all 10 stages on fresh Qwen3-0.6B + HiCache write_through.

* date: 2026-05-31
* model: Qwen/Qwen3-0.6B, flashinfer backend, HiCache write_through,
  cap 64 K tokens, GPU 5
* lines added: ~240 in `unified_radix_cache.py::apply_aginfer_migrations`
  (new payload + cascade-evict + skip-reason classes); 7 lines in
  `http_server.py` (docstring); 1 new test file at `verify/t20/verify.py`

| Stage | Result |
|---|---|
| 0  envelope shape + action_id echo | PASS |
| 1  add=[DRAM] (write_backup) | PASS — residence ['HBM', 'DRAM'] |
| 2  remove=[HBM] from {HBM, DRAM} | PASS — residence ['DRAM'] |
| 3  remove=[HBM, DRAM] (DROP) | PASS — unit absent from units[] |
| 4  add=[HBM] (load_back) | PASS — graceful decline: `promote_load_back_declined:below_threshold:kv_tokens=4<thr=256` (small tail nodes don't promote — expected) |
| 5  duplicate hash in batch | PASS — 1 applied, 1 `already_acted_this_batch` |
| 6  idempotency (re-add DRAM) | PASS — `add_already_present:DRAM` |
| 7  unknown hash | PASS — `not_in_tree` |
| 8  add=[DISK] | PASS — `disk_tier_not_yet_wired` |
| 9  action_id echo | PASS — 3 distinct action_ids all returned |
| 10 malformed payload | PASS — 3 missing-field variants all 400 |
| 11 `{HBM} → {DRAM}` combined add+remove | PASS — scheduler healthy post-action; residence == [DRAM] (added 2026-05-31 #157 RED-then-GREEN) |

### Implementation notes worth keeping

- **Cascade required for HBM eviction.**  `evict_component(target=DEVICE)`
  frees the buffer but defers `cd.value = None` to `_cascade_evict` (SWA's
  `free_swa` reads Full.value).  Without the cascade, `/aginfer/state` would
  still report HBM in residence after a `remove=[HBM]` action.  The legacy
  `target="DRAM"` path had this latent bug (T2 only verified `applied=1`,
  never re-checked residence); T20 caught it on Stage 2.
- **`load_back` declines below_threshold for short tails.**  sglang's
  load_back checks `kv_tokens >= load_back_threshold` (default 256) so
  4-token leaf nodes never promote.  Stage 4's graceful-decline branch
  is the right contract.  T26's bw_free + T34's joint DP will know to
  rank these tails differently.

* raw run log: `results/20260531_t20_initial_pass.log`

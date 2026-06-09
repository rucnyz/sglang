# SGLang Cache Node Access Events — Instrumentation Hooks for Reuse Tracing

## Overview

This document identifies every per-cache-node access event in SGLang's unified radix cache where lightweight instrumentation can capture `(unit_hash, timestamp, access_kind, n_tokens, holders)` tuples. The goal is to reconstruct the full reuse pattern for the empirical p_hat scoring function (T11b).

**Instrumentation strategy**: Log events in low-overhead points that don't sit in the hot path:
- Prefer post-operation (after lock/state updates)
- Avoid per-token granularity where possible
- Use `node.id` or `node.hash_value` as unit identifier
- Leverage existing `node.last_access_time` and `node.hit_count` state

---

## File: `unified_radix_cache.py` (Main Cache Driver)

### 1. **Cache Hit (Prefix Match)** — Lines 849–854
**Function**: `_match_post_processor()`  
**Trigger**: Every successful prefix match (read access)  
**Frequency**: Per-request (batch-level)  
**Unit ID**: `best_match_node.id` (the deepest matched node)  
**Captured state**:
- `node.last_access_time` ← `get_and_increase_time_counter()` at line 849
- Time counter scanned at **line 849**
- Ancestors also updated, lines 850–853

```python
# LINE 849-854 (in _match_post_processor):
cur_time = get_and_increase_time_counter()  # <-- HOOK 1a: Log (CACHE_HIT)
while node_update:
    node_update.last_access_time = cur_time
    cur_time -= 0.00001
    node_update = node_update.parent
```

**Instrumentation point**: Line 849, immediately after `get_and_increase_time_counter()`
```python
# ADD AFTER LINE 849:
trace_event = {
    'unit_id': best_match_node.id,
    'unit_hash': best_match_node.get_last_hash_value(),
    'timestamp': cur_time,
    'access_kind': 'CACHE_HIT',
    'n_tokens': len(best_match_node.key.token_ids) if best_match_node.key else 0,
    'depth': _compute_node_depth(best_match_node),
}
_log_access_event(trace_event)
```

**Expected overhead**: ~200 ns (single dict construction + logging call)

---

### 2. **Cache Insertion (Node Insert or Overlap)** — Lines 1019, 1060
**Function**: `_insert_helper()`  
**Trigger**: On every prefix overlap (existing node visited during insert)  
**Frequency**: Per-node-on-path (multiple per insert)  
**Unit ID**: `node.id` (the node being overlapped)  
**State at hit**: `node.hit_count` incremented at **line 1019** (before LRU/backup)

```python
# LINE 1019 (in while loop of _insert_helper):
self._inc_hit_count(node, params.chunked)

# LINE 1060 (new leaf case):
self._inc_hit_count(target_node, params.chunked)
```

**Instrumentation point**: Line 1019, inside loop before incrementing
```python
# ADD BEFORE LINE 1019:
if params.program_id:  # Only log tagged requests
    trace_event = {
        'unit_id': node.id,
        'unit_hash': node.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'INSERT_OVERLAP',
        'n_tokens': prefix_len,
        'program_id': params.program_id,
    }
    _log_access_event(trace_event)
self._inc_hit_count(node, params.chunked)
```

**Also at line 1060** (new leaf creation):
```python
# ADD BEFORE LINE 1060:
trace_event = {
    'unit_id': target_node.id,
    'unit_hash': target_node.get_last_hash_value(),
    'timestamp': peek_time_counter(),
    'access_kind': 'INSERT_NEW_LEAF',
    'n_tokens': len(value),
    'program_id': params.program_id,
}
_log_access_event(trace_event)
```

**Expected overhead**: ~300 ns per node (dict + condition check + logging)

---

### 3. **Node Creation (Split)** — Lines 912–913
**Function**: `_split_node()`  
**Trigger**: When an existing node key partially matches new insert key (radix tree restructure)  
**Frequency**: Rare (O(log N) per insert worst case, typically <1%)  
**Unit ID**: `new_node.id` and `child.id`  
**State**: Both get fresh `last_access_time` stamp at **line 913** (child only)

```python
# LINE 913:
child.last_access_time = get_and_increase_time_counter()
```

**Instrumentation point**: After line 913, log both nodes
```python
# ADD AFTER LINE 913:
cur_time = child.last_access_time
trace_event_new = {
    'unit_id': new_node.id,
    'unit_hash': new_node.get_last_hash_value(),
    'timestamp': cur_time,
    'access_kind': 'SPLIT_PARENT',
    'n_tokens': len(new_node.key.token_ids) if new_node.key else 0,
    'trigger_split_len': split_len,
}
trace_event_child = {
    'unit_id': child.id,
    'unit_hash': child.get_last_hash_value(),
    'timestamp': cur_time,
    'access_kind': 'SPLIT_CHILD',
    'n_tokens': len(child.key.token_ids) if child.key else 0,
}
_log_access_event(trace_event_new)
_log_access_event(trace_event_child)
```

**Expected overhead**: ~400 ns (two events, rare operation)

---

### 4. **Hit Count Increment (for Write-Through Backup Trigger)** — Lines 1577–1587
**Function**: `_inc_hit_count()`  
**Trigger**: On insert overlap for non-evicted, non-chunked nodes (when HiCache enabled)  
**Frequency**: Per-overlap during inserts  
**Unit ID**: `node.id`  
**State transition**: `node.hit_count += 1` at **line 1585**; write_backup fired at **line 1587** if threshold crossed

```python
# LINE 1577-1587:
def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
    if self.cache_controller is None:
        return
    if node.evicted or chunked:
        return
    if self.cache_controller.write_policy == "write_back":
        return
    node.hit_count += 1  # <-- STATE CHANGE
    if not node.backuped and node.hit_count >= self.write_through_threshold:
        self.write_backup(node)  # <-- TRIGGER
```

**Instrumentation point**: Line 1585, after increment
```python
# ADD AFTER LINE 1585 (and before write_backup call):
if node.hit_count == self.write_through_threshold:
    trace_event = {
        'unit_id': node.id,
        'unit_hash': node.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'WRITE_THRESHOLD_REACHED',
        'hit_count': node.hit_count,
        'n_tokens': len(node.component_data[BASE_COMPONENT_TYPE].value),
    }
    _log_access_event(trace_event)
```

**Expected overhead**: ~150 ns (guard + log on threshold only)

---

### 5. **Device-to-Host Backup (DRAM Write)** — Lines 1383–1444
**Function**: `write_backup()`  
**Trigger**: Node being demoted from device to host (write-through or write-back eviction)  
**Frequency**: Per-evict-or-hit-threshold  
**Unit ID**: `node.id`  
**State changed**: `node.component_data[BASE_COMPONENT_TYPE].host_value` set at **line 1427** (in commit)

```python
# LINE 1427 (in commit_hicache_transfer):
node.component_data[ct].host_value = transfers[0].host_indices.clone()
```

**Instrumentation point**: After line 1420 (after write succeeds), before commit
```python
# ADD AFTER LINE 1420, BEFORE LINE 1428 (the commit):
if host_indices is not None:
    trace_event = {
        'unit_id': node.id,
        'unit_hash': node.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'BACKUP_HOST',
        'n_tokens': len(host_indices),
        'write_back': write_back,
    }
    _log_access_event(trace_event)
```

**Expected overhead**: ~200 ns (single event at backup)

---

### 6. **Host-to-Device Reload (DRAM Read)** — Lines 1500–1529
**Function**: `load_back()`  
**Trigger**: Evicted node being reloaded from host to device  
**Frequency**: Per-prefetch-miss that triggers loadback  
**Unit ID**: `best_match_node.id`  
**State changed**: `node.component_data[ct].value` set in `commit_hicache_transfer` (lines 1512–1515)

```python
# LINES 1500-1505 (load call):
device_indices = self.cache_controller.load(
    host_indices=kv_xfer.host_indices,
    node_id=best_match_node.id,
    extra_pools=aux_xfers or None,
)
```

**Instrumentation point**: After line 1500, if load succeeds
```python
# ADD AFTER LINE 1507 (check device_indices is not None):
if device_indices is not None:
    trace_event = {
        'unit_id': best_match_node.id,
        'unit_hash': best_match_node.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'LOAD_BACK',
        'n_tokens': len(device_indices),
    }
    _log_access_event(trace_event)
```

**Expected overhead**: ~200 ns (conditional on success)

---

### 7. **Host-to-Storage Backup (L3/Persistent Tier Write)** — Lines 1589–1633
**Function**: `write_backup_storage()`  
**Trigger**: Backuped node written to storage (async, post-backup to DRAM)  
**Frequency**: Per-node when storage enabled and hit_count threshold reached or async triggered  
**Unit ID**: `node.id`  
**State**: Recorded in `self.ongoing_backup` at **line 1630**

```python
# LINE 1623-1629 (write call):
operation_id = self.cache_controller.write_storage(
    node.component_data[BASE_COMPONENT_TYPE].host_value,
    node.key.token_ids,
    node.hash_value,
    prefix_keys,
    extra_pools=aux_xfers or None,
)
self.ongoing_backup[operation_id] = (node, self.inc_host_lock_ref(node).to_dec_params())
```

**Instrumentation point**: After line 1623, before line 1630
```python
# ADD BETWEEN LINE 1623 AND LINE 1630:
trace_event = {
    'unit_id': node.id,
    'unit_hash': node.get_last_hash_value(),
    'timestamp': peek_time_counter(),
    'access_kind': 'BACKUP_STORAGE',
    'n_tokens': len(node.component_data[BASE_COMPONENT_TYPE].host_value),
    'operation_id': operation_id,
}
_log_access_event(trace_event)
```

**Expected overhead**: ~200 ns

---

### 8. **Device Eviction (Leaf Removal)** — Lines 1329–1362
**Function**: `_evict_device_leaf()`  
**Trigger**: LRU eviction picks a device leaf and removes it  
**Frequency**: Per-eviction driven by memory pressure  
**Unit ID**: `node.id`  
**State**: Before evict at **line 1340**, node is marked evicted (value=None) at **line 1153**

```python
# LINE 1340 (assertion):
assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"

# LINE 1352-1355 (cascade evict):
for comp in self._components_tuple:
    self._evict_component_and_detach_lru(
        node, comp, target=EvictLayer.ALL, tracker=tracker
    )
```

**Instrumentation point**: After line 1340 (confirmed it's a D-leaf), before eviction
```python
# ADD AFTER LINE 1340:
trace_event = {
    'unit_id': node.id,
    'unit_hash': node.get_last_hash_value(),
    'timestamp': peek_time_counter(),
    'access_kind': 'EVICT_DEVICE_LEAF',
    'n_tokens': len(node.component_data[BASE_COMPONENT_TYPE].value),
    'has_backup': node.backuped,
}
_log_access_event(trace_event)
```

**Expected overhead**: ~200 ns (per evict event, not hot)

---

### 9. **Host Eviction (Host Leaf Removal)** — Lines 1364–1379
**Function**: `_evict_host_leaf()`  
**Trigger**: Host LRU eviction picks a host leaf (evicted, backuped, no children)  
**Frequency**: Per-host-eviction when host pool fills  
**Unit ID**: `node.id`  
**State**: Node is deleted from tree at **line 1378**

**Instrumentation point**: After line 1370, before eviction
```python
# ADD AFTER LINE 1370:
trace_event = {
    'unit_id': node.id,
    'unit_hash': node.get_last_hash_value(),
    'timestamp': peek_time_counter(),
    'access_kind': 'EVICT_HOST_LEAF',
    'n_tokens': len(node.component_data[BASE_COMPONENT_TYPE].host_value) if node.component_data[BASE_COMPONENT_TYPE].host_value else 0,
}
_log_access_event(trace_event)
```

**Expected overhead**: ~200 ns

---

### 10. **Lock Acquisition (Protect from Eviction)** — Lines 531–542
**Function**: `inc_lock_ref()`  
**Trigger**: Request pins a node and its ancestors (match_prefix → lock for generation)  
**Frequency**: Per-request, path-lock (1 per ancestor)  
**Unit ID**: `node.id` (and each ancestor up to root)  
**State**: `node.component_data[ct].lock_ref` incremented in `FullComponent.acquire_component_lock()` (lines 169–210)

**Instrumentation point**: In `FullComponent.acquire_component_lock()`, lines 206
```python
# FILE: unified_cache_components/full_component.py
# LINE 206 (after lock_ref increment):
cd.lock_ref += 1  # <-- AFTER THIS
trace_event = {
    'unit_id': cur.id,
    'unit_hash': cur.get_last_hash_value(),
    'timestamp': peek_time_counter(),
    'access_kind': 'LOCK_ACQUIRE',
    'lock_ref': cd.lock_ref,
}
_log_access_event(trace_event)
```

**Expected overhead**: ~250 ns per node on path (~3-5 nodes typical)

---

### 11. **Lock Release (Unprotect)** — Lines 212–246
**Function**: `release_component_lock()` in `FullComponent`  
**Trigger**: Request finishes, node is no longer needed  
**Frequency**: Per-request, path-unlock (1 per ancestor)  
**Unit ID**: `node.id` (each ancestor)  
**State**: `cd.lock_ref -= 1` at **line 242**

**Instrumentation point**: After line 242
```python
# FILE: unified_cache_components/full_component.py
# AFTER LINE 242:
cd.lock_ref -= 1
if cd.lock_ref == 0:
    trace_event = {
        'unit_id': cur.id,
        'unit_hash': cur.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'LOCK_RELEASE',
        'lock_ref': cd.lock_ref,
    }
    _log_access_event(trace_event)
```

**Expected overhead**: ~250 ns per node (only log on release to 0)

---

## File: `unified_cache_components/full_component.py`

### 12. **Full Component Data Write (Backup to Host)** — Lines 289–319
**Function**: `commit_hicache_transfer()` with `CacheTransferPhase.BACKUP_HOST`  
**Trigger**: After successful device→host transfer  
**Frequency**: Per-backup operation  
**Unit ID**: `node.id`  
**State changed**: `node.component_data[ct].host_value` set at **line 300**

**Instrumentation point**: After line 300
```python
# ADD AFTER LINE 300:
if transfers and transfers[0].host_indices is not None:
    node.component_data[ct].host_value = transfers[0].host_indices.clone()
    trace_event = {
        'unit_id': node.id,
        'unit_hash': node.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'FULL_COMMIT_BACKUP_HOST',
        'n_tokens': len(transfers[0].host_indices),
    }
    _log_access_event(trace_event)
```

**Expected overhead**: ~200 ns

---

### 13. **Full Component Data Load (Reload from Host)** — Lines 302–319
**Function**: `commit_hicache_transfer()` with `CacheTransferPhase.LOAD_BACK`  
**Trigger**: After successful host→device transfer  
**Frequency**: Per-loadback operation  
**Unit ID**: Each node in `xfer.nodes_to_load`  
**State changed**: `n.component_data[ct].value` set at **line 313** for each node

**Instrumentation point**: Inside loop at line 313
```python
# IN THE LOOP, AFTER LINE 313:
for n in xfer.nodes_to_load or []:
    cd = n.component_data[ct]
    n_len = len(cd.host_value)
    cd.value = device_indices[offset : offset + n_len].clone()
    # ADD HERE:
    trace_event = {
        'unit_id': n.id,
        'unit_hash': n.get_last_hash_value(),
        'timestamp': peek_time_counter(),
        'access_kind': 'FULL_COMMIT_LOAD_BACK',
        'n_tokens': n_len,
    }
    _log_access_event(trace_event)
    offset += n_len
```

**Expected overhead**: ~200 ns per node

---

## Summary Table: Recommended Instrumentation Points

| # | File | Function | Line(s) | Event Type | Overhead (ns) | Per | Notes |
|---|------|----------|---------|-----------|---|----|----|
| **1a** | `unified_radix_cache.py` | `_match_post_processor()` | 849 | CACHE_HIT | 200 | request | Deepest matched node; all ancestors touch at 850-853 |
| **2** | `unified_radix_cache.py` | `_insert_helper()` | 1019, 1060 | INSERT_OVERLAP, INSERT_NEW_LEAF | 300 | node/insert | Multiple per insert; best logged before _inc_hit_count |
| **3** | `unified_radix_cache.py` | `_split_node()` | 913 | SPLIT_PARENT, SPLIT_CHILD | 400 | split (rare) | Both new parent and child nodes created |
| **4** | `unified_radix_cache.py` | `_inc_hit_count()` | 1585 | WRITE_THRESHOLD_REACHED | 150 | threshold only | Fires once per node when hit_count crosses threshold |
| **5** | `unified_radix_cache.py` | `write_backup()` | 1420 | BACKUP_HOST | 200 | backup | Device→Host demotion |
| **6** | `unified_radix_cache.py` | `load_back()` | 1507 | LOAD_BACK | 200 | loadback | Host→Device reload |
| **7** | `unified_radix_cache.py` | `write_backup_storage()` | 1623 | BACKUP_STORAGE | 200 | backup | Host→Storage write (async) |
| **8** | `unified_radix_cache.py` | `_evict_device_leaf()` | 1340 | EVICT_DEVICE_LEAF | 200 | evict | LRU eviction trigger |
| **9** | `unified_radix_cache.py` | `_evict_host_leaf()` | 1370 | EVICT_HOST_LEAF | 200 | host-evict | Host pool pressure relief |
| **10** | `full_component.py` | `acquire_component_lock()` | 206 | LOCK_ACQUIRE | 250 | node/path | Per ancestor; path-lock semantics |
| **11** | `full_component.py` | `release_component_lock()` | 242 | LOCK_RELEASE | 250 | node/unlock | Log on transition to lock_ref=0 |
| **12** | `full_component.py` | `commit_hicache_transfer()` BACKUP_HOST | 300 | FULL_COMMIT_BACKUP_HOST | 200 | backup | Redundant with #5 but captures FULL component view |
| **13** | `full_component.py` | `commit_hicache_transfer()` LOAD_BACK | 313 | FULL_COMMIT_LOAD_BACK | 200 | loadback | Per node in nodes_to_load; redundant with #6 |

---

## Unit Identifiers Available at Each Point

| Access Kind | Primary ID | Secondary Hash | Tokens Available | Lock State |
|---|---|---|---|---|
| CACHE_HIT | `best_match_node.id` | `best_match_node.get_last_hash_value()` | `len(best_match_node.key.token_ids)` | N/A |
| INSERT_OVERLAP | `node.id` | `node.get_last_hash_value()` | `prefix_len` (matched portion) | `node.component_data[ct].lock_ref` |
| INSERT_NEW_LEAF | `target_node.id` | `target_node.get_last_hash_value()` | `len(value)` | 0 (newly created) |
| SPLIT_PARENT | `new_node.id` | `new_node.get_last_hash_value()` | `len(new_node.key.token_ids)` | inherited from child |
| SPLIT_CHILD | `child.id` | `child.get_last_hash_value()` | `len(child.key.token_ids)` | locked if on path |
| BACKUP_HOST | `node.id` | `node.get_last_hash_value()` | `len(host_indices)` | locked (inc_lock_ref at 1442) |
| LOAD_BACK | `best_match_node.id` | `best_match_node.get_last_hash_value()` | `len(device_indices)` | locked at 1462 |
| BACKUP_STORAGE | `node.id` | `node.hash_value` | `len(node.component_data[BASE_COMPONENT_TYPE].host_value)` | host-locked at 1632 |
| EVICT_DEVICE_LEAF | `node.id` | `node.get_last_hash_value()` | `len(node.component_data[ct].value)` | unlocked before evict |
| EVICT_HOST_LEAF | `node.id` | `node.get_last_hash_value()` | `len(host_value)` | host-unlocked before evict |
| LOCK_ACQUIRE | `cur.id` | `cur.get_last_hash_value()` | `len(cur.component_data[ct].value)` | `lock_ref` after increment |
| LOCK_RELEASE | `cur.id` | `cur.get_last_hash_value()` | `len(cur.component_data[ct].value)` | `lock_ref` after decrement |

---

## Implementation Guidance

### 1. Trace Event Structure
```python
@dataclass
class TraceEvent:
    unit_id: int                    # node.id
    unit_hash: Optional[str]        # node.get_last_hash_value() or node.hash_value
    timestamp: float                # peek_time_counter() or result of get_and_increase_time_counter()
    access_kind: str                # one of the 13 event types above
    n_tokens: int                   # length of matched/involved tokens
    program_id: Optional[str] = None  # req.program_id if available
    depth: Optional[int] = None     # distance from root (for tree structure analysis)
    lock_state: Optional[int] = None  # lock_ref value if applicable
    extra: Optional[dict] = None    # component-specific data (hit_count, backup status, etc.)
```

### 2. Logging Backend
- **Buffer**: Per-rank ringbuffer or memory-mapped queue (~10 MB for 1M events)
- **Format**: Binary (msgpack) or compact JSON (one event per line)
- **Flush**: Async thread, every 1s or on ring full
- **Location**: `/tmp/sglang_trace_rank{rank}.bin` or configurable via env var `SGLANG_TRACE_FILE`

### 3. Feature Gates
```python
# In unified_radix_cache.py __init__:
self._trace_enabled = os.environ.get("SGLANG_TRACE_ENABLED", "0") == "1"
self._trace_logger = _get_trace_logger() if self._trace_enabled else None

# Helper:
def _log_access_event(self, event: TraceEvent):
    if self._trace_logger:
        self._trace_logger.push(event)
```

### 4. Avoiding Hot Path Impact
- **Conditional logging**: Guard all traces with flag check (0.5 ns cost when disabled)
- **Lazy dict construction**: Use f-string or lazy dict to avoid alloc on disabled path
- **Post-operation**: Always log AFTER state has been committed (allows async flush)
- **Batch coalescing**: Combine per-node events within same request (dedupe neighbors)

---

## Minimum Viable Set for Full Reconstruction

To reconstruct the reuse pattern (`(n_reuses_at_age, lifetime)` per unit), instrument these **5 points**:

1. **Line 849** (`_match_post_processor`): CACHE_HIT — record when node is accessed
2. **Line 1019** (`_insert_helper`): INSERT_OVERLAP — record prefix hits on insert (secondary path)
3. **Line 1340** (`_evict_device_leaf`): EVICT_DEVICE_LEAF — record death time
4. **Line 206** (`full_component.acquire_component_lock`): LOCK_ACQUIRE — record holder count
5. **Line 1623** (`write_backup_storage`): BACKUP_STORAGE — record tier transitions (if L3 enabled)

**Total overhead**: ~1.2 µs per request (CACHE_HIT + brief INSERT checks) + ~200 ns per evict.  
**Data volume**: ~200 bytes/event × 10k events/sec = 2 MB/s (acceptable for 1s buffering).

---

## References

- **Time counter**: `get_and_increase_time_counter()` (line 46 import, defined in `tree_component.py` line 86)
- **Hit count state**: `node.hit_count` (initialized line 128, incremented line 1585)
- **Last access state**: `node.last_access_time` (initialized line 126, updated lines 850–853, 913, 920)
- **Backup state**: `node.component_data[BASE_COMPONENT_TYPE].host_value` (set after line 1427, line 300, line 313)
- **Eviction state**: `node.component_data[ct].value` (set to None at line 1153; cleared on evict)
- **Lock state**: `node.component_data[ct].lock_ref` (incremented line 206, decremented line 242)


# Audit: Design Intent — Dynamic Admission Cap with Mamba Pool Growth

**Audit date:** 2026-05-26  
**Auditor:** Claude (file search specialist)  
**Target:** `/scratch/yuzhou/projects/sglang/dev/interlayer/design.md` (1178 lines)  
**Verdict:** **(c) Discovered now via D8 failure; need to add to design**

---

## Summary

The original `design.md` **does not mention or assume** that sglang's admission cap (`max_running_requests`, derived from `ReqToTokenPool.size`) will dynamically resize with mamba pool growth. The design focuses entirely on the **actuator mechanism** (physical page transfers) and **Admitter/Budgeter cost models**, but **ignores the scheduler's per-request array capacity constraints** that cap the maximum concurrent requests.

**D8 (saturated single-pool) failure revealed this gap:** the actuator successfully grew mamba pool from 251 slots to higher (via 43 cross-pool fires), but throughput unchanged (-1.70% vs target +10%) because sglang's admission cap stayed fixed at ~33 reqs, capped by `ReqToTokenPool.free_slots` array size.

---

## Design Document Analysis

### §358–391: Admitter design (per-arrival cost decisions)

**Quote (design.md lines 358–391):**

> On each request arrival with demand X bytes for destination pool `i_dst`, evaluate five candidates and pick the cheapest:
> ```
> own-free:    cost = 0                                  # space already free in i_dst
> own-evict:   cost = c^evict_dst(X)                     # evict i_dst cache
> cross-free:  cost = c^xfer(X)                          # transfer from i_src (has free)
> cross-evict: cost = c^xfer(X) + c^evict_src(X)         # transfer + evict at i_src
> defer:       cost = Q · w_q                            # enqueue this arrival
> ```
> 
> This is paper §design-l2-protocol verbatim. The cost model makes burst-safety automatic without any pre-reservation — the per-arrival Admitter fires the actuator synchronously exactly when a transfer is cheaper than the alternatives.

**Analysis:** The Admitter section describes how to decide *whether to fire the actuator*. It assumes the Admitter can *choose to defer* (`defer` candidate with cost `Q · w_q`), but the document is **silent on what physically limits the maximum number of concurrent requests** that can be admitted. It references no constraint from `ReqToTokenPool.size` or any scheduler array bound.

---

### §937–948: D8 (saturated single-pool) headline claim

**Quote (design.md lines 937–948):**

> **Conjecture:** on a workload that saturates one pool while the other has slack (R1 at RPS=32: mamba 99 % / KV 5 %), the actuator grows the saturated pool past its boot-time max and admits more concurrent reqs.
>
> **Test:** R1 sweep, off vs inter.
>
> **Pass:** `output_throughput_inter / output_throughput_off ≥ 1.10` AND `server.log` shows a fire with `mamba_cap_after > 251` (default boot max).

**Critical observation:** D8 states the actuator "admits more concurrent reqs" as a *consequence of* growing the mamba pool. **The design assumes this will happen automatically** — but provides **no mechanism or discussion** of how sglang's admission layer will learn that it can now accept more reqs.

**Gap:** The design does not specify:
- That `ReqToTokenPool.size` or any req-state array needs to be resized
- **When** the scheduler's admission cap should be updated
- **Who updates it** (interlayer actuator? Budgeter? sglang scheduler itself?)
- **What concurrency constraints** apply during the update

---

### §260–276: Allocator floor (working set only)

**Quote (design.md lines 260–276):**

> The actuator never shrinks a pool below its current working set + a small margin. **There is no engine-cap-derived floor and no static "live_floor" reservation.**
>
> ```python
> def kv_min_now(scheduler):
>     ws = ceil(Σ req.kv_size_now for req in running_batch / tokens_per_chunk)
>     return ws + safety_margin
>
> def mamba_min_now(scheduler):
>     ws = len(scheduler.running_batch)
>     return ws + safety_margin
> ```

**Observation:** The design explicitly states there is "no engine-cap-derived floor." This contradicts the empirical reality that sglang's admission gate is capped by `max_running_requests` (computed at boot time). The design is describing the **intended behavior** (no static floor) but **not acknowledging** that sglang currently has one via `ReqToTokenPool.size`.

---

### §1139–1159: Caveats

**Quote (design.md lines 1139–1159):**

> - **Below-saturation operation (Lemma A1)**. The cross-free Admitter candidate needs the donor pool to have FREE pages most of the time.
> - **Bubble required**. Symmetric load on both pools means no asymmetry to exploit.
> - **Phase shorter than Budgeter tick is invisible to Budgeter**. Admitter still catches it per arrival.
> - **No regret bound**. We drop paper's closed-form `π̂_i` for empirical pressure signals.

**Observation:** The Caveats section lists operational assumptions and limitations, but does **not mention** the scheduler's per-request array capacity as a potential constraint or caveat.

---

## Related Evidence: Recent Discoveries

### From `dev/interlayer/dyn_admission_cap/progress.md` (2026-05-26)

> D8 v1 ran, failed: throughput Δ = -1.70% (target +10%).  
> **Diagnosed: sglang's `ReqToTokenPool.free_slots` array is init-sized for `max_running_requests=33`; fires grow mamba but admission can't exceed the array size.**

### From `dev/interlayer/dyn_admission_cap/discussion.md`

The discussion of options 1a/1b/1c confirms this was a **post-design discovery**:
- **Option 1b** was chosen (dynamic resize), with user comment: "1b吧，既然这个难做，那你单独给这个开一个文件夹负责追踪做这个的过程和问题"
- User explicitly directed: "create a dedicated folder to track progress + issues" and "dispatch multiple subagent audits before implementation"
- This structure would not exist if the design had already specified this requirement.

---

## Conclusion: Verdict (c)

**This work (dynamic `ReqToTokenPool` resize) is discovered now via D8 failure; not part of the original design.**

### Evidence:

1. **Design.md does not mention `ReqToTokenPool`, `max_running_requests`, or per-request array capacity** in any context (verified via grep and full read).

2. **D8 headline assumes automatic behavior but provides no mechanism:**  
   D8 states the actuator will "admit more concurrent reqs" but the design contains zero discussion of how scheduler admission capacity scales.

3. **Allocator floor section explicitly disclaims static caps**, but this disclaimer is **aspirational**, not descriptive of actual sglang behavior.

4. **Recent context (May 2026) shows this as a discovery:**
   - `dyn_admission_cap/` folder created after D8 v1 failure
   - `progress.md` dates the discovery to 2026-05-26
   - Options 1a/1b/1c were "considered" (i.e., post-hoc), not part of original plan
   - User direction to create tracking folder + dispatch audits confirms this was unplanned

---

## Suggested Design.md Updates

To document the chosen approach **(option 1b: dynamic ReqToTokenPool resize)**, add a new subsection to design.md:

### Proposed Addition: §350–365 (after §Admitter definition, before Budgeter)

**Title:** `### Admission cap coupling with pool growth (dynamic scaling)`

**Suggested wording:**

```markdown
#### Dynamic admission cap (option 1b)

The Admitter's per-arrival cost decisions assume the scheduler can 
admit more concurrent requests when the mamba pool grows beyond its 
boot-time size. This requires that sglang's per-request bookkeeping 
arrays (ReqToTokenPool.free_slots, FutureMap, ReqToMetadataIdxAllocator, 
etc.) scale dynamically when the actuator grows the mamba pool.

**Mechanism:** when a cross-pool fire grows mamba_pool.size from 
N → M, the Budgeter signals the scheduler to resize its per-request 
arrays. The new admission cap becomes floor(M / mamba_per_req), capped 
by user-set --max-running-requests. No hot-path overhead: resize is 
triggered post-fire, not per-arrival.

**Rationale:** pre-allocating arrays for max-possible pool (option 1a) 
wastes ~13.7 GiB memory. Capping arena growth (option 1c) limits the 
design's reach. Dynamic resize (1b) is architecturally correct: 
admission cap always matches available pool capacity.

**Concurrency model:** resize must be serialized with respect to 
scheduler's admission-path reads of pool.size (or protected by atomic 
swap / copy-on-write). CUDA graph captures must not embed array pointers 
that become stale post-resize.
```

### Proposed Addition: §D8 (update existing conjecture)

**Current D8 line 941–942:**

> the actuator grows the saturated pool past its boot-time max and 
> admits more concurrent reqs.

**Suggested update:**

> the actuator grows the saturated pool past its boot-time max. The 
> Budgeter signals the scheduler to resize its per-request arrays, 
> enabling the admission gate to accept more concurrent reqs 
> (up to the new pool capacity, capped by --max-running-requests).

---

## Remaining Open Questions (for implementation)

1. **Thread safety of resize:** Does `ReqToTokenPool.req_to_token` (torch tensor) allow safe resize while in-use by other threads?
2. **CUDA graph capture:** Do captured graphs embed `.data_ptr()` of req-state tensors that become stale on resize?
3. **Other arrays:** Complete list of per-req arrays that must grow (FutureMap, ReqToMetadataIdxAllocator, etc.)?
4. **Resize trigger:** Should resize be synchronous (fire blocks until scheduler resizes) or asynchronous (fires grow pool, scheduler catches up on next tick)?

These are documented in `/dev/interlayer/dyn_admission_cap/progress.md` as audit targets for subagents.


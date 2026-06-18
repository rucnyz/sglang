# Test Review Checklist (for subagent reviewers)

Use this checklist when reviewing any test file under
`dev/interlayer/0_page_state_machine/`,
`dev/interlayer/1_dyn_admission_cap/`,
`dev/interlayer/2_admitter/`,
`dev/interlayer/3_budgeter/`, or
`dev/interlayer/4_e2e/`.

The goal: catch tests that LOOK rigorous but are actually shallow,
redundant, or tautological — before they ship and give us false
confidence.

Apply **three lenses** in order. A test only earns "ship" status when
it passes all three.

---

## Lens 1 — Comprehensiveness

Does the test cover the failure modes that this layer is supposed to
catch?

**For each public API symbol** in the module under test (read the
source: classes, functions, methods, parameters):

- Is every method exercised at least once?
- Is every parameter that has non-trivial branching tested (e.g.,
  `evict_policy="tail"` vs other values; `external_handle_pool=None`
  vs given)?
- Are validation guards (raises, asserts inside the SUT) triggered at
  least once each?
- Are documented contract clauses (docstring promises) tested? E.g.,
  if the docstring says "returns 0 when n==0", is n=0 tested?

**For each failure mode named in the design** (paper §section, design.md
conjecture):

- Is each failure mode that the design claims is prevented actually
  put under stress, not just the happy path?

**For the production code path specifically**:

- Does the test exercise the path used in production, not just the
  simplified variant? Example: if production uses
  `cross_arena_transfer(arena1, arena2)` but the test only calls
  `arena.transfer_chunks(pool_a, pool_b)`, that's a coverage hole.

**Output**: for each missing coverage, classify as:
- **Must-add to this test**: surfaces in normal downstream tests; cheap to add
- **Belongs in test N**: already covered elsewhere; verify and move on
- **Edge case worth a separate test**: rare but should have its own test

---

## Lens 2 — Redundancy / dead weight

Some tests look diagnostic but are **tautological** (always pass) or
**subsumed** by another test in the same file.

**Tautological patterns to flag**:

- Comparing two things that are the same object via different paths
  (e.g., `arena._free_handles is pool.free` → comparing
  `len(arena._free_handles) == len(pool.free)` always passes)
- Testing that a function returns what we just stored in it
  (round-tripping through the same dict, no transformation in between)
- Testing API existence (`assert hasattr(obj, 'method')`) instead of
  behavior
- Testing that the code "doesn't crash" without asserting anything
  about the result
- "Verifying" that an attribute equals itself after some operation
  that doesn't touch that attribute

**Subsumed patterns to flag**:

- Test X's assertion is a strict subset of test Y's assertion (Y
  already proves X as a byproduct)
- Two tests have the same setup and assertion shape but differ only in
  trivially-similar parameters (e.g., direction A→B vs B→A when the
  code routes by string name with no positional dependence)
- Test exists only to "match the title" in the test list but doesn't
  exercise anything load-bearing

**Output**: for each redundant test, recommend:
- **DELETE** — pure noise / tautological
- **MERGE into test N** — fold the unique assertion into an existing test

---

## Lens 3 — Depth (is it really testing the logic, or just a mock?)

For each test, answer this question:

> "What's the WORST plausible bug in the production code that this
> test would catch? Could an implementation that has that bug still
> pass this test?"

**Red flags that indicate shallow / mock-like testing**:

- Test only verifies **counters** when it could verify **identity**.
  Example: "Pool A lost 2 chunks, Pool B gained 2 chunks" passes for
  an implementation that releases handles and `cuMemCreate`s new ones
  on every transfer (real bug; would leak HBM). **Should also verify
  the same handle indices moved.**

- Test verifies one offset / one element when it could verify all
  bytes. Example: reading `tensor[0]` to confirm a write succeeded
  passes for an implementation that maps a 4 KiB page when 2 MiB was
  requested. **Should verify all bytes, or at least first/middle/last.**

- Test's setup determines the answer. Example: "Verify pool A keeps
  data" when only A has data, B never gets data — the test would pass
  even if A and B's reads were swapped. **Use distinct, asymmetric
  data on each pool.**

- Test asserts derived quantities that follow from the production code
  being correct. Example: `assert mapped_count + free_count == total`
  is always true by construction; it only catches the kind of bug
  that would also produce other obvious failures.

- Test relies on hidden invariants that aren't asserted. Example: A
  ping-pong test that assumes the same handles cycle through, without
  verifying that they do. Catches "no crash"; doesn't catch "leak".

**Reference question for every test**: "If the implementation were
replaced by an obvious-but-wrong stub (e.g., a function that just
returns the expected counter delta without doing anything), would my
test catch it?" If not, the test is a mock.

**Production-like timing/concurrency test**: when the SUT has a
synchronization semantic (e.g., "must wait for pending GPU work
before X"), the test MUST reproduce the production scenario
where that semantic matters — not just call the SUT in isolation
after `torch.cuda.synchronize()`. A test that always pre-syncs
will pass any implementation whether or not it has the required
sync internally. Specifically:
  - Don't pre-sync before calling the SUT in a sync-semantic test
  - Verify pending work exists at SUT entry (e.g., `cudaEvent.query()`
    returns False) so the test's premise is real
  - Make pending work HEAVY enough to outlast Python dispatch
    overhead (matmul of 8192² × 50 on H200 ≈ 1 s; lighter loads
    drain during the for-loop and the race window vanishes)
  - Beware of PyTorch ops on `from_blob` / non-tracked tensors —
    they often inject implicit syncs that drain the queue, making
    the test inconclusive

**Output**: for each test, verdict:
- **KEEP** — substantive, irreplaceable
- **TIGHTEN** — assertion is weak; specify exactly what to add
- **DELETE** — test catches nothing real

---

## Format for review output

For each test in the file:

```
test_N <name>:
  Lens 1 (coverage)    : <verdict>
  Lens 2 (redundancy)  : <verdict>
  Lens 3 (depth)       : <verdict>
  Worst bug it catches : <one sentence>
  Recommendation       : KEEP | TIGHTEN | MERGE | DELETE
  If TIGHTEN/MERGE     : <specific change>
```

End with:

```
What's still missing (after the recommendations above are applied):
  1. <missing case + which test should own it + why>
  2. ...
```

---

## Worked examples (from vmm_boot_smoke review history)

### Example 1 — Lens 2 catches tautological assertion

**Original test_10 (free-count parity)**:
```python
assert arena.free_handle_count() == pool.free_count()
```

**Why it's tautological**: `arena._free_handles is external_handle_pool.free`
(literal line in chunk_arena.py:299 — same Python list object).
Comparing `len(x) == len(x)` always passes.

**Verdict**: DELETE.

### Example 2 — Lens 3 catches shallow assertion

**Original test_2 (cross_arena_transfer)**:
```python
arena_KV.grow("kv", INIT); arena_M.grow("mamba", INIT)
cross_arena_transfer(arena_KV, "kv", arena_M, "mamba", MOVE)
assert arena_KV.pool_mapped_chunks("kv") == INIT - MOVE
assert arena_M.pool_mapped_chunks("mamba") == INIT + MOVE
```

**Why it's shallow**: counter-only. An implementation that does
`cuMemRelease(unmapped_handles)` and `cuMemCreate(new_handles)` for
the dst arena would pass this — and leak HBM under load.

**Verdict**: TIGHTEN. Add: snapshot `_handles_mapped_in(arena_KV, "kv")`
before; after transfer, assert those handle indices appear in
`_handles_mapped_in(arena_M, "mamba")`.

### Example 3 — Lens 1 catches API coverage gap

**Original vmm_boot_smoke v1**: only tested `transfer_chunks` (intra-arena);
production path is `cross_arena_transfer` (inter-arena).

**Verdict**: Must-add to vmm_boot_smoke — production headline path totally
untested.

---

## Anti-patterns to flag in commit reviews

If you see any of the following in a test diff, request rework:

- `assert hasattr(...)` without behavior check
- `assert isinstance(...)` as the only assertion
- `try: foo(); except: pass; assert True` ("it didn't throw")
- Reading the same value twice and asserting equality across the two
  reads
- A test that only asserts a return value the test itself just passed
  as an argument
- "Verify cleanup works" with comment "nothing useful to assert"
- A sync-semantic test that calls `torch.cuda.synchronize()` or
  similar BEFORE invoking the SUT (the pre-sync drains the queue,
  so any implementation passes whether or not it syncs internally)
- An async-workload test without verifying the premise via
  `cudaEvent.query()` (otherwise the workload might silently complete
  during Python dispatch and the test passes any impl)
- **"Adjusting test parameters to make a failing test pass" without
  understanding the root cause**: when a test fails, the response
  must be "debug WHY it fails" before any change to the test. If the
  fix is "reduce N", "cap the size", "increase tolerance", or any
  other parameter loosening, the test is almost certainly hiding a
  real bug or measurement issue. Example smell: va_reservation_hbm test reserving
  74 GiB failed with `CUDA_ERROR_INVALID_VALUE`; the wrong fix was
  "cap to 32 GiB", the right fix was "align size to granularity
  (74 GiB wasn't a multiple of 2 MiB)". The cap would have silently
  masked a real production limit.

---

## When to invoke this checklist

- Every new test file under a phase folder (`0_page_state_machine/`,
  `1_dyn_admission_cap/`, `2_admitter/`, `3_budgeter/`, `4_e2e/`)
- Every diff that touches an existing test (especially "add a new
  sub-test" diffs — easy to add weak sub-tests)
- Before promoting a test to ship-gating status (the conjecture's
  Implementation line in [`design.md`](design.md) §"Validation
  conjectures" flips from *pending* to a folder reference)

For subagent reviewers: read this file first; apply all three lenses;
report in the format above.

# Engineering Principles

Working agreements for the code, tests, and reviews we produce. The
goal is to keep the codebase readable, debuggable, and worth trusting
six months from now when nobody remembers the session that produced it.
Every rule below has hurt at least once when ignored.

Read this alongside whatever design and planning documents the project
maintains — those describe *what* the system does and *what's next*;
this document is the *how*.

---

## Code style — fight silent bugs

### 1. Fail-fast over defensive

**Rule**: When a precondition is violated, crash loudly at the
mutation site. Do not paper over the violation with a "safe" default,
a try/except that swallows, or a synchronization primitive that hides
the misuse.

**Why**: Defensive code converts immediate, observable failures into
delayed, distant ones. The original site that broke the invariant
runs to completion; minutes or hours later something downstream
allocates against a corrupted state, and the trace points at a victim
that has no idea where the corruption started. By the time someone
debugs it, the call stack is wrong, the logs are wrong, and the only
useful clue (which call broke the invariant) has been erased.

A loud crash at the violation site is not noise — it is the actual
bug surfacing where it can be acted on. The cost of "production now
crashes when X happens" is bounded; the cost of "production silently
misbehaves when X happens, surfacing as Y three hours later" is not.

**How to apply**:
- Prefer `assert invariant, "explain what was violated and likely cause"`
  over silent fall-through. The assertion message is documentation
  about what the code expects.
- If an invariant is genuinely platform-uncertain (CUDA driver quirk,
  network race), the design must call it out explicitly — not bury it
  behind a defensive guard in production code.
- Resist the urge to wrap a known-flaky call in retry/sync logic.
  Either the underlying contract is wrong (fix it) or the call site
  is misusing it (fix that).

### 2. No defensive fallbacks in production

**Rule**: Don't add `try/except` that swallows, `or default` for
required arguments, `if x is None: return ...` for arguments that
must not be `None`, or any other code path whose only role is to
keep the program running when an upstream contract is broken.

**Why**: Defensive fallbacks have two failure modes and they are both
bad. (a) The upstream contract is in fact never broken, and the
fallback is dead code that future readers must reason about. (b) The
upstream contract really does get broken sometimes, and the fallback
hides the breakage instead of fixing it — sometimes for a long time,
because nothing observable signals the problem.

The cleanest version of a program is one where each function's
preconditions are stated, the callers honor them, and the function
trusts them. Defensive code is a workaround for not enforcing
preconditions at boundaries — fix the boundary, not the inside.

**How to apply**:
- For test stubs vs production: if production reads `obj.foo`, the
  stub provides `foo`. Don't add `getattr(obj, "foo", default)` to
  production so the stub doesn't have to.
- For optional parameters with sensible defaults: parameter defaults
  are fine. Run-time fallbacks for missing object state are not.
- If you find yourself writing "in case it's not there," ask whether
  it can actually not be there, in production, today. If not, delete
  the guard. If yes, the missing case is a bug somewhere else —
  surface it.

**Counter-examples**:
- Wrapping a call to a collaborator in `try/except Exception: pass`
  "just to be safe" turns every bug in the collaborator into a
  silent no-op. The trace you wanted goes into the void.
- Writing `getattr(obj, "_attr", default)` in production code
  because a test stub doesn't set `_attr`. The fix is to set
  `_attr` in the stub — not to teach production code to tolerate
  a malformed object. Defensive code that exists only to
  accommodate stubs is the most common form of this rule's
  violation in practice.

**A related but distinct anti-pattern: virtual edge cases.**
Sometimes the temptation is not "in case the collaborator is
broken" but "in case some hypothetical pre-version-N build, or
some imagined future subclass, doesn't conform." Treat virtual
edge cases the same way: don't add the guard. Before adding code
to handle a scenario, verify the scenario can actually occur in
production *today*. Grep the call sites; check `__init__` of the
classes that exist; trace the actual path. If the edge case is
real, fix the contract at the boundary so it can't happen. If it
isn't, no code change is needed.

### 3. State must be initialized unconditionally; no defensive `getattr` lookups

**Rule**: Any attribute representing object state — anything an
instance owns and mutates — must be set unconditionally in `__init__`.
Readers access it as `self._x` directly. Never write
`getattr(self, "_x", None)` or `getattr(self, "_x", <default>)` for
self-state.

**Why**: A defensive `getattr` lets "the attribute was never set"
silently masquerade as a valid state. Bugs where `__init__` forgot a
field surface hundreds of lines downstream instead of at the missing
assignment. "Attribute missing" and "attribute present with empty
value" become indistinguishable to readers, and reasoning about state
requires inspecting every call site.

Direct access fails loudly with `AttributeError` at the line that
demanded the missing field — the actual bug surfacing where you can
act on it.

**How to apply**:
- For collection state: initialize as the empty form
  (`torch.empty(0, ...)`, `set()`, `[]`) in `__init__`. Same shape
  and dtype as the populated form.
- For scalar state: pick the value that satisfies the post-init
  invariant (often `0`, the initial capacity, or whatever the spec
  says "no mutation yet" should look like).
- Collapse `if x is None or x.numel() == 0:` to `if x.numel() == 0:`.
- Test stubs that stand in for production objects must mirror the
  same initializations. Don't reach for `hasattr`/`getattr` to paper
  over stub gaps — that's defensive code in production for the
  benefit of a stub.

**Exception**: `getattr` is legitimate for **cross-object lookups**
where the referenced collaborator may genuinely not exist yet — e.g.,
during boot when a downstream subsystem hasn't been constructed. The
rule forbids defensive lookups on **self-state**, not on optional
references between objects.

---

## Testing discipline

### 4. Write the failing test before the fix; the discipline holds for review-caught bugs too

**Rule**: When a bug is identified — by symptom, crash, log, code
review, or automated analysis — write a narrow test that reproduces
it *before* changing the production code. The test must fail in the
current code, and fail for the right reason.

The full workflow:
1. Observe symptom (test failure, crash, wrong output, audit finding).
2. Read the logs / traceback / report.
3. Form a hypothesis; verify by reading the relevant source.
4. **Write the reproducing test. Run it. It must FAIL, for the
   reason in your hypothesis.**
5. Apply the minimal fix. Test goes green.
6. Run the surrounding regression suite. Document the finding and
   fix in the appropriate place.

**Why**: A failing test is the only artifact that *proves* the bug
existed and that the fix actually addresses it. Without it:
- You may be fixing a different bug than the one you think.
- The bug can return six months later and nothing will warn you.
- Future readers cannot tell whether the fix is load-bearing or
  superstition.

The step that's tempting to skip is step 4, especially when an
external review or automated audit "confidently" reports a bug. The
confidence is sometimes misplaced; reviewers and tools hand-wave
plausibly-sounding edge cases that don't actually exist in
production. Without a failing test you cannot tell apart "real bug
real fix" from "imaginary bug, harmless no-op change" from "imaginary
bug, fix that breaks the real-but-different invariant."

**How to apply**:
- For correctness bugs: write a unit test that pins the precise
  invariant. Don't write a "general scenario" test that happens to
  fail for the same reason — too easy to write a passing version
  that doesn't actually exercise the bug.
- For async / concurrency bugs: write the narrowest synchronous
  reproduction that demonstrates the invariant violation, even if
  the production manifestation is via a race.
- For workload-level bugs where unit reproduction is impractical:
  capture the reproducing workload + run command and check those in
  as the regression artifact. Do not skip evidence entirely.
- Never iterate "try a fix → run live → still broken → try another
  fix" without the test step. That wastes time and obscures the real
  cause.

### 5. Performance tests pin design targets, not the absence of regression

**Rule**: Every performance-sensitive code path has a target number
in the design (a latency, a throughput, a memory bound). Performance
tests must assert that target — not merely "no slower than last
month."

**Why**: A "no regression" test ratchets the bar down over time. The
team accepts each small slowdown because each one is "within noise,"
and within a year the code is twice as slow as the design intended
and nobody notices because every test passed. A "must meet design
target" test catches the cumulative drift the first time a new commit
falls below the bar.

The implicit assumption behind "no regression" testing — that the
current behavior is correct and any change is suspicious — gets it
backwards. The design says what the system *should* do; current
behavior is just the most recent attempt at it.

**How to apply**:
- If the design specifies a numeric budget (e.g., a latency wall in
  the µs/ms range, a per-call cost ceiling), the test asserts the
  budget directly: `actual <= design_budget`, not
  `actual <= last_run * 1.1`.
- When the test fails because the design target is wrong (workload
  changed, hardware changed), update the design first, then the
  test. Don't update the test in isolation.
- Workload-dependent targets need explicit workload fixtures
  documented alongside the target — otherwise the target is
  unfalsifiable.

---

## Architecture decisions

### 6. Pick the architecturally cleanest option; cost and difficulty are not the deciding factors

**Rule**: For production code, when several solutions exist, pick the
one that produces the cleanest long-term shape — the one that makes
the next ten changes easier. If the cleaner option requires touching
adjacent code, touch it. If the cleaner option is harder to implement
now, implement it now.

**Why**: Expedient fixes accumulate. Each one is "just this once,"
each one adds a special case, and within a phase or two the code is
a thicket of one-off branches that nobody dares to refactor because
each branch was a real fix for a real bug. The cost of the cleaner
solution is paid once; the cost of the expedient one is paid every
time someone reads, extends, or debugs the resulting tangle.

This rule applies to production code on paths you intend to keep.
Test-side workarounds for missing infrastructure are tolerable; the
production code itself is held to the ideal.

**How to apply**:
- When a fix requires "just adding a special case," step back: is
  the special case telling you the underlying model is wrong? Often
  yes. Fix the model.
- When the cleaner option requires updating callers, that is the
  signal that the API was leaking implementation detail. Update the
  API and the callers.
- "We can refactor later" is almost always wrong. Either there is
  time to do it right now, or there will be even less time after
  five more features are built on top of the expedient version.

### 7. Naming should align with the design document

**Rule**: Names you use in code — folder names, file names, class
and function identifiers, comments — should match the names the
design document uses for the same concept. If the design calls
something `Cap-barrier`, the folder, the function, and the comment
referring to it all say `Cap-barrier`, not "preparation step."

**Why**: Code review and design review become the same activity when
the names match. A reader can grep for a concept and find every
mention of it across both spec and implementation. When the names
diverge, every reviewer must mentally translate, every onboarding
doc must explain two vocabularies, and errors creep in.

This includes refactoring discipline: if the design's vocabulary
evolves, the code's names follow. If a folder once corresponded to
a concept the design has since split, the rename is part of the
design change.

**How to apply**:
- New code: read the relevant design section first; use its terms.
- Reviewing code: if names diverge from design, raise it. Don't
  accept "we'll rename later."
- Comments referencing design sections should use section names,
  not line numbers (line numbers rot the next time someone edits
  the spec).

---

## Documentation and review

### 8. Comments describe current behavior, not the session that produced it

**Rule**: Comments in production code answer "what does this code do
and why does it do it this way." They do not narrate the history of
how the code came to look this way. No "audit fix," no "post-review
correction," no "added after refactor #N," no "this used to be X
until we found Y."

**Why**: Two audiences read code comments. Audience A is the future
reader who needs to understand the code. For them, session history
is noise that obscures the actual semantics. Audience B is the
historian who needs to understand how the code evolved. For them,
the right artifact is the commit message, the pull request
description, or the task tracker — not the code comment.

Session-narrative comments rot the fastest of all comments because
the chain of edits keeps drifting. "Nth audit fix" makes no sense
once two more audits land. "Post-#205 cleanup" is meaningless once
#205 is no longer the most recent context. The comments accumulate
faster than they get pruned, until reading them is reading a layered
diff with no diff tool.

**How to apply**:
- When writing a comment, ask: would this make sense to someone who
  has never seen this PR or this conversation? If no, the comment
  belongs in the PR description, not the code.
- Comments may reference durable cross-references — a task number,
  a stable design section, a published paper — provided the
  reference is genuinely load-bearing for understanding the code.
  Drop references when the *why* becomes obvious from the code or
  when the referenced artifact is no longer durable.
- A common anti-pattern: "We tried X, it didn't work, so we do Y."
  The comment should describe Y and why Y is correct. The dead-end
  X belongs in the PR description or a dev-doc entry.

### 9. Findings go into the relevant folder's docs; reviews are advice, not authority

**Rule**: Two parts.

(a) Every non-trivial finding, fix, or dead-end belongs in
written documentation under the relevant folder, not just in a
commit message. The folder README, a status document, or a
findings log — pick the artifact, but make sure the next person
who walks into that folder can read the current state without
needing to spelunk git history.

(b) Code reviews, audits, and automated analyses produce
*recommendations*. The recommending party — human reviewer, junior
contributor, automated tool — may be wrong. Before acting on a
review finding that changes production code, verify the finding
yourself against the actual code and the actual production
behavior.

**Why (a)**: Commit messages and PR descriptions are append-only
history. They answer "what changed" well, "why" sometimes, but
"what is the current state of this folder" never. Folder-local
documentation is the queryable surface — it gets updated as
findings land, dead ends are recorded next to the live code, and
the next person can read the current understanding without
reconstructing it from a year of git log.

**Why (b)**: Reviewers and tools confidently report patterns that
*look* like bugs but turn out, on close inspection, not to exist
in the production codebase. The patterns are real in general; the
specific instance is not. Examples include:
- "This attribute may not exist" — but in practice the `__init__`
  always sets it.
- "This collaborator may not be constructed yet" — but the call
  site only fires after construction.
- "This concurrent access may race" — but the surrounding code
  already holds the relevant lock.

Acting on these findings without verification produces "fixes" for
nonexistent bugs, which (i) clutter the code with defensive
guards (see §2), (ii) sometimes break the real, more subtle
invariant the original code was preserving, and (iii) train
reviewers that confidence equals authority.

**The verification loop (operational form)**:

When a review produces N findings (whether one human comment or
twenty agent-generated items), don't act on them straight through.
Walk each one:
1. **Is the finding describing a real path in production?** Read
   the call sites; trace the actual flow. If the pattern the
   review names doesn't actually occur, the finding is virtual —
   close it as accepted-with-no-action.
2. **If real, is the proposed fix correct?** A finding can be a
   genuine bug but the suggested fix can make things worse (e.g.,
   trade a silent corruption for a defensive guard). Apply §1–§4
   to the proposed fix; don't just take it.
3. **Apply TDD (§4)**: write the failing test that demonstrates
   the bug *before* the fix. This forces "the bug is real and the
   fix addresses it" to be a verified claim, not a trust statement.
4. **Re-review after the fix.** Another pass — by you, by another
   reviewer, by the same tool — often surfaces follow-on issues
   the first pass missed. Treat audit cycles as expected, not as
   admission of failure.

The cost of running a review through this loop is several minutes
per finding. The cost of skipping it is the occasional landed fix
that breaks production for an invented reason.

**How to apply**:
- After every meaningful task: write the findings, decisions, and
  what was tried into the relevant folder's README (or a sibling
  doc). Don't rely on commits.
- When a review finds a "bug," ask: did the reviewer trace the
  actual production path, or pattern-match from a general
  principle? If the latter, do the trace before changing code.
- Reviews that confirm "I checked and this is fine" are as
  valuable as reviews that find issues. Record them too.

---

## Summary

These nine rules collapse to one underlying value: *make problems
visible at the source, not at the symptom*. Fail-fast (§1) and no
fallbacks (§2, §3) keep production failures honest. TDD (§4) and
perf-targets (§5) keep tests honest. Ideal architecture (§6) and
naming alignment (§7) keep code honest. Current-behavior
comments (§8) and verified reviews (§9) keep documentation
honest.

When in doubt, pick the option that produces a louder, earlier,
more localized failure. Silent success is the second-worst
outcome; only silent failure is worse.

---

## Experiment standard — every case, every workload

1. **No-regression**: sys ≥ base on every metric.
   sys < base × 0.98 on ANY metric = our bug, must fix before proceeding.
   A regression is never "the workload's fault."
2. **Large win**: find a workload shape (from the canonical corpus) where
   sys wins significantly. A flat result means the workload doesn't
   exercise the mechanism, not that the mechanism is broken — keep
   searching.
3. **Record**: the winning workload shape + N=3 A/B become that case's
   official result.

During the search, any regression found on ANY workload is a bug, not
noise. Fix it, then continue. Never silently add workarounds or env-var
overrides to make a failing experiment pass — flag it and decide together.

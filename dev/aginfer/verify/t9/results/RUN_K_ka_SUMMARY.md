# Run K-a (admission OFF) — 2026-05-26 16:38-17:34

> ⚠️ **SUPERSEDED — N=1 single-shot, claims invalidated.**
>
> See `N3_matrix_SUMMARY.md` for the authoritative N=3 result and
> `N3_ROOT_CAUSE.md` for why the "~1.76× slower than Run H'"
> framing was wrong.  TL;DR: H' 885 s was a different setting
> (no `temperature=0.0`); under current settings even H'_now (no
> daemon, ours_greedy inline) is 1392.8 ± 53.6 s — i.e. the
> regression doesn't exist when you re-measure the baseline.
>
> The "Design-level concern" section below — that paper §7 1-step
> greedy V_u is wrong because of multi-turn horizon blindness —
> remains a valid *theoretical* worry (see `todo-empirical-phat`
> memory) but its empirical justification here (the 1.76× number)
> is dead.  T11a daemon-side was implemented and N=3-tested;
> result was Δ=−45 s (not significant).

**Variant:** kv_scheduler=enabled + admission_controller=**disabled** + HiCache ON

| metric | K full | K-a | delta |
|---|---|---|---|
| successful | 30/32 | 30/32 | 0 |
| **mean per-trial** | **1559 s** | **1549 s** | **-10 s (~0)** |
| std | 678 | 731 | +53 |
| p50 | 1500 | 1514 | +14 |
| p90 | 2511 | 2716 | +205 |
| p99 | 3120 | 3380 | +260 |
| sum | 13.86 h | 13.77 h | -0.09 h |

**Verdict:** admission_controller is NOT the source of the overhead.
Disabling it produced essentially identical per-trial times.

**Attribution narrowing — remaining suspects:**
* **kv_scheduler** (still ON; ~11k state-fetches per K-full run; per-event V_u migrate decisions on every paper §4 event)
* **HiCache + Mooncake L3** (still ON; ours_greedy_score might fight Mooncake's prefetch + backup_thread)
* **inline scorer ours_greedy** (still loaded — sglang's drive_eviction uses paper §7 V_u as the heap key)
* **daemon proxy +0.5-2 ms per chat completion** (T4 measured; 5177 × 1.5 ms ≈ 8 s aggregated; not material)

**Next discriminators:**
* **Run J** (HiCache OFF + 3 layers) — if J.mean ≈ Run H' 885 s, HiCache+Mooncake is the slowdown source.
* **Run K-b' (synthetic):** kv_scheduler OFF too, daemon just proxies — same as Run H' but with the proxy hop.  Confirms whether kv_scheduler is at fault.

**Design-level concern (per user, 2026-05-26):**
Both K and K-a being ~1.76× slower than Run H' is empirical evidence
that **paper §7 1-step greedy V_u may not be the right framework**
for swebenchpro's multi-turn long-context agent workload.  V_u
estimates `p_hat` from `hits/age` — does NOT see the multi-turn
reuse horizon that swebenchpro/terminus-2's 200-turn rollouts
exhibit.  ThunderAgent's BFD doesn't predict future reuse at all,
which may be why it doesn't fall into this trap.

Design implication: paper §7's 1-step greedy is the **first thing
to question** if we want a competitive design.  MDP solver (MPC/
RL) over real traces could see the 200-turn horizon and react
accordingly.

# Reproduce — paper results

## Purpose

This directory is the **open-source reproduction package** for the paper
(`hybrid-inference`). Its job is to make every claimed number regenerable: for
each paper Table/Figure it records the **exact command**, the **harvested result
JSONs** the paper cells are read from, and a short **analysis** tying the numbers
back to the claim. Anyone (reviewer, open-source user, or future us) can `cd`
into the matching `RQ<n>/table<k>/` folder, run the one command in its
`README.md`, and reproduce that artifact end-to-end.

It is the single pointer of record from paper → code: when a paper number is
updated (e.g. a tuned config beats the current one), update both the paper cell
and the corresponding `RQ<n>/table<k>/` here so they never drift. Results that
are not yet runnable end-to-end are marked *pending* rather than hard-coded.

## Layout

Each folder reproduces one numbered artifact in the paper, organized by research
question:

```
reproduce/
  RQ<n>/                # research question (RQ1 = end-to-end, RQ2 = ablation, RQ3 = design assumptions)
    table<k>/           # one paper Table/Figure: scripts pointer, exact command, result JSONs, analysis
      README.md
      results/*.json
```

The folder name `table<k>` matches the paper's table/figure number (e.g.
`RQ1/table1` ↔ `\label{tab:main-cross-model}`, the RQ1 main result).

## Conventions

- **GPU**: single H200-class GPU; experiments here were run on GPU 7
  (`CUDA_VISIBLE_DEVICES=7`).
- **Models**: results so far are on **Qwen3.5-9B**. Larger models
  (Qwen3.5-35B-A3B, Kimi-Linear-48B-A3B) are marked *pending* in the paper and
  here — same commands, different `--model-path`.
- **Reproducibility metric discipline**: every A/B is request-bounded (both arms
  process the identical session set, `parity == 1.0`) and run `N=3`; we report
  mean±std or the median per-run delta. Cache-hit is `Σcached / Σprompt` over the
  server's exported per-request metrics JSONL (never log-scraped).
- The harness scripts live in-tree under `dev/interlayer/4_e2e/`; each
  `table<k>/README.md` gives the exact command and the expected numbers, and
  `results/` holds the harvested JSONs the paper cells are read from.

## Status

| RQ | folder | paper artifact | status |
|---|---|---|---|
| RQ1 | `RQ1/` (3 cc workloads — see `RQ1/README.md`) | `tab:main-cross-model` (end-to-end) | workload 1 (`table1`, recurrent-bound) done on Qwen3.5-9B; workload 2 (high-concurrency/short) and workload 3 (dynamic) are planned RQ1 content, **TODO**; larger models + static-best/vLLM baselines pending |
| RQ2 | — | `tab:rq2-headline` (ablation) | pending |
| RQ3 | — | `tab:cost-curves-fit`, LPB-vs-LRU | pending |

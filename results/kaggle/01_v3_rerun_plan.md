# Method 1 version 3 incremental rerun plan

## Cancellation diagnosis

Version 2 was manually stopped by the IHES monitoring task to yield Kaggle capacity to the higher-priority bidirectional notebook. Kaggle reports `CANCEL_ACKNOWLEDGED`. This was not a Kaggle timeout, quota termination, or code failure: the downloaded log has no exception after startup, the T4 completed the smoke cell, and the full search checkpointed three successful rows before the stop request.

## Preserved evidence

- Repository commit: `69885b213433538356b37d6ad27e2040769b4da6`.
- Machine: Kaggle GPU T4 x2; the solver used `Tesla T4` through CUDA.
- Full-search protocol: beam width 365000, maximum depth 30, two symmetry frames, direct and reverse search.
- Completed rows: IDs 100–102, all replay-valid and progressive.
- The 1,003-row `submission.partial.csv` independently passes exact replay validation.
- IDs 103–120, the final summary, and final `submission.csv` are absent.

## Cheaper exact-protocol continuation

1. Do not launch while the higher-priority bidirectional notebook is active. Confirm both an available Kaggle session and sufficient daily GPU quota first.
2. Pin Method 1 version 2 as a `kernel_sources` input. Never copy its competition files or model weights into GitHub.
3. Before any GPU search, import `benchmark_rows.json` and `submission.partial.csv`. Reject the resume input unless method, beam width, model, checkpoint, reference identity, solution lengths, row uniqueness, per-row replay, and all 1,003 submission replays match.
4. Resume the exact four-search portfolio without rerunning completed IDs. Version 3 processes at most the next three missing IDs (103–105) at beam width 365000 and writes a replay-validated 1,003-row `submission.csv` plus combined checkpoint rows.
5. Pin each successful version as the next version's resume source and continue in three-ID shards. Do not count an incremental shard as the final method result. The final verdict is published only when IDs 100–120 are all present and replay-valid in one combined summary.

The three observed full rows took 7,796.49 s, 7,313.75 s, and 7,909.25 s. A three-ID shard therefore budgets approximately 6.6 T4 session-hours at the observed upper rate, stays below the interrupted nine-hour run, and preserves the requested algorithm instead of reducing the beam or transform portfolio. Reusing IDs 100–102 avoids approximately 6.4 session-hours of duplicate work.

This plan is prepared locally and in GitHub only. Publishing or launching version 3 waits for the priority notebook and capacity check.

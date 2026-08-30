# Results

No complete 21-puzzle beam-width-365000 Kaggle T4 run has been ingested yet. Three completed rows were recovered from interrupted Method 1 version 2 and independently replay-validated; the table distinguishes that partial evidence from a final method verdict. Generated notebooks and local smoke tests do not count as full benchmark results.

| # | Method | Completed puzzles | Replay-valid | Total selected length | Runtime | Model/checkpoint | Verdict |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | Parameterized transforms | 3/21 | 3/3 | 71 (completed rows) | 23,019.50 s | `1778521793` / `b19eb25b...` | partial progressive; final pending |
| 2 | Cumulative scoring | 0/21 | pending | pending | pending | pending | pending |
| 3 | Two-step ~300-head model | 0/21 | pending | pending | pending | pending | pending |
| 4 | Two-move retraining | 0/21 | pending | pending | pending | pending | pending |
| 5 | 18-neighbor student | 0/21 | pending | pending | pending | pending | pending |
| 6 | Bidirectional wide join | 0/21 | pending | pending | pending | pending | pending |
| 7 | Richer randomized input | 0/21 | pending | pending | pending | pending | pending |
| 8 | Radius-8 intersection | 0/21 | pending | pending | pending | pending | pending |
| 9 | Radius-7 intersection | 0/21 | pending | pending | pending | pending | pending |
| 10 | Immediate-inverse pruning | 0/21 | pending | pending | pending | pending | pending |
| 11 | Solved + scrambled centers | 0/21 | pending | pending | pending | pending | pending |
| 12 | Ordinary 3×3 projection | 0/21 | pending | pending | pending | pending | pending |
| 13 | GFlowNet / Flow | 0/21 | pending | pending | pending | pending | pending |

Machine-readable rows live in `results/benchmark.csv`; the header is committed so completed Kaggle artifacts can be ingested without changing the schema.

## Failed runs retained for audit

- Method 1, Kaggle version 1: public and output-producing, but not benchmark evidence. Kaggle assigned a Tesla P100 while its PyTorch 2.10.0+cu128 image omitted `sm_60`; all 21 searches recorded `AcceleratorError`. The valid 1,003-row fallback submission does not make those searches complete. See `results/kaggle/01_v1_failure.json`. Subsequent metadata pins `NvidiaTeslaT4`, and summaries now mark any error/invalid/truncated row as `failed`.
- Method 1, Kaggle version 2: ran on GPU T4 x2 for approximately eight hours, then was manually stopped by this monitoring task when the higher-priority IHES bidirectional notebook launched. Kaggle acknowledged the cancellation. The downloaded log contains no code exception after startup, so this was not a timeout, quota termination, or code failure. Recovered artifacts verify repository commit `69885b2`, a completed beam-64 smoke attempt, and three completed beam-365000 rows: puzzle 100 (24 moves, 7,796.49 s), 101 (23 moves, 7,313.75 s), and 102 (24 moves, 7,909.25 s). All three method paths replay successfully and strictly improve the official sample reference. The recovered 1,003-row partial submission also passes independent replay validation, but IDs 103–120, the final summary, and final `submission.csv` are absent; therefore the overall method verdict remains pending. See `results/kaggle/01_v2_interrupted.json` and the exact-protocol [version 3 rerun plan](results/kaggle/01_v3_rerun_plan.md).
- Method 1, Kaggle version 3: completed on GPU T4 x2 in 5 h 39 min from repository commit `ab482a4` and produced a replay-valid 1,003-row submission. It is not new aggregate evidence: the cancelled Version 2 source presented zero accepted completed rows, `resumed_puzzle_ids` was empty, and Version 3 duplicated IDs 100–102 instead of running 103–105. The defect was the combination of an unreliable cancelled-version source and a loader that silently accepted zero completed rows. Resume is now fail-closed and requires the exact expected completed-ID set before search; Version 4 pins successful Version 3. See `results/kaggle/01_v3_nonadvancing.json` and the [Version 4 plan](results/kaggle/01_v4_rerun_plan.md).

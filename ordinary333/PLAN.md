# Ordinary 333: shorter replay-verified solutions

Noon14September:13h subword run ended normally, final21552/1003valid. Compact
direct-state pilot completed:1/6 solved(ID9,length7),5timeouts(30s),0improvements.
Frame-DP resweep21552 also0improvements(54.7s). Compact adapter1960checks passed;
published portable code in commit4297596. Implement stronger exact cornerPDB
plus center lower bound next, not repeated weak-hash deep pilots. No deep optimal
score from that new implementation yet. Current Molab status in MOLAB_RUNS.md.

Latest override14September06:55MSK: local full score **21552**, all1003 replay-valid,
318 saved from user21870; twsearch subword stage alone saved150 from21702.
CPU worker23580/supervisor12016 alive, deadline11:39:47MSK. Six neural experiments
have been started by the user; exact links/status evidence in [MOLAB_RUNS.md](MOLAB_RUNS.md).
The first Chandru page currently returns an internal Molab loading error, so no
fresh GPU-step evidence. Source review found CUDA RNG tensors loaded to GPU during
resume; fixed local bundles cast them back to CPU. Request checkpoint/report,
preserve progress, do not restart from scratch. GPU verification of fix pending.
Autocontinuations saved for14September12:00 and17:30MSK (two activations).

Latest update13September22:45MSK: official Rokicki twsearch intermediate local sum
**21652**, all1003replay-valid,50moves saved versus21702. A separate13-hour CPU
queue with recovery supervisor is running; global optimality is not claimed.
Daily methods log: https://www.kaggle.com/competitions/3-3-3-twist-lab-face-and-middle-layer-turns/discussion/741191
Molab direct marimo-pair connection has succeeded. Six manual-upload standalone
notebooks pass strict structural checks; their GPU training results remain pending.

Target: decrease the sum of legal quarter-layer moves over all 1003 public states.
This project is the **ordinary 54-label cube**, not the 72-label IHES supercube.
Half turns cost two; all middle layers are legal; final labeled centers must match.

## Verified numerical ledger — 13 September 2026

| Experiment | Full sum | Replay-valid states | Interpretation |
|---|---:|---:|---|
| Initial saved two-phase solver | 30100 | 1003 | historical, superseded by supplied routes |
| Radius-5/6 exact subword rewriting | 30002 | 1003 | 98 moves saved from that baseline |
| 24-frame, charged-middle-slice rewriting | 29578 | 1003 | 424 additional moves saved |
| User-provided stronger submission | 21870 | 1003 | new reference, not our solver result |
| Identity paths empty, plus known p10 route | 21830 | 1003 | 38 + 2 moves saved |
| Exact radius-6 subword rewriting | 21708 | 1003 | 53 more cubes improved, 122 moves saved |
| Frame-aware rewriting of that file | **21702** | **1003** | three further cubes, six moves saved |

No global optimum is claimed. The aggregate comparison mixes constructive methods
by a replay-valid per-puzzle minimum. The supplied baseline is clearly attributed.

## Classical solver lanes

1. H48 HTM solving followed by X2→X.X expansion, charged center normalization,
   and 54-label replay. Enumerate several shortest/near-shortest HTM routes and
   rerank by actual competition cost. Build and coordinate bridge are completed;
   pruning-table generation is not yet completed, so no H48 score is claimed.
2. Rokicki twsearch on the exact 18-quarter-layer definition. Shallow smoke cases
   pass; use bounded deeper segment search where expected gain justifies cost.
3. nxopt direct QTM candidate; not yet built or measured.
4. Equal-length segment diversification followed by exact shortening and frame
   lifting. This explores plateaus instead of requiring immediate greedy progress.

## Neural lanes and six requested GPU slots

| Profile | Slot | Run | Controlled difference | Status |
|---|---:|---|---|---|
| Liuda | 1 | TF_QV_K24_S0 | sparse Q + V, K24 | user-reported started; step pending |
| Liuda | 2 | TF_QV_K24_S1 | independent seed | user-reported started; step pending |
| Chandru | 1 | TF_QV_K32_S0 | intermediate walk depth | user-reported started; page loading error |
| Chandru | 2 | TF_QV_K40_S0 | longer walk depth | user-reported started; step pending |
| Renuka | 1 | TF_Q_K24_S0 | remove auxiliary V loss | user-reported started; step pending |
| Renuka | 2 | MLP_QV_K24_S0 | residual MLP at same objective | user-reported started; step pending |

Train initial bounded pilots, then continue promising checkpoints and alternate
training with inference. Do not burn six GPU sessions on identical configurations.
Profiles are changed by the user, never by automatic sign-out/sign-in.

## Recipe port and deliberately disclosed deviations

`TRANSFORMER_AZ_RECIPE.md` is a tetraminx experiment record, despite being supplied
in the earlier 555 discussion. It is a useful candidate, not evidence of ordinary
333 quality. Its core is continuous sparse-Q training plus an auxiliary scalar V
head, exact BFS anchors and conjugation augmentation; no Bellman warm-start.

This initial port uses 26 verified physical pieces, Q18+V, 4-layer width256/8-head
SDPA transformer, folded slot-wise one-hot GEMMs, AdamW 3e-4/3e-3, bf16 on CUDA.
Walk targets p−1/p+1 and p are upper-bound surrogates. Test cubes never enter training.
Differences from the supplied recipe: exact anchors through d4 (all children exact
through d5), four sampled symmetry rows instead of full greedy coverage, batch128,
no compilation during calibration. These are explicit pilot settings, not a claim
of reproducing the 80-hour tetraminx model. Extend d5 anchors/full coverage after
measuring GPU throughput and the fixed benchmark.

## Comparison protocol

- Fixed 50 cubes: IDs 950–999. Maintain identical IDs, beam, wall-time and frames.
- Pilot beam2^6/2^10, then comparative beam2^14, finalists2^16–2^18.
- Forward and inverse searches, Q/V consistency λ=0 versus0.3; MLP/transformer
  blends only after each component has measured quality.
- Report solve coverage, exact validity, paired path differences, baseline-fallback
  aggregate, elapsed GPU time, paired t-test and confidence intervals. Never compare
  only successfully solved subsets. Inspect multiple-testing/checkpoint-selection bias.
- Checkpoint every two minutes. Sampled exact-anchor MAE and actual beam search matter
  more than declining training loss. No hard-coded success canary stops unrelated runs.

## Operational gates / blockers

Earlier direct creation returned403. User reports successful creation via
Fork/Duplicate of the template recorded in MOLAB_RUNS.md; use that method for
future sessions. Distinguish user-reported launches from actual training-step
and checkpoint evidence. Corrected checkpoint resume requires live verification.

New Molab notebooks do not persist ordinary runtime-created files automatically.
Before long runs, verify external checkpoint upload (private token via Secrets),
or explicitly use persistent-cache chunk results and download them. Cache retention
is not an unlimited storage guarantee. Never put account passwords in notebooks,
logs or this public repository. A GPU selection in the header is not proof of a
live training process. Record run URL, profile alias, slot, actual start and heartbeat.

Automation `333-gpu`: check every two hours and update one daily English Kaggle
Discussion report. Record its URL and date before future posts to avoid duplicates.

## Sources

- H48: https://github.com/sebastianotronto/nissy-core
- Exact generic search: https://github.com/rokicki/twsearch
- Sparse-Q upstream: https://github.com/AnanasClassic/cayleypy-training-core
- Molab persistence: https://marimo.io/blog/seamless-storage-in-molab

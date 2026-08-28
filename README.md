# IHES Cube: 13 replay-verified research methods

This public research repository turns the thirteen numbered ideas in the source brief into thirteen isolated, reproducible experiments for the [CayleyPy IHES Cube competition](https://www.kaggle.com/competitions/cayleypy-ihes-cube). Every experiment has its own directory and English Kaggle notebook, while sharing one audited implementation of the 72-point permutation puzzle, model loading, beam search, replay validation, benchmark logging, and `submission.csv` generation.

The unnumbered source note “make a PDB for IHES” is recorded as future context in [METHOD.md](METHOD.md); it is not silently promoted to a fourteenth experiment.

## Fixed evaluation protocol

- Puzzle IDs: `100..120` inclusive.
- Requested beam width: `365000`.
- Required outputs: per-puzzle solution length, exact replay validity, runtime, model/checkpoint identity, and a method verdict.
- Notebook artifact: `/kaggle/working/submission.csv`, validated by replay for all 1,003 rows.
- Verdict rule: **progressive** only when the method produces at least one replay-valid strict reduction against the declared reference and the selected total for IDs 100–120 is lower; otherwise **not-progressive**. The reference identity is stored in every run. An optional attached control result can replace the official sample submission as the reference.
- Evidence rule: an unexecuted notebook is never presented as a benchmark result. `pending` means no completed run has been ingested.

## Experiments

| # | Method | Core intervention | Kaggle | Beam-365000 result | Verdict |
|---:|---|---|---|---|---|
| 1 | [Parameterized transforms](methods/01_parameterized_transforms/README.md) | validated symmetries, path reversal, `MirrorState`, `ComplementState`, and move substitutions | [public notebook](https://www.kaggle.com/code/arabidopsisthalian/ihes-method-01-parameterized-transforms) | v2: 3/21 replay-valid, then interrupted | partial progressive; final pending |
| 2 | [Cumulative scoring](methods/02_cumulative_scoring/README.md) | accumulate scores along each beam path | pending | pending | pending |
| 3 | [Two-step ~300-head model](methods/03_two_step_300_head/README.md) | 306 non-backtracking two-move outputs | pending | pending | pending |
| 4 | [Two-move retraining](methods/04_two_move_retraining/README.md) | train and infer with two-move generators | pending | pending | pending |
| 5 | [18-neighbor student](methods/05_neighbor_score_student/README.md) | second model distills 18 neighbor scores from the first | pending | pending | pending |
| 6 | [Bidirectional wide join](methods/06_bidirectional_wide_join/README.md) | direct/reverse beams and exact meet-in-the-middle join | pending | pending | pending |
| 7 | [Richer randomized input](methods/07_richer_randomized_input/README.md) | current state, neighbors, and seeded randomized encodings | pending | pending | pending |
| 8 | [Radius-8 wait/intersect](methods/08_radius8_intersection/README.md) | exact backward-ball lookup at radius 8 | pending | pending | pending |
| 9 | [Radius-7 wait/intersect](methods/09_radius7_intersection/README.md) | exact backward-ball lookup at radius 7 | pending | pending | pending |
| 10 | [Immediate-inverse pruning](methods/10_no_immediate_inverse/README.md) | remove only the previous move’s exact inverse | pending | pending | pending |
| 11 | [Solved + scrambled centers](methods/11_dual_center_trajectories/README.md) | train on both ends of known replay-valid trajectories | pending | pending | pending |
| 12 | [Ordinary 3×3 projection](methods/12_ordinary_333_projection/README.md) | project the same states to a standard 3×3 cube and use the ordinary solution as an IHES prefix | pending | pending | pending |
| 13 | [GFlowNet / “Floy-Flow”](methods/13_gflownet_flow/README.md) | reproduce the MIT-licensed flow-regularized shortest-path approach and compare it | pending | pending | pending |

## Repository layout

```text
ihes_dual/     audited scalar-model beam/search core reused under MIT
ihes13/        method registry, transforms, model heads, trainers, runners, and benchmarking
methods/       thirteen self-contained experiment directories
notebooks/     thirteen generated English Kaggle notebooks
results/       schema-valid summaries; raw run data are not committed
tests/         algebra, replay, notebook, registry, and security checks
tools/         notebook generation, result ingestion, and repository validation
```

## Local verification

```bash
python -m pip install -e .
python -m pytest
python tools/validate_repository.py
```

The test suite uses a small synthetic permutation puzzle and does not require competition data or weights. Real smoke runs accept explicit asset paths and never copy those assets into the repository.

## Kaggle execution

Attach the competition data, `cube_symmetries.npy` where needed, and one registered IHES model asset. Each Kaggle metadata file pins `NvidiaTeslaT4`. Every notebook defaults to IDs 100–120 and beam width 365000, checkpoints its per-puzzle JSON/CSV records, validates every selected path by exact replay, and finally validates all 1,003 rows of `submission.csv`.

See [METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before interpreting results.

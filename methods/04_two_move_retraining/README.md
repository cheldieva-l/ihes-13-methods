# 04 — Two-move generators and retraining

## Hypothesis

A scalar model trained on one-step random walks may be miscalibrated when inference jumps by two moves. Retraining on the same composite generator distribution should reduce that mismatch.

## Intervention

Generate non-backtracking two-move random walks, distill/retrain a scalar student on their teacher scores and walk depths, then search using 306 composite permutations. Odd-length completions are checked with a final one-move expansion.

Every returned composite is expanded and replayed with official generators.


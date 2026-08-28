# 02 — Cumulative scoring

## Hypothesis

Ranking only by the current heuristic can favor a locally attractive child whose full path has been consistently poor. A weighted cumulative score may preserve more stable trajectories.

## Intervention

At each layer, rank by `teacher(child) + alpha * cumulative(parent)`. The configured `alpha`, normalization rule, and resulting score range are written to the run log.

## Comparison

All generators, checkpoint, beam width, depth limit, and replay rules match the common control; only path-score accumulation changes.


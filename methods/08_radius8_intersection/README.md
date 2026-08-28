# 08 — Wait/intersect at radius 8

## Hypothesis

Stopping the learned forward search at an exact state within the solved radius-8 ball can avoid the least reliable final heuristic layers.

## Intervention

Construct a deduplicated backward BFS ball through radius 8 with exact parent/move records, partition its hash index, and test every retained forward state for an exact intersection. Reconstruct the backward suffix and replay the joined path.

If a state cap is set or storage is exhausted, the run is marked `truncated`; it is not accepted as radius-8 benchmark evidence.


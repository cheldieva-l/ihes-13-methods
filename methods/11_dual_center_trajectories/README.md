# 11 — Train on solved and scrambled centers using known trajectories

## Hypothesis

Training only around the solved state underspecifies the landscape near a hard scrambled target. Replay-valid paths provide labels from both ends.

## Intervention

For each training puzzle, replay the declared reference trajectory. Prefix states receive distance-to-solved labels; inverted suffix states receive distance-to-scrambled-center labels. A held-out puzzle split is fixed before training. The log records which reference identity supplied the trajectories.

Participant submissions are never committed; they may be attached privately at runtime as an explicit control.


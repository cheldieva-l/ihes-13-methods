# 10 — Prevent the immediate inverse move

## Hypothesis

Generating the exact inverse of the previous move wastes roughly one eighteenth of expansion and creates a state already present two layers earlier.

## Intervention

At depth greater than one, exclude only `inverse(last_move)`. Signed moves are handled through the official inverse table. No face, axis, layer, or double turn is globally banned.

Adjacent inverse pairs cannot belong to a shortest path, so the pruning preserves shortest-path completeness.


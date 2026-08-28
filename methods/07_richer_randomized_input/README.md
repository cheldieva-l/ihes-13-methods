# 07 — Richer input with neighbors and randomized encodings

## Hypothesis

The single-state one-hot input hides local landscape shape. Explicit neighbor context and a fixed random code can expose local minima, curvature, and symmetry-breaking information.

## Intervention

For each state, encode the current permutation plus pooled features from all 18 neighbors. A seeded random codebook embeds sticker identities; its seed and SHA-256 fingerprint are logged. The student is distilled from the scalar teacher on non-backtracking walks.

Random does not mean irreproducible: the same seed must recreate the exact codebook.


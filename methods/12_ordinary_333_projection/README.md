# 12 — Solve the same projected states as an ordinary 3×3 cube

## Hypothesis

Ignoring center-picture orientation yields an ordinary 3×3 projection whose solution can cheaply remove edge/corner disorder before IHES search finishes the centers.

## Intervention

Recover a move-equivariant IHES-to-facelet mapping from the official generators, normalize whole-cube orientation, solve the projected facelets with an ordinary 3×3 solver, translate face turns to IHES outer-layer moves, replay that prefix, then run the registered IHES model on the residual state.

The ordinary solution alone is never submitted; only the complete replay-valid IHES path is accepted.

The optional runtime solver is `muodov/kociemba` at audited commit `e2690493b43921732960cd5eeee2b1ee91922a7b`, GPL-2.0. It is installed separately on Kaggle and no solver source or binary is copied into this repository.

# Method and benchmark contract

## Puzzle convention

An IHES state and each of the 18 named moves are length-72 permutations. Applying move `g` to state `s` is `s[g]`. A claimed solution is accepted only if replaying every official move from the original untransformed start produces the official central state exactly.

## Common controls

All methods use the same puzzle IDs, beam width, depth limit, model identity log, deterministic seeds, and submission validator. Method-specific changes are declared in `methods/*/config.json`. The shared runner writes one row per puzzle with:

```text
method_id, puzzle_id, beam_width, solution, solution_length,
replay_valid, runtime_seconds, model_id, checkpoint_sha256,
reference_length, delta_vs_reference, verdict, run_status
```

Failures remain explicit; the valid reference row is retained in `submission.csv`, so a failed search can never corrupt the competition file.

## Thirteen interventions

1. **Parameterized transforms.** Validate every relabelling by conjugating all 18 generators. `MirrorState` and `ComplementState` are separate parameterized transforms rather than hidden aliases. Path reversal uses reversed order plus named inverse moves. Axis substitutions (`frd`, `rdf`, `dft`, and all validated permutations), signs, and doubled quarter turns are accepted only when conjugation yields exact official generators. Every candidate is converted back and replayed.
2. **Cumulative scoring.** Rank a child by `h(child) + alpha * cumulative(parent)` rather than only `h(child)`. `alpha` is logged.
3. **Two-step head.** Enumerate the 306 two-move sequences that exclude an immediate inverse. A student head returns one score for each sequence from one parent evaluation and is distilled from the scalar teacher.
4. **Two-move retraining.** Build two-move composite permutations, generate two-step non-backtracking training walks, retrain a scalar student, and search in composite layers. Reconstructed composites are expanded to official moves before replay.
5. **18-neighbor student.** Distill the scalar teacher’s score for all 18 children into an 18-output model. Beam expansion consumes these parent-level action scores directly.
6. **Bidirectional wide join.** Retain declared forward and reverse depths, map the complete reverse frontier into direct coordinates, hash-index it, verify meetings by exact 72-entry equality, and reconstruct the joined path without a supplied midpoint.
7. **Richer randomized input.** Encode the current state together with all 18 neighbors. Seeded random codebooks are fixed per run and fingerprinted so “random encoding” is reproducible.
8. **Radius 8.** Build an exact deduplicated backward ball through radius 8, optionally in disk partitions, then stop the forward beam at the first exact intersection. Any configured cap marks the run `truncated` and cannot be reported as an exact-radius result.
9. **Radius 7.** The same exact contract with radius 7.
10. **No immediate inverse.** At depth `d>1`, exclude only `inverse(last_move)`. All other 17 moves remain available. This preserves completeness for shortest paths because an adjacent inverse pair can always be deleted.
11. **Two centers and known trajectories.** Replay-valid reference paths supply supervised prefixes from the solved end and suffixes from the scrambled end. Training labels and center identity are stored; test paths never enter another puzzle’s training split unless explicitly configured.
12. **Ordinary 3×3 projection.** Remove the 24 center-orientation stickers, recover a move-equivariant isomorphism to 48 ordinary non-center facelets, normalize whole-cube orientation, solve the projected state, map the ordinary face turns to IHES outer-layer moves, replay the prefix, then finish the remaining center-orientation state with IHES search. The ordinary path is a projection aid, not by itself an IHES solution.
13. **GFlowNet.** Adapt the official MIT-licensed implementation accompanying Morozov, Maksimov, Tiapkin, and Samsonov (2026): shared residual backbone, forward/backward policy heads, trajectory balance with flow regularization, and cumulative log-policy beam scoring. The IHES adaptation and any deviations are documented; third-party weights are not redistributed.

## Contextual PDB idea

The source image contains an unnumbered line, “make a PDB for IHES.” It is retained here as a future experiment idea. No PDB result is implied by the thirteen numbered experiments.

## Verdicts

- `progressive`: replay-valid selected total for IDs 100–120 is strictly below the declared reference total and at least one puzzle was strictly improved.
- `not-progressive`: a completed run does not satisfy the rule above.
- `pending`: no completed full run has been ingested.
- `failed`: all requested rows may have been recorded, but at least one search ended in `error`, `invalid`, or `truncated`; it is not benchmark evidence.
- `invalid` or `truncated`: the run violated replay/exactness requirements and is not benchmark evidence.

Runtime and length comparisons are meaningful only between rows with the same puzzle set, beam width, device class, depth limit, and declared reference.

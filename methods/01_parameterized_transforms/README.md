# 01 — Parameterized transforms

## Hypothesis

The scalar model is not perfectly invariant to valid IHES relabellings or path direction. Searching multiple exact frames can retain useful paths that the identity-frame beam drops.

## Intervention

Enumerate and validate official symmetry conjugations, direct/reverse paths, `MirrorState`, `ComplementState`, axis orders such as `frd`, `rdf`, and `dft`, signed moves, and doubled quarter turns. Mirror and complement are separate parameterized functions. A transform is usable only when every conjugated generator maps exactly to an official move or a declared two-move sequence.

## Correctness boundary

Every frame candidate is converted back to official move names and replayed from the original state. Invalid conjugations and double-move substitutions are rejected rather than guessed.

Run `python methods/01_parameterized_transforms/run.py --help` or execute the matching Kaggle notebook.


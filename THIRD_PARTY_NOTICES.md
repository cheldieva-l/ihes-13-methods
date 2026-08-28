# Third-party notices and provenance

## Reused IHES search core

Source: <https://github.com/cheldieva-l/ihes-dual-model-beam-search>

Audited source commit: `887ea66` (`887ea66 Disambiguate competition assets with symmetry inputs`). Files under `ihes_dual/` were copied from that repository and remain under its MIT license, reproduced at `licenses/IHES-DUAL-MODEL-MIT.txt`. The present repository adds new method code and tests without modifying the source checkout used by the higher-priority project.

## CayleyPy Cube

Source: <https://github.com/khoruzhii/cayleypy-cube>

Audited revision: `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`.

License: MIT. The earlier core and this repository use the published permutation convention, residual-MLP checkpoint layout, random-walk training idea, and GPU beam-search architecture. Competition data and model weights are loaded at runtime and are not redistributed.

## GFlowNet shortest-path method

Paper: Nikita Morozov, Ian Maksimov, Daniil Tiapkin, and Sergey Samsonov, “Learning Shortest Paths with Generative Flow Networks,” arXiv:2603.01786 (2026), <https://arxiv.org/abs/2603.01786>.

Official code: <https://github.com/GreatDrake/gfn-pathfinding>. Audited revision: `4c2dc375f894b1259164a17ae56fec85524d2377`. License: MIT, copyright 2026 Nikita Morozov.

Method 13 reimplements the prefix-wise trajectory-balance loss, stop-probability flow, flow regularizer, six LayerNorm residual stages, separate forward/backward policy logits, and cumulative backward-log-policy beam ranking. The IHES adaptation is PyTorch rather than JAX/Equinox, embeds 72 unique sticker labels rather than one-hot encoding six colors, learns `log_z` because the exact IHES state-space cardinality is not supplied, and excludes only an immediate inverse during sampling/search. These deviations are explicit comparison variables. No official source file or trained weight is redistributed. “Floy” in the source note is treated as a likely misspelling of “Flow”; the repository uses the paper’s actual GFlowNet terminology.

## Ordinary 3×3 projection

Method 12 rederives a move-equivariant mapping from the official IHES generators and a standard geometric 3×3 facelet model. Its optional runtime solver is [muodov/kociemba](https://github.com/muodov/kociemba), audited at `e2690493b43921732960cd5eeee2b1ee91922a7b`, licensed GPL-2.0. No Kociemba source or binary is included in this MIT repository; the notebook installs it separately at runtime. Exact Janus/Enārēs tablebases are not included.

## Kaggle assets

The CayleyPy IHES Cube competition data, model assets, symmetry arrays, user submissions, and generated outputs remain under their respective Kaggle terms. They are never committed. Model IDs and SHA-256 fingerprints are metadata used to make experiments auditable, not a redistribution of weights.

## Source brief

The thirteen experiment descriptions were translated from a user-provided Russian image/text brief. The transcription is requirements provenance, not third-party code.

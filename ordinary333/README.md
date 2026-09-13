# Ordinary 333: sparse-Q experiments and verified route shortening

[Plan and numerical ledger](PLAN.md). This directory is separate from the IHES
experiments in the repository root. It uses 54 distinct labels and all18 signed
quarter-layer moves. It is NOT HTM, and does NOT track oriented IHES center pictures.

## Start a GPU training pilot

```bash
python -m ordinary333.training --config ordinary333/configs/TF_QV_K24_S0.json --output /tmp/TF_QV_K24_S0 --seconds 1800
```

The output contains a resumable `latest.pt` and a small `status.json`.
Re-running with the same config/output resumes automatically. Download or remotely
publish the checkpoint before Molab shuts down: runtime files are not automatically
persistent. Never embed a GitHub token or an account password in source code.

## Compare the checkpoint

```bash
python -m ordinary333.search --checkpoint /tmp/TF_QV_K24_S0/latest.pt --output /tmp/eval --beam 16384 --seconds 120
```

The fixed benchmark has 50 IDs950–999, direct+inverse search, exact d4 endgame
verification and per-puzzle baseline fallback. ML training does not read those
states. The beam implementation is a correctness-first pilot, not a claim of
production multi-million-node throughput.

## Port status

The sparse-Q/V objective is adapted from the user-supplied PieceTransformer+AZ
tetraminx recipe and the public [training-core](https://github.com/AnanasClassic/cayleypy-training-core)
method. Recipe experiment results are not reproduced by this new port yet.
Exact geometry has been derived and checked on native54 permutations. CPU/GPU
training smoke status is recorded in the plan rather than presumed from syntax.

Dependencies: Python3.11+, numpy, PyTorch2.5+. GPU bf16 training preferred.

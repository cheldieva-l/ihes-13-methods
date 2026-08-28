# 13 — Study, reproduce, and compare the GFlowNet “Flow” approach

## Origin

This method targets “Learning Shortest Paths with Generative Flow Networks” by Morozov, Maksimov, Tiapkin, and Samsonov (2026), not an unidentified “Floy” codebase. The paper and official code are linked in `THIRD_PARTY_NOTICES.md`; both the official repository and this adaptation use MIT-licensed source boundaries.

## Intervention

Adapt the official shared residual backbone with separate forward/backward policy heads. Train with trajectory balance plus flow regularization on IHES random walks, then rank beam paths by cumulative negative log backward-policy probability. The run logs the regularization coefficient, optimizer settings, model fingerprint, and teacher-independent solve result.

The audited official revision is `4c2dc375f894b1259164a17ae56fec85524d2377` under MIT. This implementation keeps the prefix-wise loss and flow regularizer, but uses PyTorch, 72-label embeddings, learned `log_z`, and exact immediate-inverse pruning. No third-party trained weights are redistributed.

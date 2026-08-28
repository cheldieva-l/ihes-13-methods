# 03 — Two-step model with roughly 300 outputs

## Hypothesis

One model evaluation can rank an entire two-move neighborhood, reducing repeated teacher evaluations and looking past one misleading local step.

## Intervention

The action set contains exactly 306 ordered move pairs (`18 × 17`) after excluding an immediate inverse. A student head is distilled from the registered scalar model’s scores on the 306 resulting states. Composite paths are expanded to official moves before replay.

The run log fingerprints the trained student checkpoint; the checkpoint itself remains a Kaggle output and is not committed.


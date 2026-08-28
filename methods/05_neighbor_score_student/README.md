# 05 — Second model with 18 neighbor-score outputs

## Hypothesis

The first scalar model requires 18 child evaluations per parent. A second model trained on the first model’s complete neighbor score vector can rank all moves with one parent evaluation.

## Intervention

Distill the scalar teacher into an 18-output student. Search consumes the student’s parent-level action scores directly and records teacher/student validation error before benchmarking.

The method is not credited as progressive unless its own replay-valid paths improve the declared reference.


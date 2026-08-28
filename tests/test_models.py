from __future__ import annotations

import torch

from ihes13.scorers import GFlowNetIHES, MultiActionStudent, seeded_codebook


def test_student_output_shapes() -> None:
    states = torch.arange(72).repeat(3, 1)
    neighbor = MultiActionStudent(18, hidden_size=32, residual_blocks=1)
    two_step = MultiActionStudent(306, hidden_size=32, residual_blocks=1)
    assert neighbor(states).shape == (3, 18)
    assert two_step(states).shape == (3, 306)


def test_random_codebook_is_reproducible() -> None:
    first, first_hash = seeded_codebook(72, 8, 7)
    second, second_hash = seeded_codebook(72, 8, 7)
    assert torch.equal(first, second)
    assert first_hash == second_hash


def test_gflownet_policy_shapes() -> None:
    states = torch.arange(72).repeat(2, 1)
    model = GFlowNetIHES(action_count=18, hidden_size=32, residual_blocks=1)
    forward, backward = model(states)
    assert forward.shape == (2, 19)
    assert backward.shape == (2, 18)
    assert model.score_actions(states).shape == (2, 18)


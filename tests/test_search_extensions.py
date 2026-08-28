from __future__ import annotations

import numpy as np
import torch

from ihes_dual.beam import BeamConfig, beam_search

from ihes13.intersection import build_backward_ball, intersect_trace


class FixedActionScorer:
    def score_actions(self, states: torch.Tensor) -> torch.Tensor:
        # Prefer the inverse action for every parent.
        result = torch.zeros((len(states), 2), device=states.device)
        result[:, 0] = 1.0
        result[:, 1] = 0.0
        return result

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(states), device=states.device)


def test_parent_level_action_scorer_finds_solution(cycle_puzzle) -> None:
    start = cycle_puzzle.apply(cycle_puzzle.solved, 0)
    trace = beam_search(
        cycle_puzzle,
        start,
        FixedActionScorer(),
        BeamConfig(beam_width=2, max_depth=2, parent_chunk=2, inference_batch=2, device="cpu"),
    )
    assert trace.solution == [1]
    assert cycle_puzzle.verify_solution(start, trace.solution)


def test_cumulative_scoring_and_inverse_pruning_execute(cycle_puzzle) -> None:
    start = cycle_puzzle.replay(cycle_puzzle.solved, [0, 0])
    trace = beam_search(
        cycle_puzzle,
        start,
        FixedActionScorer(),
        BeamConfig(
            beam_width=2,
            max_depth=3,
            parent_chunk=2,
            inference_batch=2,
            device="cpu",
            cumulative_alpha=0.5,
            prune_immediate_inverse=True,
        ),
    )
    assert trace.solution is not None
    assert cycle_puzzle.verify_solution(start, trace.solution)


def test_exact_backward_ball_intersection(cycle_puzzle) -> None:
    ball = build_backward_ball(cycle_puzzle, radius=1)
    assert ball.exact_radius_covered
    start = cycle_puzzle.replay(cycle_puzzle.solved, [0, 0])
    trace = beam_search(
        cycle_puzzle,
        start,
        FixedActionScorer(),
        BeamConfig(beam_width=2, max_depth=2, parent_chunk=2, inference_batch=2, device="cpu"),
        retain_depths=(1, 2),
    )
    joined = intersect_trace(cycle_puzzle, start, trace, ball, (1, 2))
    assert joined is not None
    assert cycle_puzzle.verify_solution(start, joined.path)


def test_backward_ball_cap_marks_truncation(cycle_puzzle) -> None:
    ball = build_backward_ball(cycle_puzzle, radius=2, state_cap=1)
    assert not ball.exact_radius_covered


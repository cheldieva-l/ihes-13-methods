from __future__ import annotations

import numpy as np

from ihes13.transforms import (
    complement_state,
    composite_puzzle,
    mirror_state,
    two_move_table,
)


def test_mirror_inverse_roundtrip(cycle_puzzle) -> None:
    state = cycle_puzzle.apply(cycle_puzzle.solved, 0)
    assert np.array_equal(mirror_state(mirror_state(state)), state)


def test_complement_modes_are_explicitly_invertible(cycle_puzzle) -> None:
    goal = cycle_puzzle.apply(cycle_puzzle.solved, 0)
    state = cycle_puzzle.apply(goal, 0)
    left = complement_state(state, goal, mode="left_goal")
    right = complement_state(state, goal, mode="right_goal")
    assert np.array_equal(goal[left], state)
    assert np.array_equal(right[goal], state)


def test_two_move_table_reconstructs_official_moves(cycle_puzzle) -> None:
    table = two_move_table(cycle_puzzle)
    composite = composite_puzzle(cycle_puzzle, table)
    assert len(table.pairs) == 2
    for index, pair in enumerate(table.pairs):
        expected = cycle_puzzle.replay(cycle_puzzle.solved, pair)
        observed = composite.apply(composite.solved, index)
        assert np.array_equal(observed, expected)
        assert table.expand_path([index]) == list(pair)


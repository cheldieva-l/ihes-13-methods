from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihes_dual.puzzle import IHESPuzzle
from ihes13.projection333 import IHES333Projection


@pytest.mark.skipif("IHES_PUZZLE_INFO" not in os.environ, reason="real IHES assets not configured")
def test_real_projection_is_move_equivariant() -> None:
    puzzle = IHESPuzzle.from_puzzle_info(os.environ["IHES_PUZZLE_INFO"])
    projection = IHES333Projection.from_puzzle(puzzle)
    solved = projection.project_state(puzzle.solved)
    assert np.array_equal(solved, np.arange(54))
    for move_index in range(puzzle.generator_count):
        projected = projection.project_state(puzzle.apply(puzzle.solved, move_index))
        assert np.array_equal(np.sort(projected), np.arange(54))


from __future__ import annotations

import numpy as np
import pytest

from ihes_dual.puzzle import IHESPuzzle


@pytest.fixture
def cycle_puzzle() -> IHESPuzzle:
    move = np.asarray([1, 2, 3, 0], dtype=np.uint8)
    inverse = np.asarray([3, 0, 1, 2], dtype=np.uint8)
    return IHESPuzzle(
        solved=np.arange(4, dtype=np.uint8),
        moves=np.stack((move, inverse)),
        move_names=("a", "-a"),
        inverse_move=np.asarray([1, 0], dtype=np.int16),
    )


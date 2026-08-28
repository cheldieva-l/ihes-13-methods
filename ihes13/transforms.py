from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from ihes_dual.puzzle import IHESPuzzle, invert_path, invert_permutation
from ihes_dual.symmetry import SymmetryFrame


def mirror_state(
    state: Sequence[int] | np.ndarray,
    *,
    mode: str = "inverse",
    mirror: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Apply a parameterized ``MirrorState`` transform.

    ``inverse`` mirrors path direction by inverting the state permutation.
    ``conjugate`` applies an explicitly supplied relabelling ``M`` as
    ``M[state[M^-1]]``.  Keeping these modes explicit prevents the historical
    name from silently meaning different algebra in different experiments.
    """

    array = np.asarray(state, dtype=np.uint8)
    if mode == "inverse":
        return invert_permutation(array).astype(np.uint8)
    if mode == "conjugate":
        if mirror is None:
            raise ValueError("conjugate MirrorState requires a mirror permutation")
        permutation = np.asarray(mirror, dtype=np.uint8)
        inverse = invert_permutation(permutation).astype(np.uint8)
        return permutation[array[inverse]]
    raise ValueError(f"unknown MirrorState mode {mode!r}")


def complement_state(
    state: Sequence[int] | np.ndarray,
    goal: Sequence[int] | np.ndarray,
    *,
    mode: str = "left_goal",
) -> np.ndarray:
    """Apply a parameterized goal-relative ``ComplementState`` transform."""

    state_array = np.asarray(state, dtype=np.uint8)
    goal_array = np.asarray(goal, dtype=np.uint8)
    goal_inverse = invert_permutation(goal_array).astype(np.uint8)
    if mode == "left_goal":
        # Relabel values so the goal becomes identity.
        return goal_inverse[state_array]
    if mode == "right_goal":
        # Relabel positions so the goal becomes identity.
        return state_array[goal_inverse]
    raise ValueError(f"unknown ComplementState mode {mode!r}")


def reverse_path(path: Sequence[int], puzzle: IHESPuzzle) -> list[int]:
    return invert_path(path, puzzle.inverse_move)


def _split_move(name: str) -> tuple[bool, str, int]:
    signed = name.startswith("-")
    token = name[1:] if signed else name
    if len(token) != 2 or token[0] not in "frd" or token[1] not in "012":
        raise ValueError(f"unsupported IHES move name {name!r}")
    return signed, token[0], int(token[1])


def axis_substitution(
    puzzle: IHESPuzzle,
    order: str,
    *,
    signs: Sequence[int] = (1, 1, 1),
) -> np.ndarray:
    """Return a name-level move substitution for an axis order and signs.

    This is only a candidate mapping.  ``validated_substitution_frames`` must
    tie it to an exact state relabelling before it is used for a solution.
    """

    if len(order) != 3 or set(order) != set("frd"):
        raise ValueError("axis order must be a permutation of 'frd'")
    if len(signs) != 3 or any(value not in (-1, 1) for value in signs):
        raise ValueError("signs must contain three values from {-1, +1}")
    source_axes = "frd"
    mapping = dict(zip(source_axes, order))
    sign_by_axis = dict(zip(source_axes, signs))
    lookup = {name: index for index, name in enumerate(puzzle.move_names)}
    result = np.empty(puzzle.generator_count, dtype=np.int16)
    for index, name in enumerate(puzzle.move_names):
        negative, axis, layer = _split_move(name)
        flip = sign_by_axis[axis] < 0
        target_negative = negative ^ flip
        target = ("-" if target_negative else "") + mapping[axis] + str(layer)
        result[index] = lookup[target]
    return result


def validated_substitution_frames(
    puzzle: IHESPuzzle,
    frames: Sequence[SymmetryFrame],
    orders: Iterable[str] = ("frd", "rdf", "dft"),
    *,
    include_signed: bool = True,
) -> dict[str, list[int]]:
    """Find exact symmetry frames matching requested axis/sign substitutions."""

    result: dict[str, list[int]] = {}
    sign_options = tuple(product((-1, 1), repeat=3)) if include_signed else ((1, 1, 1),)
    frame_tables = [tuple(map(int, frame.frame_to_original_move)) for frame in frames]
    for order in orders:
        for signs in sign_options:
            key = f"{order}:{','.join(map(str, signs))}"
            try:
                table = tuple(map(int, axis_substitution(puzzle, order, signs=signs)))
            except ValueError:
                # The source brief explicitly includes "dft", although the
                # official IHES axis alphabet is f/r/d.  Preserve it in the
                # audit as an invalid substitution instead of guessing what
                # the unknown "t" should mean.
                result[key] = []
                continue
            matches = [index for index, candidate in enumerate(frame_tables) if candidate == table]
            result[key] = matches
    return result


@dataclass(frozen=True)
class CompositeMoveTable:
    pairs: tuple[tuple[int, int], ...]
    permutations: np.ndarray
    names: tuple[str, ...]
    inverse: np.ndarray

    def expand_path(self, composite_path: Sequence[int]) -> list[int]:
        result: list[int] = []
        for index in composite_path:
            result.extend(self.pairs[int(index)])
        return result


def two_move_table(
    puzzle: IHESPuzzle,
    *,
    exclude_immediate_inverse: bool = True,
) -> CompositeMoveTable:
    pairs = tuple(
        (first, second)
        for first in range(puzzle.generator_count)
        for second in range(puzzle.generator_count)
        if not exclude_immediate_inverse or second != int(puzzle.inverse_move[first])
    )
    permutations = np.asarray(
        [puzzle.moves[first][puzzle.moves[second]] for first, second in pairs],
        dtype=np.uint8,
    )
    names = tuple(f"{puzzle.move_names[first]}|{puzzle.move_names[second]}" for first, second in pairs)
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    inverse = np.asarray(
        [
            pair_to_index[
                (int(puzzle.inverse_move[second]), int(puzzle.inverse_move[first]))
            ]
            for first, second in pairs
        ],
        dtype=np.int16,
    )
    return CompositeMoveTable(pairs, permutations, names, inverse)


def composite_puzzle(puzzle: IHESPuzzle, table: CompositeMoveTable) -> IHESPuzzle:
    composite = IHESPuzzle(
        solved=puzzle.solved.copy(),
        moves=table.permutations.copy(),
        move_names=table.names,
        inverse_move=table.inverse.copy(),
    )
    composite.validate()
    return composite


def double_move_paths(puzzle: IHESPuzzle) -> tuple[tuple[int, int], ...]:
    return tuple((index, index) for index in range(puzzle.generator_count))

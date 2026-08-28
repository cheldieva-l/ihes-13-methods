from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ihes_dual.beam import BeamTrace
from ihes_dual.puzzle import IHESPuzzle


@dataclass(frozen=True)
class BackwardRecord:
    next_key: bytes | None
    move_to_goal: int | None
    distance: int


@dataclass
class BackwardBall:
    puzzle: IHESPuzzle
    radius: int
    records: dict[bytes, BackwardRecord]
    states: dict[bytes, np.ndarray]
    exact_radius_covered: bool
    layer_sizes: list[int]

    def contains(self, state: Sequence[int] | np.ndarray) -> bool:
        return np.ascontiguousarray(state, dtype=np.uint8).tobytes() in self.records

    def path_to_goal(self, state: Sequence[int] | np.ndarray) -> list[int]:
        key = np.ascontiguousarray(state, dtype=np.uint8).tobytes()
        if key not in self.records:
            raise KeyError("state is not in the backward ball")
        result: list[int] = []
        while True:
            record = self.records[key]
            if record.next_key is None:
                return result
            assert record.move_to_goal is not None
            result.append(record.move_to_goal)
            key = record.next_key


def build_backward_ball(
    puzzle: IHESPuzzle,
    radius: int,
    *,
    state_cap: int | None = None,
) -> BackwardBall:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    goal = np.asarray(puzzle.solved, dtype=np.uint8)
    goal_key = np.ascontiguousarray(goal).tobytes()
    records = {goal_key: BackwardRecord(None, None, 0)}
    states = {goal_key: goal.copy()}
    frontier = [goal_key]
    layer_sizes = [1]
    exact = True
    for depth in range(1, radius + 1):
        next_frontier: list[bytes] = []
        for parent_key in frontier:
            parent = states[parent_key]
            for move_index in range(puzzle.generator_count):
                child = puzzle.apply(parent, move_index).astype(np.uint8, copy=False)
                child_key = np.ascontiguousarray(child).tobytes()
                if child_key in records:
                    continue
                if state_cap is not None and len(records) >= state_cap:
                    exact = False
                    return BackwardBall(
                        puzzle, radius, records, states, exact, layer_sizes + [len(next_frontier)]
                    )
                records[child_key] = BackwardRecord(
                    next_key=parent_key,
                    move_to_goal=int(puzzle.inverse_move[move_index]),
                    distance=depth,
                )
                states[child_key] = child.copy()
                next_frontier.append(child_key)
        frontier = next_frontier
        layer_sizes.append(len(frontier))
        if not frontier:
            break
    return BackwardBall(puzzle, radius, records, states, exact, layer_sizes)


@dataclass(frozen=True)
class IntersectionSolution:
    path: tuple[int, ...]
    forward_depth: int
    backward_distance: int
    frontier_index: int


def intersect_trace(
    puzzle: IHESPuzzle,
    start: Sequence[int] | np.ndarray,
    trace: BeamTrace,
    ball: BackwardBall,
    depths: Sequence[int],
) -> IntersectionSolution | None:
    best: IntersectionSolution | None = None
    for depth in depths:
        frontier = trace.frontiers[int(depth)]
        for index, state in enumerate(frontier.states):
            key = np.ascontiguousarray(state, dtype=np.uint8).tobytes()
            record = ball.records.get(key)
            if record is None:
                continue
            prefix = trace.reconstruct(int(depth), index)
            suffix = ball.path_to_goal(state)
            path = prefix + suffix
            if not puzzle.verify_solution(start, path):
                raise AssertionError("forward/backward-ball path failed exact replay")
            candidate = IntersectionSolution(
                tuple(path), int(depth), int(record.distance), int(index)
            )
            if best is None or (
                len(candidate.path), candidate.forward_depth, candidate.frontier_index
            ) < (len(best.path), best.forward_depth, best.frontier_index):
                best = candidate
    return best


from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from ihes_dual.puzzle import IHESPuzzle


FACE_NORMAL = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}
FACE_BASIS = {
    "U": ((1, 0, 0), (0, 0, 1)),
    "R": ((0, 0, -1), (0, -1, 0)),
    "F": ((1, 0, 0), (0, -1, 0)),
    "D": ((1, 0, 0), (0, 0, -1)),
    "L": ((0, 0, 1), (0, -1, 0)),
    "B": ((-1, 0, 0), (0, -1, 0)),
}
BLOCK_FACE = {0: "F", 5: "B", 2: "R", 4: "L", 1: "D", 3: "U"}
MOVE_GEOMETRY = {
    "f0": ("z", +1),
    "f1": ("z", 0),
    "f2": ("z", -1),
    "r0": ("x", +1),
    "r1": ("x", 0),
    "r2": ("x", -1),
    "d0": ("y", -1),
    "d1": ("y", 0),
    "d2": ("y", +1),
}


def _add(*vectors: Sequence[int]) -> tuple[int, int, int]:
    return tuple(sum(parts) for parts in zip(*vectors))  # type: ignore[return-value]


def _scale(vector: Sequence[int], scalar: int) -> tuple[int, int, int]:
    return tuple(scalar * value for value in vector)  # type: ignore[return-value]


def _rotate90(vector: Sequence[int], axis: str, quarter: int) -> tuple[int, int, int]:
    x, y, z = vector
    for _ in range(quarter % 4):
        if axis == "x":
            x, y, z = x, -z, y
        elif axis == "y":
            x, y, z = z, y, -x
        elif axis == "z":
            x, y, z = -y, x, z
        else:
            raise ValueError(axis)
    return int(x), int(y), int(z)


@dataclass(frozen=True)
class StandardGeometry:
    positions: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    faces: tuple[str, ...]
    index: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int]
    centers: np.ndarray
    noncenters: np.ndarray
    noncenter_index: dict[int, int]
    center_for_face: dict[str, int]

    @classmethod
    def create(cls) -> "StandardGeometry":
        positions: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        faces: list[str] = []
        for face in ("U", "R", "F", "D", "L", "B"):
            normal = FACE_NORMAL[face]
            right, down = FACE_BASIS[face]
            for row in range(3):
                for column in range(3):
                    coordinate = _add(
                        normal,
                        _scale(right, column - 1),
                        _scale(down, row - 1),
                    )
                    positions.append((coordinate, normal))
                    faces.append(face)
        position_tuple = tuple(positions)
        index = {position: idx for idx, position in enumerate(position_tuple)}
        centers = np.asarray(
            [idx for idx, (coordinate, normal) in enumerate(position_tuple) if coordinate == normal],
            dtype=np.int16,
        )
        center_set = set(map(int, centers))
        noncenters = np.asarray([idx for idx in range(54) if idx not in center_set], dtype=np.int16)
        return cls(
            positions=position_tuple,
            faces=tuple(faces),
            index=index,
            centers=centers,
            noncenters=noncenters,
            noncenter_index={absolute: reduced for reduced, absolute in enumerate(noncenters)},
            center_for_face={faces[idx]: int(idx) for idx in centers},
        )

    def move(self, axis: str, layer: int, quarter: int) -> np.ndarray:
        permutation = np.arange(54, dtype=np.int16)
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        for source, (coordinate, normal) in enumerate(self.positions):
            if coordinate[axis_index] != layer:
                continue
            destination = self.index[
                (_rotate90(coordinate, axis, quarter), _rotate90(normal, axis, quarter))
            ]
            permutation[destination] = source
        return permutation

    def reduced_move(self, axis: str, layer: int, quarter: int) -> np.ndarray:
        full = self.move(axis, layer, quarter)
        sources = full[self.noncenters]
        return np.asarray([self.noncenter_index[int(source)] for source in sources], dtype=np.int16)


def _orbits(permutations: dict[str, np.ndarray], size: int) -> list[list[int]]:
    unseen = set(range(size))
    result: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        orbit = {seed}
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for permutation in permutations.values():
                nxt = int(permutation[current])
                if nxt not in orbit:
                    orbit.add(nxt)
                    queue.append(nxt)
        unseen -= orbit
        result.append(sorted(orbit))
    return result


@dataclass(frozen=True)
class IHES333Projection:
    puzzle: IHESPuzzle
    geometry: StandardGeometry
    ihes_to_standard: dict[int, int]
    axis_signs: dict[str, int]
    orientations: tuple[np.ndarray, ...]
    notation_to_ihes: dict[str, tuple[int, ...]]

    @classmethod
    def from_puzzle(cls, puzzle: IHESPuzzle) -> "IHES333Projection":
        geometry = StandardGeometry.create()
        ihes_noncenters = np.asarray(
            [idx for idx in range(72) if idx % 12 not in {4, 5, 6, 7}], dtype=np.int16
        )
        ihes_noncenter_index = {
            int(absolute): reduced for reduced, absolute in enumerate(ihes_noncenters)
        }
        name_to_index = {name: idx for idx, name in enumerate(puzzle.move_names)}
        forward_names = ("f0", "f1", "f2", "r0", "r1", "r2", "d0", "d1", "d2")
        reduced_ihes: dict[str, np.ndarray] = {}
        for name in forward_names:
            sources = puzzle.moves[name_to_index[name]][ihes_noncenters]
            if any(int(source) not in ihes_noncenter_index for source in sources):
                raise ValueError("IHES non-center orbit is not closed")
            reduced_ihes[name] = np.asarray(
                [ihes_noncenter_index[int(source)] for source in sources], dtype=np.int16
            )
        orbits = _orbits(reduced_ihes, 48)
        if sorted(map(len, orbits)) != [24, 24]:
            raise ValueError(f"unexpected IHES non-center orbits: {list(map(len, orbits))}")

        def propagate(
            seed_h: int,
            seed_s: int,
            standard_moves: dict[str, np.ndarray],
        ) -> dict[int, int] | None:
            mapping = {seed_h: seed_s}
            reverse = {seed_s: seed_h}
            queue = deque([seed_h])
            while queue:
                h = queue.popleft()
                s = mapping[h]
                for name in forward_names:
                    h2 = int(reduced_ihes[name][h])
                    s2 = int(standard_moves[name][s])
                    expected_face = BLOCK_FACE[int(ihes_noncenters[h2]) // 12]
                    actual_face = geometry.faces[int(geometry.noncenters[s2])]
                    if expected_face != actual_face:
                        return None
                    if h2 in mapping and mapping[h2] != s2:
                        return None
                    if s2 in reverse and reverse[s2] != h2:
                        return None
                    if h2 not in mapping:
                        mapping[h2] = s2
                        reverse[s2] = h2
                        queue.append(h2)
            return mapping

        reduced_mapping: dict[int, int] | None = None
        found_signs: dict[str, int] | None = None
        for sign_f, sign_r, sign_d in product((-1, 1), repeat=3):
            signs = {"f": sign_f, "r": sign_r, "d": sign_d}
            standard_moves = {
                name: geometry.reduced_move(*MOVE_GEOMETRY[name], signs[name[0]])
                for name in forward_names
            }
            combined: dict[int, int] = {}
            used_standard: set[int] = set()
            success = True
            for orbit in orbits:
                seed_h = orbit[0]
                expected_face = BLOCK_FACE[int(ihes_noncenters[seed_h]) // 12]
                candidates = [
                    reduced
                    for reduced, absolute in enumerate(geometry.noncenters)
                    if geometry.faces[int(absolute)] == expected_face
                    and reduced not in used_standard
                ]
                selected = None
                for seed_s in candidates:
                    attempt = propagate(seed_h, seed_s, standard_moves)
                    if (
                        attempt is not None
                        and len(attempt) == len(orbit)
                        and not (set(attempt.values()) & used_standard)
                    ):
                        selected = attempt
                        break
                if selected is None:
                    success = False
                    break
                combined.update(selected)
                used_standard |= set(selected.values())
            if success and len(combined) == 48 and len(set(combined.values())) == 48:
                reduced_mapping = combined
                found_signs = signs
                break
        if reduced_mapping is None or found_signs is None:
            raise RuntimeError("could not derive the IHES-to-3x3 isomorphism")
        ihes_to_standard = {
            int(ihes_noncenters[h]): int(geometry.noncenters[s])
            for h, s in reduced_mapping.items()
        }

        whole_rotations = []
        for axis in ("x", "y", "z"):
            full = np.arange(54, dtype=np.int16)
            for layer in (-1, 0, 1):
                full = full[geometry.move(axis, layer, 1)]
            whole_rotations.append(full)
        orientation_set = {tuple(range(54))}
        queue = deque([np.arange(54, dtype=np.int16)])
        while queue:
            current = queue.popleft()
            for rotation in whole_rotations:
                composed = current[rotation]
                key = tuple(map(int, composed))
                if key not in orientation_set:
                    orientation_set.add(key)
                    queue.append(composed)
        if len(orientation_set) != 24:
            raise AssertionError("whole-cube orientation group is not size 24")
        orientations = tuple(np.asarray(item, dtype=np.int16) for item in orientation_set)

        full_standard_moves: dict[str, np.ndarray] = {}
        for name in puzzle.move_names:
            negative = name.startswith("-")
            positive = name[1:] if negative else name
            axis, layer = MOVE_GEOMETRY[positive]
            quarter = found_signs[positive[0]] * (-1 if negative else 1)
            full_standard_moves[name] = geometry.move(axis, layer, quarter)
        full_lookup = {tuple(map(int, move)): name for name, move in full_standard_moves.items()}
        notation_to_ihes: dict[str, tuple[int, ...]] = {}
        face_axis = {
            "U": ("y", +1), "D": ("y", -1),
            "R": ("x", +1), "L": ("x", -1),
            "F": ("z", +1), "B": ("z", -1),
        }
        for face, (axis, normal_sign) in face_axis.items():
            clockwise = geometry.move(axis, normal_sign, -normal_sign)
            counterclockwise = geometry.move(axis, normal_sign, normal_sign)
            clockwise_name = full_lookup.get(tuple(map(int, clockwise)))
            counterclockwise_name = full_lookup.get(tuple(map(int, counterclockwise)))
            if clockwise_name is None or counterclockwise_name is None:
                raise ValueError(f"ordinary face {face} does not map to an IHES outer move")
            clockwise_index = name_to_index[clockwise_name]
            counterclockwise_index = name_to_index[counterclockwise_name]
            notation_to_ihes[face] = (clockwise_index,)
            notation_to_ihes[face + "'"] = (counterclockwise_index,)
            notation_to_ihes[face + "2"] = (clockwise_index, clockwise_index)
        return cls(
            puzzle=puzzle,
            geometry=geometry,
            ihes_to_standard=ihes_to_standard,
            axis_signs=found_signs,
            orientations=orientations,
            notation_to_ihes=notation_to_ihes,
        )

    def _raw_project_state(self, state: Sequence[int] | np.ndarray) -> np.ndarray:
        ihes_state = np.asarray(state, dtype=np.int16)
        standard = np.empty(54, dtype=np.int16)
        for h_position, s_position in self.ihes_to_standard.items():
            h_label = int(ihes_state[h_position])
            if h_label not in self.ihes_to_standard:
                raise ValueError("a non-center position contains an IHES center label")
            standard[s_position] = self.ihes_to_standard[h_label]
        for block, face in BLOCK_FACE.items():
            labels = ihes_state[block * 12 + np.asarray([4, 5, 6, 7])]
            color_blocks = labels // 12
            if not np.all(color_blocks == color_blocks[0]):
                raise ValueError("center picture has inconsistent face colors")
            color_face = BLOCK_FACE[int(color_blocks[0])]
            standard[self.geometry.center_for_face[face]] = self.geometry.center_for_face[color_face]
        if not np.array_equal(np.sort(standard), np.arange(54)):
            raise ValueError("projected state is not a permutation")
        return standard

    def project_state_with_orientation(
        self, state: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        standard = self._raw_project_state(state)
        for orientation in self.orientations:
            candidate = standard[orientation]
            if np.array_equal(candidate[self.geometry.centers], self.geometry.centers):
                return candidate, orientation
        raise ValueError("no whole-cube orientation puts the six centers home")

    def project_state(self, state: Sequence[int] | np.ndarray) -> np.ndarray:
        projected, _ = self.project_state_with_orientation(state)
        return projected

    def project_state_in_orientation(
        self,
        state: Sequence[int] | np.ndarray,
        orientation: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        return self._raw_project_state(state)[np.asarray(orientation, dtype=np.int16)]

    def facelet_string(self, state: Sequence[int] | np.ndarray) -> str:
        projected = self.project_state(state)
        solved_colors = np.asarray(list("U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9))
        return "".join(solved_colors[projected])

    def translate_notation(self, solution: str) -> list[int]:
        result: list[int] = []
        for token in solution.split():
            if token not in self.notation_to_ihes:
                raise ValueError(f"unsupported ordinary 3x3 token {token!r}")
            result.extend(self.notation_to_ihes[token])
        return result

    def solve_prefix(self, state: Sequence[int] | np.ndarray) -> tuple[list[int], str]:
        try:
            import kociemba
        except ImportError as error:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("method 12 requires the optional 'kociemba' package") from error
        projected, orientation = self.project_state_with_orientation(state)
        solved_colors = np.asarray(list("U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9))
        facelets = "".join(solved_colors[projected])
        notation = kociemba.solve(facelets).strip()

        # Normalizing the six centers may rotate the whole cube.  Derive the
        # official IHES move corresponding to each ordinary face turn in this
        # fixed orientation instead of assuming the identity frame.
        face_axis = {
            "U": ("y", +1), "D": ("y", -1),
            "R": ("x", +1), "L": ("x", -1),
            "F": ("z", +1), "B": ("z", -1),
        }
        standard_tokens: dict[tuple[int, ...], str] = {}
        for face, (axis, normal_sign) in face_axis.items():
            clockwise = self.geometry.move(axis, normal_sign, -normal_sign)
            counterclockwise = self.geometry.move(axis, normal_sign, normal_sign)
            standard_tokens[tuple(map(int, clockwise))] = face
            standard_tokens[tuple(map(int, counterclockwise))] = face + "'"
        projected_inverse = np.argsort(projected)
        dynamic_mapping: dict[str, int] = {}
        for move_index in range(self.puzzle.generator_count):
            moved = self.project_state_in_orientation(
                self.puzzle.apply(state, move_index), orientation
            )
            conjugated = tuple(map(int, projected_inverse[moved]))
            token = standard_tokens.get(conjugated)
            if token is not None:
                dynamic_mapping[token] = move_index
        required = {face for face in face_axis} | {face + "'" for face in face_axis}
        if set(dynamic_mapping) != required:
            missing = sorted(required - set(dynamic_mapping))
            raise AssertionError(f"ordinary-to-IHES move mapping is incomplete: {missing}")
        path: list[int] = []
        for token in notation.split():
            if token.endswith("2"):
                quarter = token[0]
                path.extend((dynamic_mapping[quarter], dynamic_mapping[quarter]))
            else:
                path.append(dynamic_mapping[token])
        residual = self.puzzle.replay(state, path)
        if not np.array_equal(
            self.project_state_in_orientation(residual, orientation), np.arange(54)
        ):
            raise AssertionError("ordinary 3x3 prefix did not solve the projected state")
        return path, notation

"""Exact labelled 54-facelet adapter for official twsearch, retaining moving centers.

No search is performed by this module. All permutations use destination <- source.
The ordered cubie facelets come from the existing ordinary333 geometry artifact.
"""

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
GEOMETRY = Path(os.environ.get("ORDINARY333_GEOMETRY", LOCAL / "assets/geometry.json"))
PUBLIC = Path(os.environ.get("ORDINARY333_PUBLIC", HERE / "data"))
EXE = Path(os.environ.get("TWSEARCH_EXE", "twsearch"))
ORBITS = (("CORNER", 8, 3), ("EDGE", 12, 2), ("CENTER", 6, 1))


@dataclass(frozen=True)
class CubieState:
    cp: tuple
    co: tuple
    ep: tuple
    eo: tuple
    centers: tuple

    def orbits(self):
        return ((self.cp, self.co), (self.ep, self.eo),
                (self.centers, (0,) * 6))

    def as_dict(self):
        return dict(cp=list(self.cp), co=list(self.co), ep=list(self.ep),
                    eo=list(self.eo), centers=list(self.centers))


def native_replay(state, word, moves):
    current = list(state)
    for token in word:
        current = [current[i] for i in moves[token]]
    return current


def to_twsearch(token):
    return token.lstrip("-").upper() + ("'" if token.startswith("-") else "")


def from_twsearch(token):
    if re.fullmatch(r"[FRD][012]'?", token) is None:
        return None
    return ("-" if token.endswith("'") else "") + token.rstrip("'").lower()


class CompactAdapter:
    """Cubie encoding with checked, lossless reconstruction of all labelled stickers."""

    def __init__(self, geometry_path=GEOMETRY):
        self.geometry_path = Path(geometry_path)
        raw = self.geometry_path.read_bytes()
        data = json.loads(raw)
        self.geometry_sha256 = hashlib.sha256(raw).hexdigest()
        self.central = list(data["central_state"])
        self.moves = dict(zip(data["names"], data["moves"]))
        self.pieces = tuple(tuple(tuple(p) for p in data["pieces"] if len(p) == size)
                            for _, _, size in ORBITS)
        self.piece_ids = tuple({frozenset(p): i for i, p in enumerate(pieces)}
                               for pieces in self.pieces)
        self.transforms = {}
        self.errors = []
        for name, _, _ in ORBITS:
            if sum(1 for orbit, _, _ in ORBITS if orbit == name) != 1:
                self.errors.append("duplicate_orbit_name:" + name)
        self.solved, error = self.encode(self.central)
        if error:
            self.errors.append("central_state:" + error)
        for token, permutation in self.moves.items():
            move, error = self.encode(native_replay(self.central, [token], self.moves))
            if error:
                self.errors.append(token + ":" + error)
            else:
                self.transforms[token] = move

    def encode(self, facelets):
        """Return (state, None) or (None, reason); failed inputs remain reportable."""
        if len(facelets) != 54 or sorted(facelets) != list(range(54)):
            return None, "not_a_permutation_of_54_unique_labels"
        encoded = []
        for (name, count, modulus), pieces, lookup in zip(ORBITS, self.pieces, self.piece_ids):
            if len(pieces) != count:
                return None, "geometry_piece_count:" + name
            permutation, orientation = [], []
            for slot, positions in enumerate(pieces):
                labels = tuple(facelets[p] for p in positions)
                identity = lookup.get(frozenset(labels))
                if identity is None:
                    return None, f"nonrigid_piece:{name}:{slot}"
                source = pieces[identity]
                offset = labels.index(source[0])
                expected = tuple(source[(k - offset) % modulus] for k in range(modulus))
                if labels != expected:
                    return None, f"noncyclic_orientation:{name}:{slot}"
                permutation.append(identity)
                orientation.append(offset)
            encoded.append((tuple(permutation), tuple(orientation)))
        return CubieState(*encoded[0], *encoded[1], encoded[2][0]), None

    def decode(self, state):
        result = [-1] * 54
        for (_, _, modulus), pieces, (permutation, orientation) in zip(
                ORBITS, self.pieces, state.orbits()):
            for slot, positions in enumerate(pieces):
                source = pieces[permutation[slot]]
                for k, position in enumerate(positions):
                    result[position] = source[(k - orientation[slot]) % modulus]
        return result

    @staticmethod
    def compose(state, move):
        result = []
        for (_, _, modulus), (p, o), (mp, mo) in zip(ORBITS, state.orbits(), move.orbits()):
            result.append((tuple(p[j] for j in mp),
                           tuple((o[mp[i]] + mo[i]) % modulus for i in range(len(mp)))))
        return CubieState(*result[0], *result[1], result[2][0])

    @staticmethod
    def state_block(state):
        lines = []
        for (name, _, _), (permutation, orientation) in zip(ORBITS, state.orbits()):
            lines.extend([name, " ".join(map(str, permutation)),
                          " ".join(map(str, orientation))])
        return "\n".join(lines) + "\nEnd\n"

    def definition_text(self):
        lines = ["Name Ordinary333ExactCubiesQuarterLayers"]
        lines.extend(f"Set {name} {count} {modulus}" for name, count, modulus in ORBITS)
        result = "\n".join(lines) + "\n\nStartState\n" + self.state_block(self.solved)
        for token in self.moves:
            if not token.startswith("-"):
                result += "\nMoveTransformation " + to_twsearch(token) + "\n"
                result += self.state_block(self.transforms[token])
        return result

    def scramble_text(self, name, state):
        return "ScrambleState " + str(name) + "\n" + self.state_block(state) + "\n"


def read_source_states(public=PUBLIC):
    states, errors = {}, []
    with (Path(public) / "test.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            try:
                puzzle_id = int(row["initial_state_id"])
                state = list(map(int, row["initial_state"].split(",")))
                if puzzle_id in states:
                    errors.append(dict(puzzle_id=puzzle_id, reason="duplicate_source_id"))
                else:
                    states[puzzle_id] = state
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(dict(reason="source_row_parse_error", detail=str(exc)))
    return states, errors


def export_sources(adapter, states, directory=HERE):
    """Export every encodable source state; report failures without replacing them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    definition = directory / "ordinary333_compact_exact18.tws"
    definition.write_text(adapter.definition_text(), encoding="utf-8")
    blocks, records, failures = [], [], []
    for puzzle_id in sorted(states):
        state, reason = adapter.encode(states[puzzle_id])
        if reason:
            failures.append(dict(puzzle_id=puzzle_id, status="F", reason=reason))
            continue
        if adapter.decode(state) != states[puzzle_id]:
            failures.append(dict(puzzle_id=puzzle_id, status="F", reason="source_roundtrip_failed"))
            continue
        blocks.append(adapter.scramble_text(f"puzzle_{puzzle_id}", state))
        records.append(dict(puzzle_id=puzzle_id, **state.as_dict()))
    scrambles = directory / "all_source_states.scr"
    scrambles.write_text("".join(blocks), encoding="utf-8")
    (directory / "source_states.json").write_text(json.dumps(records), encoding="utf-8")
    receipt = dict(source_count=len(states), exported_count=len(records), failures=failures,
                   source_geometry=str(adapter.geometry_path),
                   source_geometry_sha256=adapter.geometry_sha256,
                   definition=str(definition), scrambles=str(scrambles),
                   definition_sha256=hashlib.sha256(definition.read_bytes()).hexdigest(),
                   scrambles_sha256=hashlib.sha256(scrambles.read_bytes()).hexdigest(),
                   metric="18 signed quarter-layer moves; exact 54 labels; moving centers retained",
                   centers_normalized=False, python_state_value_count=46,
                   twsearch_state_bytes=52, original_twsearch_state_bytes=108)
    (directory / "export_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt



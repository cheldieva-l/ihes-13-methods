from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from ihes_dual.assets import CompetitionAssets, find_competition_assets
from ihes_dual.puzzle import IHESPuzzle
from ihes_dual.submission import validate_submission


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_length(text: str | float | None) -> int:
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
        return 0
    return len(str(text).split("."))


@dataclass(frozen=True)
class BenchmarkRow:
    method_id: int
    method_slug: str
    puzzle_id: int
    beam_width: int
    solution: str | None
    solution_length: int | None
    replay_valid: bool
    runtime_seconds: float
    model_id: str
    checkpoint_sha256: str
    reference_identity: str
    reference_length: int
    delta_vs_reference: int | None
    verdict: str
    run_status: str
    metadata: dict[str, object]


class BenchmarkSession:
    def __init__(
        self,
        asset_root: str | Path,
        output_root: str | Path,
        *,
        method_id: int,
        method_slug: str,
        beam_width: int,
        model_id: str,
        checkpoint_sha256: str,
        reference_submission: str | Path | None = None,
    ) -> None:
        self.assets: CompetitionAssets = find_competition_assets(asset_root)
        self.puzzle = IHESPuzzle.from_puzzle_info(self.assets.puzzle_info)
        self.test = pd.read_csv(self.assets.test_csv).sort_values("initial_state_id")
        if not np.array_equal(
            self.test["initial_state_id"].to_numpy(dtype=int), np.arange(len(self.test))
        ):
            raise ValueError("test IDs are not exactly 0..N-1")
        self.states = np.stack(
            self.test["initial_state"].map(
                lambda value: np.fromstring(value, sep=",", dtype=np.uint8)
            )
        )
        self.sample = pd.read_csv(self.assets.sample_submission)
        if list(self.sample.columns) != ["initial_state_id", "path"]:
            raise ValueError("sample submission must have columns initial_state_id,path")
        if reference_submission is None:
            self.reference = self.sample.copy()
            self.reference_identity = "official-sample-submission"
        else:
            reference_path = Path(reference_submission)
            reference = pd.read_csv(reference_path)
            unnamed = [column for column in reference.columns if column.lower().startswith("unnamed:")]
            if unnamed:
                reference = reference.drop(columns=unnamed)
            if list(reference.columns) != ["initial_state_id", "path"]:
                raise ValueError("explicit reference must normalize to initial_state_id,path")
            temporary = Path(output_root) / ".reference.normalized.csv"
            temporary.parent.mkdir(parents=True, exist_ok=True)
            reference.to_csv(temporary, index=False)
            validate_submission(temporary, self.assets.test_csv, self.puzzle)
            temporary.unlink(missing_ok=True)
            self.reference = reference
            self.reference_identity = f"sha256:{sha256_file(reference_path)}"
        self.reference_lengths = {
            int(row.initial_state_id): _path_length(row.path)
            for row in self.reference.itertuples(index=False)
        }
        self.selected = self.reference.copy()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.method_id = int(method_id)
        self.method_slug = method_slug
        self.beam_width = int(beam_width)
        self.model_id = str(model_id)
        self.checkpoint_sha256 = checkpoint_sha256
        self.rows: list[BenchmarkRow] = []

    def state(self, puzzle_id: int) -> np.ndarray:
        return self.states[int(puzzle_id)].copy()

    def reference_path(self, puzzle_id: int) -> list[int]:
        text = self.reference.loc[
            self.reference["initial_state_id"] == int(puzzle_id), "path"
        ].iloc[0]
        return self.puzzle.decode_path(str(text))

    def record(
        self,
        puzzle_id: int,
        path: Sequence[int] | None,
        runtime_seconds: float,
        *,
        run_status: str = "completed",
        metadata: dict[str, object] | None = None,
    ) -> BenchmarkRow:
        reference_length = self.reference_lengths[int(puzzle_id)]
        replay_valid = path is not None and self.puzzle.verify_solution(self.state(puzzle_id), path)
        encoded = self.puzzle.encode_path(path) if replay_valid and path is not None else None
        length = len(path) if replay_valid and path is not None else None
        delta = None if length is None else int(length - reference_length)
        verdict = "progressive" if delta is not None and delta < 0 else "not-progressive"
        if run_status in {"error", "invalid", "truncated"}:
            verdict = run_status
        row = BenchmarkRow(
            method_id=self.method_id,
            method_slug=self.method_slug,
            puzzle_id=int(puzzle_id),
            beam_width=self.beam_width,
            solution=encoded,
            solution_length=length,
            replay_valid=bool(replay_valid),
            runtime_seconds=round(float(runtime_seconds), 6),
            model_id=self.model_id,
            checkpoint_sha256=self.checkpoint_sha256,
            reference_identity=self.reference_identity,
            reference_length=reference_length,
            delta_vs_reference=delta,
            verdict=verdict,
            run_status=run_status,
            metadata=metadata or {},
        )
        self.rows.append(row)
        if replay_valid and length is not None and length < reference_length:
            self.selected.loc[
                self.selected["initial_state_id"] == int(puzzle_id), "path"
            ] = encoded
        self._checkpoint()
        return row

    def _checkpoint(self) -> None:
        rows_path = self.output_root / "benchmark_rows.json"
        rows_path.write_text(
            json.dumps([asdict(row) for row in self.rows], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.selected.to_csv(self.output_root / "submission.partial.csv", index=False)

    def finalize(self, benchmark_ids: Sequence[int]) -> dict[str, object]:
        expected = tuple(map(int, benchmark_ids))
        requested_rows_recorded, completed, status_counts = _completion_status(
            self.rows, expected
        )
        submission_path = self.output_root / "submission.csv"
        self.selected.to_csv(submission_path, index=False)
        validation = validate_submission(submission_path, self.assets.test_csv, self.puzzle)
        selected_lengths = {
            int(row.initial_state_id): _path_length(row.path)
            for row in self.selected.itertuples(index=False)
            if int(row.initial_state_id) in expected
        }
        reference_total = sum(self.reference_lengths[puzzle_id] for puzzle_id in expected)
        selected_total = sum(selected_lengths[puzzle_id] for puzzle_id in expected)
        improved = sum(
            selected_lengths[puzzle_id] < self.reference_lengths[puzzle_id]
            for puzzle_id in expected
        )
        full_exact = completed and all(
            row.run_status == "completed" and row.replay_valid
            for row in self.rows
            if row.puzzle_id in expected
        )
        verdict = (
            "progressive"
            if completed and improved > 0 and selected_total < reference_total
            else "not-progressive"
        )
        summary = {
            "method_id": self.method_id,
            "method_slug": self.method_slug,
            "puzzle_ids": list(expected),
            "beam_width": self.beam_width,
            "completed": completed,
            "requested_rows_recorded": requested_rows_recorded,
            "run_status_counts": status_counts,
            "all_method_paths_replay_valid": full_exact,
            "reference_identity": self.reference_identity,
            "reference_total": reference_total,
            "selected_total": selected_total,
            "strictly_improved_puzzles": improved,
            "method_verdict": verdict if completed else "failed",
            "model_id": self.model_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "submission_validation": validation,
        }
        (self.output_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


def _completion_status(
    rows: Sequence[BenchmarkRow], expected: Sequence[int]
) -> tuple[bool, bool, dict[str, int]]:
    expected_ids = tuple(map(int, expected))
    requested = [row for row in rows if row.puzzle_id in expected_ids]
    observed = tuple(row.puzzle_id for row in requested)
    requested_rows_recorded = (
        len(observed) == len(expected_ids) and set(observed) == set(expected_ids)
    )
    status_counts: dict[str, int] = {}
    for row in requested:
        status_counts[row.run_status] = status_counts.get(row.run_status, 0) + 1
    failed_statuses = {"error", "invalid", "truncated"}
    completed = requested_rows_recorded and not any(
        status_counts.get(status, 0) for status in failed_statuses
    )
    return requested_rows_recorded, completed, status_counts


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.started

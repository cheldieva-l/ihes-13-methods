from __future__ import annotations

from types import SimpleNamespace

from ihes13.benchmark import _completion_status


def _row(puzzle_id: int, status: str) -> SimpleNamespace:
    return SimpleNamespace(puzzle_id=puzzle_id, run_status=status)


def test_recorded_error_is_not_completed() -> None:
    rows = [_row(100, "completed"), _row(101, "error")]
    recorded, completed, counts = _completion_status(rows, (100, 101))
    assert recorded is True
    assert completed is False
    assert counts == {"completed": 1, "error": 1}


def test_not_found_is_a_completed_search_attempt() -> None:
    rows = [_row(100, "completed"), _row(101, "not_found")]
    recorded, completed, counts = _completion_status(rows, (100, 101))
    assert recorded is True
    assert completed is True
    assert counts == {"completed": 1, "not_found": 1}


def test_duplicate_row_does_not_satisfy_coverage() -> None:
    rows = [_row(100, "completed"), _row(100, "completed")]
    recorded, completed, _ = _completion_status(rows, (100, 101))
    assert recorded is False
    assert completed is False

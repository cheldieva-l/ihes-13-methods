from __future__ import annotations

import json

import pandas as pd
import pytest

from ihes13.benchmark import BenchmarkSession


def _write_assets(root) -> None:
    assets = root / "competition"
    assets.mkdir(parents=True)
    (assets / "puzzle_info.json").write_text(
        json.dumps(
            {
                "central_state": [0, 1, 2, 3],
                "generators": {
                    "a": [1, 2, 3, 0],
                    "-a": [3, 0, 1, 2],
                },
            }
        ),
        encoding="utf-8",
    )
    initial = "1,2,3,0"
    pd.DataFrame(
        {"initial_state_id": [0, 1], "initial_state": [initial, initial]}
    ).to_csv(assets / "test.csv", index=False)
    pd.DataFrame(
        {"initial_state_id": [0, 1], "path": ["-a", "-a"]}
    ).to_csv(assets / "sample_submission.csv", index=False)


def _session(asset_root, output_root, *, beam_width=2, resume_from=None):
    return BenchmarkSession(
        asset_root,
        output_root,
        method_id=1,
        method_slug="parameterized_transforms",
        beam_width=beam_width,
        model_id="model-test",
        checkpoint_sha256="checkpoint-test",
        resume_from=resume_from,
    )


def test_resume_replays_rows_and_reports_pending_ids(tmp_path) -> None:
    asset_root = tmp_path / "assets"
    _write_assets(asset_root)
    source_root = tmp_path / "source"
    source = _session(asset_root, source_root)
    source.record(0, [1], 1.25)

    resumed = _session(
        asset_root,
        tmp_path / "resumed",
        resume_from=source_root,
    )
    assert resumed.completed_ids == {0}
    assert resumed.resumed_puzzle_ids == (0,)
    summary = resumed.finalize((0, 1))
    assert summary["completed"] is False
    assert summary["method_verdict"] == "pending"
    assert summary["pending_puzzle_ids"] == [1]


def test_resume_rejects_identity_mismatch(tmp_path) -> None:
    asset_root = tmp_path / "assets"
    _write_assets(asset_root)
    source_root = tmp_path / "source"
    source = _session(asset_root, source_root)
    source.record(0, [1], 1.25)

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _session(
            asset_root,
            tmp_path / "resumed",
            beam_width=3,
            resume_from=source_root,
        )

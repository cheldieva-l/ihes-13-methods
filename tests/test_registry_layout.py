from __future__ import annotations

import json
from pathlib import Path

from ihes13.registry import METHODS


ROOT = Path(__file__).resolve().parents[1]


def test_exactly_thirteen_methods() -> None:
    assert [item.method_id for item in METHODS] == list(range(1, 14))


def test_every_method_is_self_contained() -> None:
    for method in METHODS:
        directory = ROOT / "methods" / method.directory_name
        assert (directory / "README.md").is_file()
        assert (directory / "run.py").is_file()
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert config["method_id"] == method.method_id


def test_exactly_thirteen_notebooks_and_metadata() -> None:
    assert len(list((ROOT / "notebooks").glob("*.ipynb"))) == 13
    assert len(list((ROOT / "kaggle-metadata").glob("*.json"))) == 13


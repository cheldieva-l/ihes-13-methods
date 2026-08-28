from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_english_unexecuted_and_emit_submission() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        text = "\n".join(str(cell.get("source", "")) for cell in payload["cells"])
        assert not re.search(r"[А-Яа-яЁё]", text)
        assert "submission.csv" in text
        assert "365_000" in text
        assert "tuple(range(100, 121))" in text
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


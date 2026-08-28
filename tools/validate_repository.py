from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ihes13.registry import METHODS


FORBIDDEN_NAMES = {
    "kaggle.json",
    ".env",
    "credentials.json",
    "token.json",
}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".rar", ".7z", ".zip"
}
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.name.lower() in FORBIDDEN_NAMES:
            fail(f"forbidden credential filename: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"forbidden binary/archive file: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            fail(f"oversized repository file: {relative}")
        if path.suffix.lower() in {".py", ".md", ".json", ".toml", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="strict")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    fail(f"secret-like content in {relative}: {pattern.pattern}")

    for method in METHODS:
        directory = ROOT / "methods" / method.directory_name
        for name in ("README.md", "config.json", "run.py"):
            if not (directory / name).is_file():
                fail(f"method {method.method_id} is missing {name}")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("method_id") != method.method_id:
            fail(f"method {method.method_id} config has the wrong id")

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 13:
        fail(f"expected 13 notebooks, found {len(notebooks)}")
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        text = "\n".join(str(cell.get("source", "")) for cell in payload["cells"])
        if re.search(r"[А-Яа-яЁё]", text):
            fail(f"non-English Cyrillic text in {notebook.name}")
        if "submission.csv" not in text or "365_000" not in text:
            fail(f"benchmark/submission contract missing from {notebook.name}")
        for cell in payload["cells"]:
            if cell["cell_type"] == "code" and (cell["execution_count"] is not None or cell["outputs"]):
                fail(f"executed output committed in {notebook.name}")

    for required in (
        "README.md", "METHOD.md", "RESULTS.md", "THIRD_PARTY_NOTICES.md",
        "SECURITY.md", "LICENSE", "licenses/IHES-DUAL-MODEL-MIT.txt",
    ):
        if not (ROOT / required).is_file():
            fail(f"missing required repository document: {required}")
    print(json.dumps({"files": len(files), "methods": 13, "notebooks": 13, "status": "ok"}))


if __name__ == "__main__":
    main()


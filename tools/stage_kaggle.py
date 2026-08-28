from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ihes13.registry import get_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage one public Kaggle notebook for upload")
    parser.add_argument("method_id", type=int, choices=range(1, 14))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    method = get_method(args.method_id)
    name = f"{method.method_id:02d}_{method.slug}.ipynb"
    source_notebook = ROOT / "notebooks" / name
    source_metadata = ROOT / "kaggle-metadata" / f"{method.method_id:02d}_{method.slug}.json"
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_notebook, destination / name)
    shutil.copy2(source_metadata, destination / "kernel-metadata.json")
    print(json.dumps({"staged": str(destination), "notebook": name}, indent=2))


if __name__ == "__main__":
    main()

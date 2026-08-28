from __future__ import annotations

import json
from pathlib import Path
import sys
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ihes13.registry import METHODS


NOTEBOOKS = ROOT / "notebooks"
METADATA = ROOT / "kaggle-metadata"
REPOSITORY_URL = "https://github.com/cheldieva-l/ihes-13-methods"


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


def build_notebook(method_id: int, title: str, summary: str) -> dict[str, object]:
    return {
        "cells": [
            markdown(
                f"""
                # IHES Method {method_id:02d} — {title}

                {summary}

                This notebook is one of thirteen controlled experiments. It runs a small smoke search first and then the requested T4 benchmark for puzzle IDs 100–120 at beam width 365000. Every accepted path is replayed with the 18 official generators. The final `submission.csv` is replay-validated for all 1,003 competition rows.

                The notebook never contains credentials, competition data, participant submissions, or model weights. All substantial external ideas and licenses are documented in the public repository.
                """
            ),
            code(
                f"""
                from pathlib import Path

                METHOD_ID = {method_id}
                RUN_SMOKE = True
                RUN_FULL = True
                SMOKE_PUZZLE_IDS = (100,)
                FULL_PUZZLE_IDS = tuple(range(100, 121))
                SMOKE_BEAM_WIDTH = 64
                FULL_BEAM_WIDTH = 365_000
                DEVICE = "cuda"
                ASSET_ROOT = Path("/kaggle/input")
                MODEL_ROOT = Path("/kaggle/input")
                WORKING = Path("/kaggle/working")
                REFERENCE_SUBMISSION = None  # Optional attached replay-valid control CSV.
                """
            ),
            code(
                f"""
                import shutil
                import subprocess
                import sys

                repository_url = {REPOSITORY_URL!r}
                checkout = WORKING / "ihes-13-methods"
                if checkout.exists():
                    shutil.rmtree(checkout)
                subprocess.run(
                    ["git", "clone", "--depth", "1", repository_url, str(checkout)],
                    check=True,
                )
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", str(checkout), "--no-deps", "-q"],
                    check=True,
                )
                if METHOD_ID == 12:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "kociemba>=1.2", "-q"],
                        check=True,
                    )
                sys.path.insert(0, str(checkout))
                repository_commit = subprocess.check_output(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
                ).strip()
                print({{"repository_commit": repository_commit, "method_id": METHOD_ID}})
                """
            ),
            code(
                """
                import json
                import torch
                from ihes13.runner import load_method_config, run_experiment

                method_config = load_method_config(METHOD_ID)
                print(json.dumps(method_config, indent=2, sort_keys=True))
                if DEVICE.startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError("A Kaggle GPU session is required for the requested benchmark")
                print({
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "asset_root": str(ASSET_ROOT),
                })
                """
            ),
            code(
                """
                smoke_summary = None
                if RUN_SMOKE:
                    smoke_summary = run_experiment(
                        METHOD_ID,
                        asset_root=ASSET_ROOT,
                        model_root=MODEL_ROOT,
                        output_root=WORKING / f"method-{METHOD_ID:02d}" / "smoke",
                        puzzle_ids=SMOKE_PUZZLE_IDS,
                        beam_width=SMOKE_BEAM_WIDTH,
                        device=DEVICE,
                        smoke=True,
                        reference_submission=REFERENCE_SUBMISSION,
                    )
                    print("Smoke summary")
                    print(json.dumps(smoke_summary, indent=2, sort_keys=True))
                """
            ),
            code(
                """
                full_summary = None
                if RUN_FULL:
                    full_output = WORKING / f"method-{METHOD_ID:02d}" / "full"
                    full_summary = run_experiment(
                        METHOD_ID,
                        asset_root=ASSET_ROOT,
                        model_root=MODEL_ROOT,
                        output_root=full_output,
                        puzzle_ids=FULL_PUZZLE_IDS,
                        beam_width=FULL_BEAM_WIDTH,
                        device=DEVICE,
                        smoke=False,
                        reference_submission=REFERENCE_SUBMISSION,
                    )
                    shutil.copy2(full_output / "submission.csv", WORKING / "submission.csv")
                    shutil.copy2(full_output / "benchmark_rows.json", WORKING / f"method_{METHOD_ID:02d}_benchmark_rows.json")
                    shutil.copy2(full_output / "summary.json", WORKING / f"method_{METHOD_ID:02d}_summary.json")
                    print("Full benchmark summary")
                    print(json.dumps(full_summary, indent=2, sort_keys=True))
                """
            ),
            code(
                """
                from ihes_dual.assets import find_competition_assets
                from ihes_dual.puzzle import IHESPuzzle
                from ihes_dual.submission import validate_submission

                final_submission = WORKING / "submission.csv"
                if not final_submission.is_file():
                    source = WORKING / f"method-{METHOD_ID:02d}" / "smoke" / "submission.csv"
                    shutil.copy2(source, final_submission)
                assets = find_competition_assets(ASSET_ROOT)
                puzzle = IHESPuzzle.from_puzzle_info(assets.puzzle_info)
                validation = validate_submission(final_submission, assets.test_csv, puzzle)
                print({"submission": str(final_submission), "validation": validation})
                """
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "dataSources": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        name = f"{method.method_id:02d}_{method.slug}.ipynb"
        notebook_path = NOTEBOOKS / name
        notebook_path.write_text(
            json.dumps(
                build_notebook(method.method_id, method.title, method.summary),
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata = {
            "id": f"cheldieva-l/ihes-method-{method.method_id:02d}-{method.slug.replace('_', '-')}",
            "title": f"IHES Method {method.method_id:02d} - {method.title}",
            "code_file": name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": False,
            "enable_gpu": True,
            "enable_internet": True,
            "competition_sources": ["cayleypy-ihes-cube"],
            "dataset_sources": ["arabidopsisthalian/ihes-model-1778521793"],
        }
        (METADATA / f"{method.method_id:02d}_{method.slug}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(notebook_path)


if __name__ == "__main__":
    main()

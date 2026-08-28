from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

import torch

from ihes_dual.assets import find_competition_assets
from ihes_dual.puzzle import IHESPuzzle

from .benchmark import BenchmarkSession, Timer
from .methods_impl import prepare_method, solve_prepared
from .registry import BENCHMARK_IDS, DEFAULT_BEAM_WIDTH, get_method


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_method_config(method_id: int) -> dict[str, object]:
    spec = get_method(method_id)
    path = spec.config_path(REPOSITORY_ROOT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("method_id", -1)) != spec.method_id:
        raise ValueError(f"{path} has the wrong method_id")
    return payload


def _device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def run_experiment(
    method_id: int,
    *,
    asset_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    puzzle_ids: Sequence[int] = BENCHMARK_IDS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    device: str = "auto",
    smoke: bool = False,
    reference_submission: str | Path | None = None,
) -> dict[str, object]:
    spec = get_method(method_id)
    config = load_method_config(method_id)
    selected_device = _device(device)
    assets = find_competition_assets(asset_root)
    puzzle = IHESPuzzle.from_puzzle_info(assets.puzzle_info)
    prepared = prepare_method(
        spec,
        config,
        puzzle,
        asset_root=asset_root,
        model_root=model_root,
        output_root=output_root,
        device=selected_device,
        smoke=smoke,
    )
    session = BenchmarkSession(
        asset_root,
        output_root,
        method_id=spec.method_id,
        method_slug=spec.slug,
        beam_width=int(beam_width),
        model_id=prepared.model_id,
        checkpoint_sha256=prepared.checkpoint_sha256,
        reference_submission=reference_submission,
    )
    for puzzle_id in map(int, puzzle_ids):
        start = session.state(puzzle_id)
        reference_path = session.reference_path(puzzle_id) if spec.kind == "dual_center" else None
        try:
            with Timer() as timer:
                solution = solve_prepared(
                    prepared,
                    puzzle,
                    start,
                    puzzle_id=puzzle_id,
                    beam_width=int(beam_width),
                    smoke=smoke,
                    reference_path=reference_path,
                    output_root=output_root,
                )
            session.record(
                puzzle_id,
                solution.path,
                timer.elapsed,
                run_status=solution.run_status,
                metadata=solution.metadata,
            )
        except Exception as error:
            # A per-puzzle failure remains explicit while the notebook can still
            # emit a fully valid reference-backed submission for inspection.
            session.record(
                puzzle_id,
                None,
                getattr(locals().get("timer"), "elapsed", 0.0),
                run_status="error",
                metadata={"error_type": type(error).__name__, "error": str(error)},
            )
    summary = session.finalize(tuple(map(int, puzzle_ids)))
    summary["device"] = selected_device
    summary["accelerator_name"] = (
        torch.cuda.get_device_name(0)
        if selected_device.startswith("cuda") and torch.cuda.is_available()
        else "CPU"
    )
    summary["smoke"] = bool(smoke)
    summary["preparation"] = prepared.metadata
    (Path(output_root) / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _parse_ids(text: str) -> tuple[int, ...]:
    if "-" in text and "," not in text:
        first, last = map(int, text.split("-", 1))
        return tuple(range(first, last + 1))
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def main_for_method(method_id: int) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run IHES research method {method_id:02d}: {get_method(method_id).title}"
    )
    parser.add_argument("--asset-root", default="/kaggle/input")
    parser.add_argument("--model-root", default="/kaggle/input")
    parser.add_argument("--output-root", default=f"/kaggle/working/method-{method_id:02d}")
    parser.add_argument("--puzzle-ids", default="100-120")
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reference-submission")
    args = parser.parse_args()
    summary = run_experiment(
        method_id,
        asset_root=args.asset_root,
        model_root=args.model_root,
        output_root=args.output_root,
        puzzle_ids=_parse_ids(args.puzzle_ids),
        beam_width=args.beam_width,
        device=args.device,
        smoke=args.smoke,
        reference_submission=args.reference_submission,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one of the 13 IHES research methods")
    parser.add_argument("method_id", type=int, choices=range(1, 14))
    args, remaining = parser.parse_known_args()
    # Rebuild argv for the method-specific parser without inventing a second
    # configuration surface.
    import sys

    sys.argv = [sys.argv[0], *remaining]
    main_for_method(args.method_id)


if __name__ == "__main__":
    main()

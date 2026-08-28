from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ihes_dual.assets import find_symmetry_file
from ihes_dual.beam import BeamConfig, BeamTrace, beam_search
from ihes_dual.model import load_mlp2rb
from ihes_dual.puzzle import IHESPuzzle
from ihes_dual.registry import resolve_model
from ihes_dual.solve import solve_bidirectional, solve_symmetry_reverse
from ihes_dual.symmetry import SymmetryFrame, load_symmetry_frames

from .intersection import BackwardBall, build_backward_ball, intersect_trace
from .projection333 import IHES333Projection
from .registry import MethodSpec
from .scorers import (
    ActionScoreScorer,
    DualCenterGoalScorer,
    DualCenterStudent,
    GFlowNetIHES,
    MultiActionStudent,
    RichNeighborStudent,
    ScalarStudent,
    seeded_codebook,
)
from .training import (
    TrainingReport,
    train_action_student,
    train_dual_center_student,
    train_gflownet,
    train_scalar_student,
)
from .transforms import (
    CompositeMoveTable,
    complement_state,
    composite_puzzle,
    mirror_state,
    two_move_table,
    validated_substitution_frames,
)


@dataclass
class PreparedMethod:
    spec: MethodSpec
    config: dict[str, Any]
    scorer: Callable[[torch.Tensor], torch.Tensor] | Any
    model_id: str
    checkpoint_sha256: str
    device: str
    metadata: dict[str, Any]
    frames: list[SymmetryFrame] | None = None
    composite_table: CompositeMoveTable | None = None
    backward_ball: BackwardBall | None = None
    projection: IHES333Projection | None = None


@dataclass(frozen=True)
class MethodSolution:
    path: tuple[int, ...] | None
    run_status: str
    metadata: dict[str, Any]


def _beam_config(
    config: dict[str, Any],
    device: str,
    *,
    beam_width: int,
    max_depth: int,
    prune_immediate_inverse: bool = False,
    cumulative_alpha: float = 0.0,
) -> BeamConfig:
    return BeamConfig(
        beam_width=int(beam_width),
        max_depth=int(max_depth),
        parent_chunk=min(25_000, max(1, int(beam_width))),
        inference_batch=8192 if device.startswith("cuda") else 128,
        device=device,
        autocast=device.startswith("cuda"),
        prune_immediate_inverse=prune_immediate_inverse,
        cumulative_alpha=float(cumulative_alpha),
        hash_seed=int(config.get("seed", 0x1A2B3C4D)),
    )


def _training_values(config: dict[str, Any], smoke: bool) -> dict[str, int | float]:
    return {
        "steps": 1 if smoke else int(config.get("train_steps", 2500)),
        "batch_size": min(8, int(config.get("batch_size", 512))) if smoke else int(config.get("batch_size", 512)),
        "hidden_size": 64 if smoke else int(config.get("hidden_size", 1024)),
        "max_walk_depth": 4 if smoke else int(config.get("max_depth", config.get("max_official_depth", 29))),
        "learning_rate": float(config.get("learning_rate", 3e-4)),
    }


def prepare_method(
    spec: MethodSpec,
    config: dict[str, Any],
    puzzle: IHESPuzzle,
    *,
    asset_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    device: str,
    smoke: bool,
) -> PreparedMethod:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    frames = None
    composite_table = None
    backward_ball = None
    projection = None
    metadata: dict[str, Any] = {}

    if spec.kind == "gflownet":
        values = _training_values(config, smoke)
        model = GFlowNetIHES(
            action_count=puzzle.generator_count,
            hidden_size=int(values["hidden_size"]),
            residual_blocks=1 if smoke else int(config.get("residual_blocks", 6)),
        )
        report = train_gflownet(
            puzzle,
            model,
            steps=int(values["steps"]),
            batch_size=int(values["batch_size"]),
            trajectory_length=3 if smoke else int(config.get("random_trajectory_length", 29)),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            regularization_coefficient=float(config.get("regularization_coefficient", 1e-6)),
            gradient_clip=float(config.get("gradient_clip", 100.0)),
            seed=int(config["seed"]),
            device=device,
            checkpoint_path=output / "checkpoints" / "method13_gflownet.pt",
        )
        metadata["training"] = asdict(report)
        return PreparedMethod(
            spec,
            config,
            ActionScoreScorer(model),
            str(config.get("model_id", "gfn-ihes-local")),
            report.checkpoint_sha256,
            device,
            metadata,
        )

    base_model_id = str(config.get("model_id", config.get("teacher_model_id", "1778521793")))
    model_spec = resolve_model(model_root, base_model_id)
    base_model = load_mlp2rb(model_spec, device)
    checkpoint_sha256 = next(
        record.sha256
        for record in __import__("ihes_dual.registry", fromlist=["ASSET_REGISTRY"]).ASSET_REGISTRY
        if record.model_id == model_spec.model_id and record.sha256 is not None
    )
    scorer: Any = base_model
    metadata["base_checkpoint"] = {
        "model_id": model_spec.model_id,
        "epoch": model_spec.epoch,
        "filename": model_spec.checkpoint.name,
        "sha256": checkpoint_sha256,
    }

    if spec.kind == "transforms":
        frames_all = load_symmetry_frames(str(find_symmetry_file(asset_root)), puzzle)
        limit = min(len(frames_all), int(config.get("symmetry_limit", len(frames_all))))
        frames = frames_all[:limit]
        substitution_audit = validated_substitution_frames(
            puzzle,
            frames_all,
            config.get("axis_orders", ("frd", "rdf", "dft")),
            include_signed=bool(config.get("include_signed", True)),
        )
        metadata["transform_audit"] = {
            "validated_symmetry_count": len(frames_all),
            "searched_symmetry_pool": limit,
            "axis_sign_substitutions": substitution_audit,
            "mirror_modes": config.get("mirror_modes", []),
            "complement_modes": config.get("complement_modes", []),
            "double_move_count": puzzle.generator_count,
        }
    elif spec.kind in {"two_step_head", "two_move_retrain"}:
        composite_table = two_move_table(puzzle)
        composite = composite_puzzle(puzzle, composite_table)
        if len(composite_table.pairs) != 306:
            raise AssertionError("the non-backtracking two-move action set is not 306")
        values = _training_values(config, smoke)
        if spec.kind == "two_step_head":
            student = MultiActionStudent(
                len(composite_table.pairs),
                hidden_size=int(values["hidden_size"]),
                residual_blocks=1 if smoke else 2,
            )
            report = train_action_student(
                puzzle,
                composite_table.permutations,
                base_model,
                student,
                steps=int(values["steps"]),
                batch_size=int(values["batch_size"]),
                max_walk_depth=int(values["max_walk_depth"]),
                learning_rate=float(values["learning_rate"]),
                seed=int(config["seed"]),
                device=device,
                checkpoint_path=output / "checkpoints" / "method03_two_step_head.pt",
            )
            scorer = ActionScoreScorer(student)
        else:
            student = ScalarStudent(
                hidden_size=int(values["hidden_size"]), residual_blocks=1 if smoke else 2
            )
            report = train_scalar_student(
                composite,
                base_model,
                student,
                steps=int(values["steps"]),
                batch_size=int(values["batch_size"]),
                max_walk_depth=max(2, int(values["max_walk_depth"]) // 2),
                learning_rate=float(values["learning_rate"]),
                seed=int(config["seed"]),
                device=device,
                checkpoint_path=output / "checkpoints" / "method04_two_move_scalar.pt",
            )
            scorer = student
        metadata["training"] = asdict(report)
        metadata["composite_action_count"] = len(composite_table.pairs)
        checkpoint_sha256 = report.checkpoint_sha256
        base_model_id = f"method-{spec.method_id:02d}-student"
    elif spec.kind == "neighbor_head":
        values = _training_values(config, smoke)
        student = MultiActionStudent(
            puzzle.generator_count,
            hidden_size=int(values["hidden_size"]),
            residual_blocks=1 if smoke else 2,
        )
        report = train_action_student(
            puzzle,
            puzzle.moves,
            base_model,
            student,
            steps=int(values["steps"]),
            batch_size=int(values["batch_size"]),
            max_walk_depth=int(values["max_walk_depth"]),
            learning_rate=float(values["learning_rate"]),
            seed=int(config["seed"]),
            device=device,
            checkpoint_path=output / "checkpoints" / "method05_neighbor_head.pt",
        )
        scorer = ActionScoreScorer(student)
        metadata["training"] = asdict(report)
        checkpoint_sha256 = report.checkpoint_sha256
        base_model_id = "method-05-neighbor-student"
    elif spec.kind == "rich_input":
        values = _training_values(config, smoke)
        codebook, codebook_sha256 = seeded_codebook(
            puzzle.state_size,
            int(config.get("random_code_dim", 16)),
            int(config["seed"]),
        )
        student = RichNeighborStudent(
            puzzle, codebook, hidden_size=int(values["hidden_size"])
        )
        report = train_scalar_student(
            puzzle,
            base_model,
            student,
            steps=int(values["steps"]),
            batch_size=int(values["batch_size"]),
            max_walk_depth=int(values["max_walk_depth"]),
            learning_rate=float(values["learning_rate"]),
            seed=int(config["seed"]),
            device=device,
            checkpoint_path=output / "checkpoints" / "method07_rich_input.pt",
        )
        scorer = student
        metadata["training"] = asdict(report)
        metadata["random_codebook_sha256"] = codebook_sha256
        checkpoint_sha256 = report.checkpoint_sha256
        base_model_id = "method-07-rich-student"
    elif spec.kind == "radius_intersection":
        radius = int(config["intersection_radius"])
        cap = config.get("backward_state_cap")
        if smoke and cap is None:
            cap = 50_000
        backward_ball = build_backward_ball(
            puzzle, radius if not smoke else min(radius, 3), state_cap=None if cap is None else int(cap)
        )
        metadata["backward_ball"] = {
            "requested_radius": radius,
            "built_radius": radius if not smoke else min(radius, 3),
            "states": len(backward_ball.records),
            "layer_sizes": backward_ball.layer_sizes,
            "exact_radius_covered": backward_ball.exact_radius_covered,
        }
    elif spec.kind == "projection333":
        projection = IHES333Projection.from_puzzle(puzzle)
        metadata["projection"] = {
            "isomorphism_size": len(projection.ihes_to_standard),
            "axis_signs": projection.axis_signs,
            "whole_cube_orientations": len(projection.orientations),
        }
    elif spec.kind == "dual_center":
        base_model_id = "method-11-per-puzzle-dual-center"
        checkpoint_sha256 = "per-puzzle-see-row-metadata"

    return PreparedMethod(
        spec,
        config,
        scorer,
        base_model_id,
        str(checkpoint_sha256),
        device,
        metadata,
        frames=frames,
        composite_table=composite_table,
        backward_ball=backward_ball,
        projection=projection,
    )


def _odd_completion(
    puzzle: IHESPuzzle,
    start: np.ndarray,
    trace: BeamTrace,
    table: CompositeMoveTable,
) -> list[int] | None:
    best: list[int] | None = None
    for depth, frontier in trace.frontiers.items():
        for index, state in enumerate(frontier.states):
            children = state[puzzle.moves]
            matches = np.flatnonzero(np.all(children == puzzle.solved, axis=1))
            for move in matches:
                composite_path = trace.reconstruct(depth, index)
                candidate = table.expand_path(composite_path) + [int(move)]
                if puzzle.verify_solution(start, candidate) and (
                    best is None or len(candidate) < len(best)
                ):
                    best = candidate
    return best


def solve_prepared(
    prepared: PreparedMethod,
    puzzle: IHESPuzzle,
    start: np.ndarray,
    *,
    puzzle_id: int,
    beam_width: int,
    smoke: bool,
    reference_path: Sequence[int] | None = None,
    output_root: str | Path | None = None,
) -> MethodSolution:
    config = prepared.config
    width = min(64, beam_width) if smoke else int(beam_width)
    metadata: dict[str, Any] = dict(prepared.metadata)
    path: Sequence[int] | None = None
    status = "completed"

    if prepared.spec.kind == "transforms":
        assert prepared.frames is not None
        identity = next(
            frame for frame in prepared.frames if np.array_equal(frame.rotation, np.arange(puzzle.state_size))
        )
        search_count = 1 if smoke else int(config.get("full_search_frames", 2))
        frames = [identity] + [frame for frame in prepared.frames if frame is not identity][: max(0, search_count - 1)]
        # Explicit per-state audit of MirrorState and ComplementState.
        mirror_inverse = mirror_state(start, mode="inverse")
        mirror_roundtrip = np.array_equal(mirror_state(mirror_inverse, mode="inverse"), start)
        complement_checks = {
            "left_goal": np.array_equal(
                puzzle.solved[complement_state(start, puzzle.solved, mode="left_goal")], start
            ),
            "right_goal": np.array_equal(
                complement_state(start, puzzle.solved, mode="right_goal")[puzzle.solved], start
            ),
        }
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=6 if smoke else int(config["max_depth"]),
        )
        best, runs = solve_symmetry_reverse(
            puzzle,
            start,
            prepared.scorer,
            run_config,
            frames,
            model_id=prepared.model_id,
            include_direct=bool(config.get("include_direct", True)),
            include_reverse=bool(config.get("include_reverse", True)),
        )
        path = None if best is None else best.path
        metadata.update(
            {
                "searched_frames": len(frames),
                "mirror_inverse_roundtrip": mirror_roundtrip,
                "complement_roundtrips": complement_checks,
                "runs": runs,
            }
        )
    elif prepared.spec.kind == "cumulative":
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=6 if smoke else int(config["max_depth"]),
            cumulative_alpha=float(config["cumulative_alpha"]),
        )
        trace = beam_search(puzzle, start, prepared.scorer, run_config)
        path = trace.solution
        metadata["depths"] = trace.diagnostic_report()["depths"]
    elif prepared.spec.kind in {"two_step_head", "two_move_retrain"}:
        assert prepared.composite_table is not None
        composite = composite_puzzle(puzzle, prepared.composite_table)
        maximum = 6 if smoke else int(config["max_official_depth"])
        composite_depth = math.ceil(maximum / 2)
        retain = tuple(range(1, composite_depth + 1))
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=composite_depth,
        )
        trace = beam_search(
            composite,
            start,
            prepared.scorer,
            run_config,
            retain_depths=retain,
        )
        if trace.solution is not None:
            path = prepared.composite_table.expand_path(trace.solution)
        else:
            path = _odd_completion(puzzle, start, trace, prepared.composite_table)
        metadata["composite_depths"] = trace.diagnostic_report()["depths"]
    elif prepared.spec.kind in {"neighbor_head", "rich_input", "inverse_prune"}:
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=6 if smoke else int(config["max_depth"]),
            prune_immediate_inverse=prepared.spec.kind == "inverse_prune",
        )
        trace = beam_search(puzzle, start, prepared.scorer, run_config)
        path = trace.solution
        metadata["depths"] = trace.diagnostic_report()["depths"]
    elif prepared.spec.kind == "bidirectional":
        frame = SymmetryFrame.identity(puzzle)
        forward_depths = tuple(range(1, 4)) if smoke else tuple(map(int, config["forward_depths"]))
        reverse_depths = tuple(range(1, 4)) if smoke else tuple(map(int, config["reverse_depths"]))
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=max(max(forward_depths), max(reverse_depths)),
        )
        run = solve_bidirectional(
            puzzle,
            start,
            prepared.scorer,
            run_config,
            frame,
            forward_depths=forward_depths,
            reverse_depths=reverse_depths,
        )
        path = None if run.joined is None else run.joined.original_path
        metadata["meeting"] = None if run.joined is None else asdict(run.joined.meeting)
    elif prepared.spec.kind == "radius_intersection":
        assert prepared.backward_ball is not None
        maximum = 6 if smoke else int(config["max_forward_depth"])
        depths = tuple(range(1, maximum + 1))
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=maximum,
        )
        trace = beam_search(
            puzzle, start, prepared.scorer, run_config, retain_depths=depths
        )
        joined = intersect_trace(puzzle, start, trace, prepared.backward_ball, depths)
        path = None if joined is None else joined.path
        if not prepared.backward_ball.exact_radius_covered:
            status = "truncated"
        metadata["intersection"] = None if joined is None else asdict(joined)
    elif prepared.spec.kind == "dual_center":
        if reference_path is None:
            raise ValueError("method 11 requires a replay-valid reference path")
        values = _training_values(config, smoke)
        model = DualCenterStudent(
            hidden_size=int(values["hidden_size"]), residual_blocks=1 if smoke else 2
        )
        checkpoint = None
        if output_root is not None:
            checkpoint = Path(output_root) / "checkpoints" / f"method11_puzzle{puzzle_id}.pt"
        report = train_dual_center_student(
            puzzle,
            start,
            reference_path,
            model,
            steps=int(values["steps"]),
            learning_rate=float(values["learning_rate"]),
            seed=int(config["seed"]) + int(puzzle_id),
            device=prepared.device,
            checkpoint_path=checkpoint,
        )
        scorer = DualCenterGoalScorer(model)
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=6 if smoke else int(config["max_depth"]),
        )
        trace = beam_search(puzzle, start, scorer, run_config)
        path = trace.solution
        metadata["per_puzzle_training"] = asdict(report)
    elif prepared.spec.kind == "projection333":
        assert prepared.projection is not None
        prefix, notation = prepared.projection.solve_prefix(start)
        residual = puzzle.replay(start, prefix)
        remaining_depth = 6 if smoke else int(config["max_depth_after_prefix"])
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=remaining_depth,
        )
        trace = beam_search(puzzle, residual, prepared.scorer, run_config)
        path = None if trace.solution is None else prefix + trace.solution
        metadata.update(
            {
                "ordinary_notation": notation,
                "ordinary_qtm_prefix_length": len(prefix),
                "residual_search_depths": trace.diagnostic_report()["depths"],
            }
        )
    elif prepared.spec.kind == "gflownet":
        run_config = _beam_config(
            config,
            prepared.device,
            beam_width=width,
            max_depth=6 if smoke else int(config["max_depth"]),
            cumulative_alpha=1.0,
            prune_immediate_inverse=True,
        )
        trace = beam_search(puzzle, start, prepared.scorer, run_config)
        path = trace.solution
        metadata["depths"] = trace.diagnostic_report()["depths"]
    else:  # pragma: no cover - registry exhaustiveness guard
        raise NotImplementedError(prepared.spec.kind)

    if path is not None and not puzzle.verify_solution(start, path):
        return MethodSolution(None, "invalid", {**metadata, "invalid_candidate_length": len(path)})
    return MethodSolution(None if path is None else tuple(map(int, path)), status, metadata)

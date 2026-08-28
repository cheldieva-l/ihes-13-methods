from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BENCHMARK_IDS = tuple(range(100, 121))
DEFAULT_BEAM_WIDTH = 365_000


@dataclass(frozen=True)
class MethodSpec:
    method_id: int
    slug: str
    title: str
    kind: str
    summary: str
    requires_training: bool = False
    requires_symmetry: bool = False
    optional_dependencies: tuple[str, ...] = ()

    @property
    def directory_name(self) -> str:
        return f"{self.method_id:02d}_{self.slug}"

    def config_path(self, repository_root: str | Path) -> Path:
        return Path(repository_root) / "methods" / self.directory_name / "config.json"


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(1, "parameterized_transforms", "Parameterized transforms", "transforms", "Symmetry, reverse, mirror, complement, axis, sign, and double-move transforms.", requires_symmetry=True),
    MethodSpec(2, "cumulative_scoring", "Cumulative scoring", "cumulative", "Accumulate heuristic scores over complete beam paths."),
    MethodSpec(3, "two_step_300_head", "Two-step ~300-head model", "two_step_head", "Distill 306 non-backtracking two-move scores into one parent evaluation.", requires_training=True),
    MethodSpec(4, "two_move_retraining", "Two-move retraining", "two_move_retrain", "Retrain on composite two-move walks and search in composite layers.", requires_training=True),
    MethodSpec(5, "neighbor_score_student", "18-neighbor student", "neighbor_head", "Distill all 18 neighbor scores from the scalar teacher.", requires_training=True),
    MethodSpec(6, "bidirectional_wide_join", "Bidirectional wide join", "bidirectional", "Join complete direct and reverse frontiers by exact mapped equality."),
    MethodSpec(7, "richer_randomized_input", "Richer randomized input", "rich_input", "Use current state, neighbors, and deterministic randomized encodings.", requires_training=True),
    MethodSpec(8, "radius8_intersection", "Radius-8 wait/intersect", "radius_intersection", "Intersect the forward beam with an exact backward ball of radius 8."),
    MethodSpec(9, "radius7_intersection", "Radius-7 wait/intersect", "radius_intersection", "Intersect the forward beam with an exact backward ball of radius 7."),
    MethodSpec(10, "no_immediate_inverse", "Immediate-inverse pruning", "inverse_prune", "Remove only the exact inverse of the previous move."),
    MethodSpec(11, "dual_center_trajectories", "Solved + scrambled centers", "dual_center", "Train on prefixes and suffixes of replay-valid known trajectories.", requires_training=True),
    MethodSpec(12, "ordinary_333_projection", "Ordinary 3x3 projection", "projection333", "Solve the ordinary 3x3 projection, replay its IHES prefix, then finish centers.", optional_dependencies=("kociemba",)),
    MethodSpec(13, "gflownet_flow", "GFlowNet / Flow", "gflownet", "Adapt flow-regularized trajectory balance and cumulative policy beam scoring.", requires_training=True),
)


def get_method(method_id: int) -> MethodSpec:
    matches = [item for item in METHODS if item.method_id == int(method_id)]
    if len(matches) != 1:
        raise KeyError(f"unknown method id {method_id}")
    return matches[0]


def validate_registry() -> None:
    if tuple(item.method_id for item in METHODS) != tuple(range(1, 14)):
        raise AssertionError("method ids must be exactly 1..13")
    if len({item.slug for item in METHODS}) != len(METHODS):
        raise AssertionError("method slugs must be unique")


validate_registry()


from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ihes_dual.puzzle import IHESPuzzle


class PermutationBackbone(nn.Module):
    """Compact trainable encoder used by method-specific student models."""

    def __init__(
        self,
        state_size: int = 72,
        class_count: int = 72,
        embed_dim: int = 16,
        hidden_size: int = 1024,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.state_size = int(state_size)
        self.class_count = int(class_count)
        self.embedding = nn.Embedding(class_count, embed_dim)
        self.input = nn.Linear(state_size * embed_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(residual_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.input(self.embedding(states.long()).flatten(1)))
        for block in self.blocks:
            hidden = F.relu(hidden + block(hidden))
        return self.output_norm(hidden)


class MultiActionStudent(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        state_size: int = 72,
        class_count: int = 72,
        embed_dim: int = 16,
        hidden_size: int = 1024,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = PermutationBackbone(
            state_size, class_count, embed_dim, hidden_size, residual_blocks
        )
        self.head = nn.Linear(hidden_size, output_dim)
        self.output_dim = int(output_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(states))


class ScalarStudent(nn.Module):
    def __init__(self, *, hidden_size: int = 1024, residual_blocks: int = 2) -> None:
        super().__init__()
        self.backbone = PermutationBackbone(hidden_size=hidden_size, residual_blocks=residual_blocks)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(states)).squeeze(-1)


class DualCenterStudent(nn.Module):
    """Two outputs: distance-to-solved and distance-to-current-scramble."""

    def __init__(self, *, hidden_size: int = 1024, residual_blocks: int = 2) -> None:
        super().__init__()
        self.backbone = PermutationBackbone(hidden_size=hidden_size, residual_blocks=residual_blocks)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(states))


class DualCenterGoalScorer:
    def __init__(self, model: DualCenterStudent) -> None:
        self.model = model

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)[:, 0]


class ActionScoreScorer:
    """Adapter consumed by the parent-level action path in ``beam_search``."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def score_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states).min(dim=1).values


def seeded_codebook(
    class_count: int = 72,
    dimension: int = 16,
    seed: int = 0,
) -> tuple[torch.Tensor, str]:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((class_count, dimension), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    digest = hashlib.sha256(matrix.tobytes()).hexdigest()
    return torch.from_numpy(matrix), digest


class RichNeighborStudent(nn.Module):
    """Scalar student with current-state and complete-neighborhood features."""

    def __init__(
        self,
        puzzle: IHESPuzzle,
        codebook: torch.Tensor,
        *,
        hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        if tuple(codebook.shape)[0] != puzzle.state_size:
            raise ValueError("codebook class count does not match the puzzle")
        self.register_buffer("moves", torch.from_numpy(puzzle.moves.astype(np.int64)))
        self.register_buffer("codebook", codebook.float())
        feature_width = puzzle.state_size * int(codebook.shape[1]) * 4
        self.network = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.long()
        current = self.codebook[states]
        # states[:, self.moves] -> batch x action x position
        neighbors = states[:, self.moves]
        neighbor_codes = self.codebook[neighbors]
        pooled_mean = neighbor_codes.mean(dim=1)
        pooled_min = neighbor_codes.amin(dim=1)
        pooled_max = neighbor_codes.amax(dim=1)
        features = torch.cat((current, pooled_mean, pooled_min, pooled_max), dim=-1)
        return self.network(features.flatten(1)).squeeze(-1)


class GFlowNetIHES(nn.Module):
    """Residual policy network adapted from the official MIT GFlowNet method."""

    def __init__(
        self,
        action_count: int = 18,
        *,
        hidden_size: int = 1024,
        residual_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.backbone = PermutationBackbone(
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
        self.forward_head = nn.Linear(hidden_size, action_count + 1)
        self.backward_head = nn.Linear(hidden_size, action_count)
        self.log_z = nn.Parameter(torch.zeros(()))
        self.action_count = int(action_count)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(states)
        return self.forward_head(hidden), self.backward_head(hidden)

    def score_actions(self, states: torch.Tensor) -> torch.Tensor:
        _, backward_logits = self(states)
        return -F.log_softmax(backward_logits, dim=-1)


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

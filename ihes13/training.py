from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ihes_dual.puzzle import IHESPuzzle

from .scorers import DualCenterStudent, GFlowNetIHES, model_fingerprint


@dataclass(frozen=True)
class TrainingReport:
    steps: int
    final_loss: float
    validation_loss: float | None
    checkpoint_sha256: str
    checkpoint_path: str | None


def random_walk_states(
    puzzle: IHESPuzzle,
    count: int,
    max_depth: int,
    *,
    seed: int,
    non_backtracking: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or max_depth <= 0:
        raise ValueError("count and max_depth must be positive")
    rng = np.random.default_rng(seed)
    depths = rng.integers(1, max_depth + 1, size=count, dtype=np.int16)
    states = np.repeat(puzzle.solved[None, :], count, axis=0).astype(np.uint8)
    last = np.full(count, -1, dtype=np.int16)
    for depth in range(1, max_depth + 1):
        active = np.flatnonzero(depths >= depth)
        if len(active) == 0:
            break
        moves = rng.integers(0, puzzle.generator_count, size=len(active), dtype=np.int16)
        if non_backtracking and depth > 1:
            forbidden = puzzle.inverse_move[last[active]]
            collision = moves == forbidden
            while np.any(collision):
                moves[collision] = rng.integers(
                    0, puzzle.generator_count, size=int(collision.sum()), dtype=np.int16
                )
                collision = moves == forbidden
        states[active] = puzzle.apply_many(states[active], moves).astype(np.uint8, copy=False)
        last[active] = moves
    return states, depths.astype(np.float32)


def _score_teacher(
    teacher: Callable[[torch.Tensor], torch.Tensor],
    states: np.ndarray,
    *,
    device: str,
    batch_size: int = 8192,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    device_type = torch.device(device).type
    for offset in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[offset : offset + batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=device_type == "cuda",
        ):
            score = teacher(batch)
        parts.append(score.detach().float().cpu().reshape(-1))
    return torch.cat(parts) if parts else torch.empty(0)


def _save_model(model: nn.Module, checkpoint_path: str | Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({"state_dict": model.state_dict()}, temporary)
    temporary.replace(path)
    return str(path)


def train_action_student(
    puzzle: IHESPuzzle,
    action_moves: np.ndarray,
    teacher: Callable[[torch.Tensor], torch.Tensor],
    student: nn.Module,
    *,
    steps: int,
    batch_size: int,
    max_walk_depth: int,
    learning_rate: float,
    seed: int,
    device: str,
    teacher_batch_size: int = 8192,
    checkpoint_path: str | Path | None = None,
) -> TrainingReport:
    actions = np.asarray(action_moves, dtype=np.uint8)
    if actions.ndim != 2 or actions.shape[1] != puzzle.state_size:
        raise ValueError("action_moves have the wrong shape")
    torch.manual_seed(seed)
    model = student.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    final_loss = float("nan")
    for step in range(steps):
        states, _ = random_walk_states(
            puzzle,
            batch_size,
            max_walk_depth,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        children = states[:, actions].reshape(-1, puzzle.state_size)
        targets = _score_teacher(
            teacher, children, device=device, batch_size=teacher_batch_size
        ).reshape(batch_size, len(actions)).to(device)
        inputs = torch.from_numpy(states).to(device)
        predictions = model(inputs)
        if predictions.shape != targets.shape:
            raise ValueError(f"student output {predictions.shape} != targets {targets.shape}")
        loss = F.smooth_l1_loss(predictions.float(), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    validation_states, _ = random_walk_states(
        puzzle, min(2048, max(128, batch_size)), max_walk_depth, seed=seed + 1
    )
    validation_children = validation_states[:, actions].reshape(-1, puzzle.state_size)
    validation_targets = _score_teacher(
        teacher, validation_children, device=device, batch_size=teacher_batch_size
    ).reshape(len(validation_states), len(actions))
    with torch.inference_mode():
        validation_predictions = model(torch.from_numpy(validation_states).to(device)).float().cpu()
    validation_loss = float(F.smooth_l1_loss(validation_predictions, validation_targets))
    saved = _save_model(model, checkpoint_path)
    return TrainingReport(
        steps=steps,
        final_loss=final_loss,
        validation_loss=validation_loss,
        checkpoint_sha256=model_fingerprint(model),
        checkpoint_path=saved,
    )


def train_scalar_student(
    puzzle: IHESPuzzle,
    teacher: Callable[[torch.Tensor], torch.Tensor],
    student: nn.Module,
    *,
    steps: int,
    batch_size: int,
    max_walk_depth: int,
    learning_rate: float,
    seed: int,
    device: str,
    checkpoint_path: str | Path | None = None,
) -> TrainingReport:
    torch.manual_seed(seed)
    model = student.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    final_loss = float("nan")
    for _ in range(steps):
        states, depths = random_walk_states(
            puzzle,
            batch_size,
            max_walk_depth,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        teacher_score = _score_teacher(teacher, states, device=device).to(device)
        depth_tensor = torch.from_numpy(depths).to(device)
        # A small exact-depth anchor preserves local monotonicity while the
        # teacher remains the dominant target.
        target = teacher_score + 0.05 * depth_tensor
        prediction = model(torch.from_numpy(states).to(device)).float()
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    validation_states, validation_depths = random_walk_states(
        puzzle, min(2048, max(128, batch_size)), max_walk_depth, seed=seed + 1
    )
    validation_target = _score_teacher(teacher, validation_states, device=device) + 0.05 * torch.from_numpy(validation_depths)
    with torch.inference_mode():
        validation_prediction = model(torch.from_numpy(validation_states).to(device)).float().cpu()
    validation_loss = float(F.smooth_l1_loss(validation_prediction, validation_target))
    saved = _save_model(model, checkpoint_path)
    return TrainingReport(steps, final_loss, validation_loss, model_fingerprint(model), saved)


def trajectory_states(
    puzzle: IHESPuzzle,
    start: Sequence[int] | np.ndarray,
    path: Sequence[int],
) -> np.ndarray:
    states = [np.asarray(start, dtype=np.uint8).copy()]
    for move in path:
        states.append(puzzle.apply(states[-1], int(move)).astype(np.uint8, copy=False))
    result = np.stack(states)
    if not np.array_equal(result[-1], puzzle.solved):
        raise ValueError("training trajectory is not replay-valid")
    return result


def train_dual_center_student(
    puzzle: IHESPuzzle,
    start: Sequence[int] | np.ndarray,
    path: Sequence[int],
    model: DualCenterStudent,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str,
    checkpoint_path: str | Path | None = None,
) -> TrainingReport:
    states = trajectory_states(puzzle, start, path)
    length = len(path)
    labels = np.stack(
        (
            np.arange(length, -1, -1, dtype=np.float32),
            np.arange(0, length + 1, dtype=np.float32),
        ),
        axis=1,
    )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    final_loss = float("nan")
    for _ in range(steps):
        indices = rng.integers(0, len(states), size=max(32, len(states) * 2))
        inputs = torch.from_numpy(states[indices]).to(device)
        targets = torch.from_numpy(labels[indices]).to(device)
        prediction = model(inputs).float()
        loss = F.smooth_l1_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(states).to(device)).float().cpu()
    validation_loss = float(F.smooth_l1_loss(prediction, torch.from_numpy(labels)))
    saved = _save_model(model, checkpoint_path)
    return TrainingReport(steps, final_loss, validation_loss, model_fingerprint(model), saved)


def train_gflownet(
    puzzle: IHESPuzzle,
    model: GFlowNetIHES,
    *,
    steps: int,
    batch_size: int,
    trajectory_length: int,
    learning_rate: float,
    weight_decay: float,
    regularization_coefficient: float,
    gradient_clip: float,
    seed: int,
    device: str,
    checkpoint_path: str | Path | None = None,
) -> TrainingReport:
    """Train a compact IHES adaptation of regularized trajectory balance.

    This follows the official implementation's prefix-wise objective: forward
    and backward log-probabilities are accumulated at every trajectory prefix,
    log-flow is the negative log stop probability, and the regularizer sums
    exponentiated flows.  The IHES state-space cardinality is not known here,
    so ``log_z`` is learned instead of fixed to the exact Rubik group size.
    """

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    moves_tensor = torch.from_numpy(puzzle.moves.astype(np.int64)).to(device)
    inverse_tensor = torch.from_numpy(puzzle.inverse_move.astype(np.int64)).to(device)
    final_loss = float("nan")
    for _ in range(steps):
        current = torch.from_numpy(
            np.repeat(puzzle.solved[None, :], batch_size, axis=0).astype(np.uint8)
        ).to(device)
        last = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        sampled_states = [current]
        sampled_actions: list[torch.Tensor] = []
        with torch.no_grad():
            for depth in range(trajectory_length):
                forward_logits, _ = model(current)
                action_logits = forward_logits[:, : puzzle.generator_count]
                if depth > 0:
                    forbidden = inverse_tensor[last]
                    action_logits = action_logits.scatter(
                        1, forbidden[:, None], torch.finfo(action_logits.dtype).min
                    )
                action = torch.distributions.Categorical(logits=action_logits).sample()
                current = torch.gather(current, 1, moves_tensor[action])
                sampled_actions.append(action)
                sampled_states.append(current)
                last = action

        state_tensor = torch.stack(sampled_states, dim=0)
        action_tensor = torch.stack(sampled_actions, dim=0)
        flat_states = state_tensor.reshape(-1, puzzle.state_size)
        forward_logits, backward_logits = model(flat_states)
        forward_log_probs = F.log_softmax(forward_logits, dim=-1).reshape(
            trajectory_length + 1, batch_size, puzzle.generator_count + 1
        )
        backward_log_probs = F.log_softmax(backward_logits, dim=-1).reshape(
            trajectory_length + 1, batch_size, puzzle.generator_count
        )
        selected_forward = torch.gather(
            forward_log_probs[:-1, :, : puzzle.generator_count],
            2,
            action_tensor[:, :, None],
        ).squeeze(-1)
        inverse_actions = inverse_tensor[action_tensor]
        selected_backward = torch.gather(
            backward_log_probs[1:], 2, inverse_actions[:, :, None]
        ).squeeze(-1)
        zero = torch.zeros((1, batch_size), device=device)
        forward_prefix = torch.cat(
            (zero, torch.cumsum(selected_forward, dim=0)), dim=0
        )
        backward_prefix = torch.cat(
            (zero, torch.cumsum(selected_backward, dim=0)), dim=0
        )
        log_flows = -forward_log_probs[:, :, -1]
        balance = model.log_z + forward_prefix - backward_prefix - log_flows
        tb_loss = balance.square().mean()
        regularizer = regularization_coefficient * torch.exp(
            torch.logsumexp(log_flows[1:], dim=0)
        ).mean()
        loss = tb_loss + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    saved = _save_model(model, checkpoint_path)
    return TrainingReport(steps, final_loss, None, model_fingerprint(model), saved)

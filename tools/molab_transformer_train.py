"""Run one restartable IHES PieceTransformer training job on a Molab GPU.

Set IHES_RUN_ID, IHES_SEED and IHES_K_MAX before starting this file.  Checkpoints
are written under /marimo/ihes_runs and reused automatically after a 12-hour
Molab session replacement.  Every function below has one small responsibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


TRAINING_REPOSITORY = "https://github.com/AnanasClassic/cayleypy-training-core.git"
TRAINING_COMMIT = "5a72174"
CONFIG_NAME = "ihes_p901_t000_piece_transformer.json"


def read_settings() -> dict:
    """Read environment settings; return a complete reproducible run config."""
    return {
        "run_id": os.environ.get("IHES_RUN_ID", "E010_T1_SEED42_K23"),
        "seed": int(os.environ.get("IHES_SEED", "42")),
        "k_max": int(os.environ.get("IHES_K_MAX", "23")),
        "epochs": int(os.environ.get("IHES_EPOCHS", "65536")),
    }


def prepare_repository(repository_dir: Path) -> dict:
    """Clone and pin the public training core; return command status details."""
    if (repository_dir / ".git").is_dir():
        commands = [
            ["git", "-C", str(repository_dir), "fetch", "--depth", "1", "origin", TRAINING_COMMIT],
            ["git", "-C", str(repository_dir), "checkout", "--detach", "FETCH_HEAD"],
        ]
    elif repository_dir.exists():
        return {"ok": False, "error": f"unexpected path exists: {repository_dir}"}
    else:
        commands = [
            ["git", "clone", "--filter=blob:none", TRAINING_REPOSITORY, str(repository_dir)],
            ["git", "-C", str(repository_dir), "checkout", "--detach", TRAINING_COMMIT],
        ]

    tails = []
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True)
        tails.append((result.stdout + result.stderr)[-500:])
        if result.returncode != 0:
            return {"ok": False, "return_code": result.returncode, "tail": tails[-1]}
    return {"ok": True, "tail": tails[-1]}


def install_training_core(repository_dir: Path) -> dict:
    """Install the pinned core in the current Molab kernel; return status details."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(repository_dir)],
        text=True,
        capture_output=True,
    )
    return {
        "ok": result.returncode == 0,
        "return_code": result.returncode,
        "tail": (result.stdout + result.stderr)[-500:],
    }


def build_training_command(repository_dir: Path, settings: dict, output_dir: Path) -> list[str]:
    """Build the unified-training command; return argv without shell quoting."""
    config = repository_dir / "configs" / CONFIG_NAME
    latest = output_dir / f"p901-t000-q-rw_{settings['run_id']}_latest.pt"
    command = [
        sys.executable,
        "-m",
        "unified_training",
        str(config),
        "--set", f"artifacts.run_id={json.dumps(settings['run_id'])}",
        "--set", f"artifacts.output_dir={json.dumps(str(output_dir))}",
        "--set", "training.device=\"cuda:0\"",
        "--set", "training.amp=\"bf16\"",
        "--set", f"training.seed={settings['seed']}",
        "--set", f"sampler.k_max={settings['k_max']}",
        "--set", f"training.epochs={settings['epochs']}",
        "--set", "training.val_size=8192",
        "--set", "training.val_batch_size=2048",
        "--set", "training.val_every=10",
        "--set", "training.save_every=1",
        "--set", "training.log_every=10",
    ]
    if latest.exists():
        command.extend(["--set", f"training.resume={json.dumps(str(latest))}"])
    return command


def write_status(path: Path, payload: dict) -> None:
    """Atomically persist a small status record; input payload, no return value."""
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stream_training(command: list[str], status_path: Path, settings: dict) -> dict:
    """Run training and mirror concise progress; return the final process record."""
    started = time.time()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    tail: list[str] = []
    status = {**settings, "state": "running", "started_unix": started, "pid": process.pid}
    write_status(status_path, status)
    print("IHES_TRAIN_START", json.dumps(status, sort_keys=True), flush=True)
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail = (tail + [line])[-20:]
            if '"event": "epoch"' in line:
                status.update(last_event=line.strip(), updated_unix=time.time())
                write_status(status_path, status)
    return_code = process.wait()
    status.update(
        state="complete" if return_code == 0 else "failed",
        return_code=return_code,
        elapsed_seconds=round(time.time() - started, 3),
        output_tail="".join(tail),
        updated_unix=time.time(),
    )
    write_status(status_path, status)
    print("IHES_TRAIN_FINISH", json.dumps(status, sort_keys=True), flush=True)
    return status


def run() -> dict:
    """Prepare, resume and run one job; return a fail-soft machine-readable receipt."""
    settings = read_settings()
    repository_dir = Path("/tmp/cayleypy-training-core")
    output_dir = Path("/marimo/ihes_runs") / settings["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "molab_status.json"
    try:
        prepared = prepare_repository(repository_dir)
        if not prepared.get("ok"):
            receipt = {**settings, "state": "prepare_failed", "detail": prepared}
        else:
            installed = install_training_core(repository_dir)
            if not installed.get("ok"):
                receipt = {**settings, "state": "install_failed", "detail": installed}
            else:
                command = build_training_command(repository_dir, settings, output_dir)
                receipt = stream_training(command, status_path, settings)
    except Exception as error:
        receipt = {
            **settings,
            "state": "launcher_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_status(status_path, receipt)
    if receipt.get("state") not in {"running", "complete", "failed"}:
        print("IHES_TRAIN_ERROR", json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


if __name__ == "__main__":
    run()

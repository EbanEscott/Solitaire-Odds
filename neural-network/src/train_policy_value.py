"""
Train a policy-value network on Solitaire logs from the Java engine.

The trainer supports resumable training by writing periodic checkpoints that include
the model state, optimizer state, RNG state, and structured epoch metrics.

Run from the neural-network project root as:

    # Single file, default architecture (256 hidden, 2 layers)
    python -m src.train_policy_value ../engine/logs/episode.log

    # Larger model (512 hidden, 3 layers)
    python -m src.train_policy_value --hidden-dim 512 --num-layers 3 logs/episode.log

    # Resume from a checkpoint written by an earlier run
    python -m src.train_policy_value \
        --resume-from checkpoints/policy_value_latest.pt \
        --epochs 10 \
        ../engine/logs/episode.log

Variable naming convention:
- Variables prefixed with `validation_` represent metrics computed on the validation dataset.
- Variables prefixed with `value_` or containing `value` represent the value head outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import SolitaireStateDataset, TrajectoryConfig
from .model import PolicyValueNet


TRAIN_INTERRUPTED_EXIT_CODE = 99
DATASET_SPLIT_SEED = 42
DEFAULT_CHECKPOINT_PREFIX = "policy_value"


def _resolve_log_paths(argv: List[str]) -> List[Path]:
    """Resolve command-line log arguments into concrete existing log file paths."""

    resolved: List[Path] = []

    for arg in argv:
        path = Path(arg)
        if "*" in arg or "?" in arg:
            parent = path.parent
            pattern = path.name
            matches = sorted(parent.glob(pattern))
            if not matches:
                print(f"Warning: glob pattern '{arg}' matched no files")
            resolved.extend(matches)
        elif path.exists():
            resolved.append(path)
        else:
            print(f"Error: file or pattern not found: {arg}")
            raise SystemExit(1)

    return resolved


def _get_git_commit(repo_path: Path) -> str:
    """Return the current git commit hash for a repository, or 'unknown' if unavailable."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"

    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def _capture_rng_state() -> Dict[str, Any]:
    """Capture the RNG state needed to resume shuffled training deterministically."""

    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: Dict[str, Any] | None) -> None:
    """Restore previously captured RNG state when resuming a checkpoint."""

    if not rng_state:
        return
    python_random_state = rng_state.get("python_random")
    if python_random_state is not None:
        random.setstate(python_random_state)

    torch_random_state = rng_state.get("torch")
    if torch_random_state is not None:
        torch.set_rng_state(torch_random_state)

    cuda_random_state = rng_state.get("cuda")
    if cuda_random_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_random_state)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move any tensor-valued optimizer state onto the active training device."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _append_epoch_metrics(metrics_output_path: Path | None, metrics: Dict[str, Any]) -> None:
    """Append one structured epoch metrics record to a JSONL file if configured."""

    if metrics_output_path is None:
        return

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output_path.open("a", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)
        handle.write("\n")


def _build_checkpoint_metadata(
    *,
    args: argparse.Namespace,
    log_paths: List[Path],
    train_size: int,
    validation_size: int,
    total_training_duration_seconds: float,
    latest_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the metadata block written into every checkpoint payload."""

    repo_root = Path(__file__).parent.parent.parent
    engine_repo = repo_root / "engine"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_neural": _get_git_commit(repo_root),
        "git_commit_engine": _get_git_commit(engine_repo),
        "training_samples": train_size,
        "validation_samples": validation_size,
        "architecture": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "batch_norm": args.batch_norm,
            "residual": args.residual,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "checkpoint_every_epochs": args.save_every_epochs,
        },
        "final_metrics": latest_metrics,
        "training_duration_seconds": total_training_duration_seconds,
        "data_sources": [str(path) for path in log_paths],
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "resume_from_checkpoint": str(args.resume_from) if args.resume_from is not None else None,
        "metrics_output": str(args.metrics_output) if args.metrics_output is not None else None,
        "dataset_split_seed": DATASET_SPLIT_SEED,
    }


def _save_checkpoint(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    dataset: SolitaireStateDataset,
    state_dim: int,
    num_actions: int,
    log_paths: List[Path],
    train_size: int,
    validation_size: int,
    current_epoch: int,
    global_step: int,
    total_training_duration_seconds: float,
    latest_metrics: Dict[str, Any],
    checkpoint_reason: str,
) -> Path:
    """Write an epoch-specific checkpoint and refresh the latest checkpoint alias."""

    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / f"{args.checkpoint_prefix}_latest.pt"
    epoch_path = out_dir / f"{args.checkpoint_prefix}_epoch_{current_epoch:04d}.pt"

    payload = {
        "feature_dim": state_dim,
        "action_space_size": num_actions,
        "action_index_map": dataset.action_space.index_to_action,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": _build_checkpoint_metadata(
            args=args,
            log_paths=log_paths,
            train_size=train_size,
            validation_size=validation_size,
            total_training_duration_seconds=total_training_duration_seconds,
            latest_metrics=latest_metrics,
        ),
        "training_state": {
            "current_epoch": current_epoch,
            "target_epochs": args.epochs,
            "global_step": global_step,
            "accumulated_training_duration_seconds": total_training_duration_seconds,
            "checkpoint_reason": checkpoint_reason,
            "checkpoint_prefix": args.checkpoint_prefix,
            "latest_metrics": latest_metrics,
        },
        "rng_state": _capture_rng_state(),
    }

    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return latest_path


def main(argv: List[str] | None = None) -> None:
    """Train the policy-value network, optionally resuming from a prior checkpoint."""

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="python -m src.train_policy_value",
        description="Train a configurable, resumable policy-value network on Solitaire episodes.",
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Log files or glob patterns (for example 'logs/episode*.log')",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension (default: 256). Increase to 512-2048 for larger models.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2). Increase to 3-4 for deeper networks.",
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Use batch normalization (experimental)",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use residual connections (experimental, requires num-layers > 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Total number of training epochs to reach (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where periodic checkpoints are written (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default=DEFAULT_CHECKPOINT_PREFIX,
        help="Prefix for checkpoint filenames (default: policy_value)",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=1,
        help="Save a resumable checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume training from a checkpoint written by a previous run",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Write one JSONL metrics record per completed epoch to this path",
    )
    parser.add_argument(
        "--simulate-interrupt-after-epoch",
        type=int,
        help="Exit after saving the checkpoint for this epoch to validate resume behavior",
    )

    args = parser.parse_args(argv)

    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.save_every_epochs < 1:
        raise SystemExit("--save-every-epochs must be >= 1")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be > 0")
    if args.simulate_interrupt_after_epoch is not None and args.simulate_interrupt_after_epoch < 1:
        raise SystemExit("--simulate-interrupt-after-epoch must be >= 1")

    log_paths = _resolve_log_paths(args.log_files)
    if not log_paths:
        print("Error: no valid log files found")
        raise SystemExit(1)

    trajectory_config = TrajectoryConfig(
        use_trajectory_value=True,
        use_bootstrapped_value=False,
    )

    dataset = SolitaireStateDataset(log_paths, trajectory_config=trajectory_config)
    if len(dataset) == 0:
        print("Dataset is empty; ensure the Java engine was run with -Dlog.episodes=true.")
        raise SystemExit(1)

    validation_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(DATASET_SPLIT_SEED)
    train_ds, validation_ds = random_split(dataset, [train_size, validation_size], generator=generator)

    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    num_actions = sample["policy"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyValueNet(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_batch_norm=args.batch_norm,
        use_residual=args.residual,
    )
    model.to(device)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(
        f"Training on {len(train_ds)} samples, validating on {len(validation_ds)} samples "
        f"(state_dim={state_dim}, num_actions={num_actions}, device={device})"
    )
    print(
        f"Model Architecture: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
        f"batch_norm={args.batch_norm}, residual={args.residual}"
    )
    print(f"Model Size: {total_params:,} total parameters, {trainable_params:,} trainable")
    print(f"Estimated checkpoint size: {(total_params * 4) / (1024 * 1024):.2f} MB")
    print(
        f"Training target: {args.epochs} epochs, batch_size={args.batch_size}, "
        f"lr={args.learning_rate}, save_every_epochs={args.save_every_epochs}"
    )

    completed_epochs = 0
    global_step = 0
    accumulated_training_duration_seconds = 0.0

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            _move_optimizer_state_to_device(optimizer, device)

        training_state = checkpoint.get("training_state", {})
        completed_epochs = int(training_state.get("current_epoch", checkpoint.get("epoch", 0) or 0))
        global_step = int(training_state.get("global_step", 0))
        accumulated_training_duration_seconds = float(
            training_state.get("accumulated_training_duration_seconds", 0.0)
        )
        _restore_rng_state(checkpoint.get("rng_state"))

        print(
            f"Resuming from {args.resume_from} at epoch {completed_epochs} "
            f"(global_step={global_step})"
        )

    start_epoch = completed_epochs + 1
    if start_epoch > args.epochs:
        print(
            f"Checkpoint already completed {completed_epochs} epochs, which satisfies "
            f"the requested target of {args.epochs}. Nothing to do."
        )
        return

    weight_value = 0.3
    attempt_start_time = time.time()
    latest_checkpoint_path = args.checkpoint_dir / f"{args.checkpoint_prefix}_latest.pt"
    last_epoch_metrics: Dict[str, Any] | None = None

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct_policy = 0
        total_correct_value = 0
        total_examples = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                print(f"  Epoch {epoch}/{args.epochs} - Batch {batch_idx:04d}/{len(train_loader):04d}")

            states = batch["state"].to(device)
            target_actions = batch["policy"].argmax(dim=-1).to(device)
            target_values = batch["value"].to(device)

            outputs = model(states)
            policy_logits = outputs["policy"]
            value_logits = outputs["value"].squeeze(-1)

            policy_loss = policy_loss_fn(policy_logits, target_actions)
            value_loss = value_loss_fn(value_logits, target_values)
            loss = policy_loss + weight_value * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            with torch.no_grad():
                predicted_actions = policy_logits.argmax(dim=-1)
                policy_correct = (predicted_actions == target_actions).sum().item()

                value_probabilities = torch.sigmoid(value_logits)
                predicted_values = (value_probabilities >= 0.5).float()
                value_correct = (predicted_values == target_values).sum().item()

            batch_size = states.size(0)
            total_examples += batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_correct_policy += policy_correct
            total_correct_value += value_correct

        avg_policy_loss = total_policy_loss / total_examples
        avg_value_loss = total_value_loss / total_examples
        train_policy_accuracy = total_correct_policy / total_examples
        train_value_accuracy = total_correct_value / total_examples

        model.eval()
        validation_policy_loss = 0.0
        validation_value_loss = 0.0
        validation_correct_policy = 0
        validation_correct_value = 0
        validation_examples = 0

        with torch.no_grad():
            for batch in validation_loader:
                states = batch["state"].to(device)
                target_actions = batch["policy"].argmax(dim=-1).to(device)
                target_values = batch["value"].to(device)

                outputs = model(states)
                policy_logits = outputs["policy"]
                value_logits = outputs["value"].squeeze(-1)

                policy_loss = policy_loss_fn(policy_logits, target_actions)
                value_loss = value_loss_fn(value_logits, target_values)

                predicted_actions = policy_logits.argmax(dim=-1)
                policy_correct = (predicted_actions == target_actions).sum().item()

                value_probabilities = torch.sigmoid(value_logits)
                predicted_values = (value_probabilities >= 0.5).float()
                value_correct = (predicted_values == target_values).sum().item()

                batch_size = states.size(0)
                validation_examples += batch_size
                validation_policy_loss += policy_loss.item() * batch_size
                validation_value_loss += value_loss.item() * batch_size
                validation_correct_policy += policy_correct
                validation_correct_value += value_correct

        avg_validation_policy_loss = validation_policy_loss / validation_examples
        avg_validation_value_loss = validation_value_loss / validation_examples
        validation_policy_accuracy = validation_correct_policy / validation_examples
        validation_value_accuracy = validation_correct_value / validation_examples
        epoch_duration_seconds = time.time() - epoch_start_time

        last_epoch_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "target_epochs": args.epochs,
            "global_step": global_step,
            "epoch_duration_seconds": epoch_duration_seconds,
            "training_policy_loss": float(avg_policy_loss),
            "training_value_loss": float(avg_value_loss),
            "training_policy_accuracy": float(train_policy_accuracy),
            "training_value_accuracy": float(train_value_accuracy),
            "validation_policy_loss": float(avg_validation_policy_loss),
            "validation_value_loss": float(avg_validation_value_loss),
            "validation_policy_accuracy": float(validation_policy_accuracy),
            "validation_value_accuracy": float(validation_value_accuracy),
        }
        _append_epoch_metrics(args.metrics_output, last_epoch_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss(p={avg_policy_loss:.3f}, v={avg_value_loss:.3f}), "
            f"train_accuracy(p={train_policy_accuracy:.3f}, v={train_value_accuracy:.3f}) "
            f"- validation_loss(p={avg_validation_policy_loss:.3f}, v={avg_validation_value_loss:.3f}), "
            f"validation_accuracy(p={validation_policy_accuracy:.3f}, v={validation_value_accuracy:.3f})"
        )

        should_save_checkpoint = (
            epoch == args.epochs
            or epoch % args.save_every_epochs == 0
            or args.simulate_interrupt_after_epoch == epoch
        )
        if should_save_checkpoint:
            total_training_duration_seconds = (
                accumulated_training_duration_seconds + (time.time() - attempt_start_time)
            )
            checkpoint_reason = "final" if epoch == args.epochs else "epoch"
            latest_checkpoint_path = _save_checkpoint(
                args=args,
                out_dir=args.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                dataset=dataset,
                state_dim=state_dim,
                num_actions=num_actions,
                log_paths=log_paths,
                train_size=len(train_ds),
                validation_size=len(validation_ds),
                current_epoch=epoch,
                global_step=global_step,
                total_training_duration_seconds=total_training_duration_seconds,
                latest_metrics=last_epoch_metrics,
                checkpoint_reason=checkpoint_reason,
            )
            print(f"Saved checkpoint to {latest_checkpoint_path}")

        if args.simulate_interrupt_after_epoch == epoch:
            print(
                f"Simulating interrupt after epoch {epoch}. "
                f"Resume from {latest_checkpoint_path} to continue training."
            )
            raise SystemExit(TRAIN_INTERRUPTED_EXIT_CODE)

    if last_epoch_metrics is None:
        raise SystemExit("Training finished without producing any epoch metrics")

    print(f"Saved model checkpoint to {latest_checkpoint_path}")


if __name__ == "__main__":
    main()
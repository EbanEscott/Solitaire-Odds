"""
Train a policy–value network on Solitaire logs from the Java engine with configurable architecture.

Supports both supervised learning (bootstrap from A* or other AI) and self-play RL loops.

Run from the project root as:

    # Single file, default architecture (256 hidden, 2 layers)
    python -m src.train_policy_value /Users/ebo/Code/solitaire/engine/logs/episode.log
    
    # Larger model (512 hidden, 3 layers)
    python -m src.train_policy_value --hidden-dim 512 --num-layers 3 logs/episode.log
    
    # Extra large (1024 hidden, 4 layers) for full game tree training
    python -m src.train_policy_value --hidden-dim 1024 --num-layers 4 logs/episode.log
    
    # With batch norm and residual connections (experimental)
    python -m src.train_policy_value --hidden-dim 512 --num-layers 3 --batch-norm --residual logs/episode.log
    
    # Multiple files
    python -m src.train_policy_value logs/episode.1.log logs/episode.2.log logs/episode.3.log
    
    # Glob pattern (quote to prevent shell expansion)
    python -m src.train_policy_value "logs/episode*.log"

Variable naming convention:
- Variables prefixed with `validation_` represent metrics computed on the validation dataset
  (e.g., validation_policy_loss, validation_policy_accuracy)
- Variables prefixed with `value_` or containing `value` represent the value head predictions
  (e.g., value_logits, avg_value_loss)
This distinction is critical: `validation_` = validation data split, `value_` = value head output.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import SolitaireStateDataset, TrajectoryConfig
from .model import PolicyValueNet


def _resolve_log_paths(argv: List[str]) -> List[Path]:
    """
    Resolve arguments to a list of log file paths.
    
    Handles:
    - Explicit file paths: /path/to/file.log
    - Glob patterns: /path/to/episode*.log
    """
    resolved = []
    
    for arg in argv:
        p = Path(arg)
        
        # If it's a glob pattern (contains * or ?), expand it
        if '*' in arg or '?' in arg:
            parent = p.parent
            pattern = p.name
            matches = sorted(parent.glob(pattern))
            if not matches:
                print(f"Warning: glob pattern '{arg}' matched no files")
            resolved.extend(matches)
        elif p.exists():
            resolved.append(p)
        else:
            print(f"Error: file or pattern not found: {arg}")
            raise SystemExit(1)
    
    return resolved


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Parse command-line arguments for network configuration
    parser = argparse.ArgumentParser(
        prog="python -m src.train_policy_value",
        description="Train a configurable policy-value network on Solitaire episodes."
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        help="Log files or glob patterns (e.g., 'logs/episode*.log')"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension (default: 256). Increase to 512-2048 for larger models."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2). Increase to 3-4 for deeper networks."
    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="Use batch normalization (experimental)"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Use residual connections (experimental, requires num-layers > 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)"
    )
    
    args = parser.parse_args(argv)
    
    # Resolve log file paths
    log_paths = _resolve_log_paths(args.log_files)
    
    if not log_paths:
        print("Error: no valid log files found")
        raise SystemExit(1)

    # Create trajectory config (for future self-play bootstrapping)
    trajectory_config = TrajectoryConfig(
        use_trajectory_value=True,  # Use full game outcome
        use_bootstrapped_value=False,  # Self-play will enable this
    )

    dataset = SolitaireStateDataset(log_paths, trajectory_config=trajectory_config)
    if len(dataset) == 0:
        print("Dataset is empty; ensure the Java engine was run with -Dlog.episodes=true.")
        raise SystemExit(1)

    validation_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(42)
    train_ds, validation_ds = random_split(dataset, [train_size, validation_size], generator=generator)

    sample = dataset[0]
    state_dim = sample['state'].shape[0]
    num_actions = sample['policy'].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with configurable architecture
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

    num_epochs = args.epochs
    
    # Loss weight for value head
    weight_value = 0.3

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    print(f"Training: {num_epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct_policy = 0
        total_correct_value = 0
        total_examples = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                print(f"  Epoch {epoch}/{num_epochs} - Batch {batch_idx:04d}/{len(train_loader):04d}")
            
            # Extract tensors from batch dict
            states = batch['state'].to(device)
            target_actions = batch['policy'].argmax(dim=-1).to(device)
            target_values = batch['value'].to(device)

            # Forward pass
            outputs = model(states)
            policy_logits = outputs['policy']
            value_logits = outputs['value'].squeeze(-1)

            # Compute losses
            p_loss = policy_loss_fn(policy_logits, target_actions)
            v_loss = value_loss_fn(value_logits, target_values)
            
            # Combined loss with weight on value head
            loss = p_loss + weight_value * v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracies
            with torch.no_grad():
                pred_actions = policy_logits.argmax(dim=-1)
                policy_correct = (pred_actions == target_actions).sum().item()

                value_probs = torch.sigmoid(value_logits)
                value_pred = (value_probs >= 0.5).float()
                value_correct = (value_pred == target_values).sum().item()

            batch_size = states.size(0)
            total_examples += batch_size
            total_policy_loss += p_loss.item() * batch_size
            total_value_loss += v_loss.item() * batch_size
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
                states = batch['state'].to(device)
                target_actions = batch['policy'].argmax(dim=-1).to(device)
                target_values = batch['value'].to(device)

                outputs = model(states)
                policy_logits = outputs['policy']
                value_logits = outputs['value'].squeeze(-1)

                p_loss = policy_loss_fn(policy_logits, target_actions)
                v_loss = value_loss_fn(value_logits, target_values)

                pred_actions = policy_logits.argmax(dim=-1)
                policy_correct = (pred_actions == target_actions).sum().item()

                value_probs = torch.sigmoid(value_logits)
                value_pred = (value_probs >= 0.5).float()
                value_correct = (value_pred == target_values).sum().item()

                batch_size = states.size(0)
                validation_examples += batch_size
                validation_policy_loss += p_loss.item() * batch_size
                validation_value_loss += v_loss.item() * batch_size
                validation_correct_policy += policy_correct
                validation_correct_value += value_correct

        avg_validation_policy_loss = validation_policy_loss / validation_examples
        avg_validation_value_loss = validation_value_loss / validation_examples
        validation_policy_accuracy = validation_correct_policy / validation_examples
        validation_value_accuracy = validation_correct_value / validation_examples

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss(p={avg_policy_loss:.3f}, v={avg_value_loss:.3f}), "
            f"train_accuracy(p={train_policy_accuracy:.3f}, v={train_value_accuracy:.3f}) "
            f"- validation_loss(p={avg_validation_policy_loss:.3f}, v={avg_validation_value_loss:.3f}), "
            f"validation_accuracy(p={validation_policy_accuracy:.3f}, v={validation_value_accuracy:.3f})"
        )

    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "policy_value_latest.pt"
    torch.save(
        {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "index_to_action": dataset.action_space.index_to_action,
            "model_state_dict": model.state_dict(),
        },
        out_path,
    )
    print(f"Saved model checkpoint to {out_path}")


if __name__ == "__main__":
    main()


"""
Train a policy–value network on Solitaire logs from the Java engine.

Run from the project root as:

    # Single file
    python -m src.train_policy_value /Users/ebo/Code/solitaire/engine/logs/episode.log
    
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
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import SolitaireStateDataset
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

    if not argv:
        print(
            "Usage: python -m src.train_policy_value "
            "/path/to/episode.log [more_logs.log ...]\n"
            "\n"
            "Supports glob patterns:\n"
            "  python -m src.train_policy_value 'logs/episode*.log'"
        )
        raise SystemExit(1)

    log_paths = _resolve_log_paths(argv)
    
    if not log_paths:
        print("Error: no valid log files found")
        raise SystemExit(1)

    dataset = SolitaireStateDataset(log_paths)
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

    model = PolicyValueNet(state_dim=state_dim, num_actions=num_actions)
    model.to(device)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.BCEWithLogitsLoss()
    metric_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=64, shuffle=False)

    num_epochs = 5
    
    # Loss weights for multi-task learning
    weight_value = 0.3
    weight_metrics = 0.5

    print(
        f"Training on {len(train_ds)} samples, validating on {len(validation_ds)} samples "
        f"(state_dim={state_dim}, num_actions={num_actions}, device={device})"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_metric_loss = 0.0
        total_correct_policy = 0
        total_correct_value = 0
        total_correct_foundation = 0
        total_correct_revealed = 0
        total_correct_talon = 0
        total_correct_cascade = 0
        total_examples = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                print(f"  Epoch {epoch}/{num_epochs} - Batch {batch_idx:04d}/{len(train_loader):04d}")
            
            # Extract tensors from batch dict
            states = batch['state'].to(device)
            target_actions = batch['policy'].argmax(dim=-1).to(device)
            target_values = batch['value'].to(device)
            target_foundation = batch['foundation_move'].to(device)
            target_revealed = batch['revealed_facedown'].to(device)
            target_talon = batch['talon_move'].to(device)
            target_cascade = batch['is_cascading_move'].to(device)

            # Forward pass
            outputs = model(states)
            policy_logits = outputs['policy']
            value_logits = outputs['value'].squeeze(-1)
            foundation_logits = outputs['foundation_move'].squeeze(-1)
            revealed_logits = outputs['revealed_facedown'].squeeze(-1)
            talon_logits = outputs['talon_move'].squeeze(-1)
            cascade_logits = outputs['cascading_move'].squeeze(-1)

            # Compute losses
            p_loss = policy_loss_fn(policy_logits, target_actions)
            v_loss = value_loss_fn(value_logits, target_values)
            foundation_loss = metric_loss_fn(foundation_logits, target_foundation)
            revealed_loss = metric_loss_fn(revealed_logits, target_revealed)
            talon_loss = metric_loss_fn(talon_logits, target_talon)
            cascade_loss = metric_loss_fn(cascade_logits, target_cascade)
            
            # Combined loss with weights
            metric_losses = (foundation_loss + revealed_loss + talon_loss + cascade_loss) / 4.0
            loss = p_loss + weight_value * v_loss + weight_metrics * metric_losses

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
                
                foundation_probs = torch.sigmoid(foundation_logits)
                foundation_pred = (foundation_probs >= 0.5).float()
                foundation_correct = (foundation_pred == target_foundation).sum().item()
                
                revealed_probs = torch.sigmoid(revealed_logits)
                revealed_pred = (revealed_probs >= 0.5).float()
                revealed_correct = (revealed_pred == target_revealed).sum().item()
                
                talon_probs = torch.sigmoid(talon_logits)
                talon_pred = (talon_probs >= 0.5).float()
                talon_correct = (talon_pred == target_talon).sum().item()
                
                cascade_probs = torch.sigmoid(cascade_logits)
                cascade_pred = (cascade_probs >= 0.5).float()
                cascade_correct = (cascade_pred == target_cascade).sum().item()

            batch_size = states.size(0)
            total_examples += batch_size
            total_policy_loss += p_loss.item() * batch_size
            total_value_loss += v_loss.item() * batch_size
            total_metric_loss += metric_losses.item() * batch_size
            total_correct_policy += policy_correct
            total_correct_value += value_correct
            total_correct_foundation += foundation_correct
            total_correct_revealed += revealed_correct
            total_correct_talon += talon_correct
            total_correct_cascade += cascade_correct

        avg_policy_loss = total_policy_loss / total_examples
        avg_value_loss = total_value_loss / total_examples
        avg_metric_loss = total_metric_loss / total_examples
        train_policy_accuracy = total_correct_policy / total_examples
        train_value_accuracy = total_correct_value / total_examples
        train_foundation_accuracy = total_correct_foundation / total_examples
        train_revealed_accuracy = total_correct_revealed / total_examples
        train_talon_accuracy = total_correct_talon / total_examples
        train_cascade_accuracy = total_correct_cascade / total_examples

        model.eval()
        validation_policy_loss = 0.0
        validation_value_loss = 0.0
        validation_metric_loss = 0.0
        validation_correct_policy = 0
        validation_correct_value = 0
        validation_correct_foundation = 0
        validation_correct_revealed = 0
        validation_correct_talon = 0
        validation_correct_cascade = 0
        validation_examples = 0

        with torch.no_grad():
            for batch in validation_loader:
                states = batch['state'].to(device)
                target_actions = batch['policy'].argmax(dim=-1).to(device)
                target_values = batch['value'].to(device)
                target_foundation = batch['foundation_move'].to(device)
                target_revealed = batch['revealed_facedown'].to(device)
                target_talon = batch['talon_move'].to(device)
                target_cascade = batch['is_cascading_move'].to(device)

                outputs = model(states)
                policy_logits = outputs['policy']
                value_logits = outputs['value'].squeeze(-1)
                foundation_logits = outputs['foundation_move'].squeeze(-1)
                revealed_logits = outputs['revealed_facedown'].squeeze(-1)
                talon_logits = outputs['talon_move'].squeeze(-1)
                cascade_logits = outputs['cascading_move'].squeeze(-1)

                p_loss = policy_loss_fn(policy_logits, target_actions)
                v_loss = value_loss_fn(value_logits, target_values)
                foundation_loss = metric_loss_fn(foundation_logits, target_foundation)
                revealed_loss = metric_loss_fn(revealed_logits, target_revealed)
                talon_loss = metric_loss_fn(talon_logits, target_talon)
                cascade_loss = metric_loss_fn(cascade_logits, target_cascade)
                
                metric_losses = (foundation_loss + revealed_loss + talon_loss + cascade_loss) / 4.0

                pred_actions = policy_logits.argmax(dim=-1)
                policy_correct = (pred_actions == target_actions).sum().item()

                value_probs = torch.sigmoid(value_logits)
                value_pred = (value_probs >= 0.5).float()
                value_correct = (value_pred == target_values).sum().item()
                
                foundation_probs = torch.sigmoid(foundation_logits)
                foundation_pred = (foundation_probs >= 0.5).float()
                foundation_correct = (foundation_pred == target_foundation).sum().item()
                
                revealed_probs = torch.sigmoid(revealed_logits)
                revealed_pred = (revealed_probs >= 0.5).float()
                revealed_correct = (revealed_pred == target_revealed).sum().item()
                
                talon_probs = torch.sigmoid(talon_logits)
                talon_pred = (talon_probs >= 0.5).float()
                talon_correct = (talon_pred == target_talon).sum().item()
                
                cascade_probs = torch.sigmoid(cascade_logits)
                cascade_pred = (cascade_probs >= 0.5).float()
                cascade_correct = (cascade_pred == target_cascade).sum().item()

                batch_size = states.size(0)
                validation_examples += batch_size
                validation_policy_loss += p_loss.item() * batch_size
                validation_value_loss += v_loss.item() * batch_size
                validation_metric_loss += metric_losses.item() * batch_size
                validation_correct_policy += policy_correct
                validation_correct_value += value_correct
                validation_correct_foundation += foundation_correct
                validation_correct_revealed += revealed_correct
                validation_correct_talon += talon_correct
                validation_correct_cascade += cascade_correct

        avg_validation_policy_loss = validation_policy_loss / validation_examples
        avg_validation_value_loss = validation_value_loss / validation_examples
        avg_validation_metric_loss = validation_metric_loss / validation_examples
        validation_policy_accuracy = validation_correct_policy / validation_examples
        validation_value_accuracy = validation_correct_value / validation_examples
        validation_foundation_accuracy = validation_correct_foundation / validation_examples
        validation_revealed_accuracy = validation_correct_revealed / validation_examples
        validation_talon_accuracy = validation_correct_talon / validation_examples
        validation_cascade_accuracy = validation_correct_cascade / validation_examples

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss(p={avg_policy_loss:.3f}, v={avg_value_loss:.3f}, m={avg_metric_loss:.3f}), "
            f"train_accuracy(p={train_policy_accuracy:.3f}, v={train_value_accuracy:.3f}, "
            f"f={train_foundation_accuracy:.3f}, r={train_revealed_accuracy:.3f}, "
            f"t={train_talon_accuracy:.3f}, c={train_cascade_accuracy:.3f}) "
            f"- validation_loss(p={avg_validation_policy_loss:.3f}, v={avg_validation_value_loss:.3f}, m={avg_validation_metric_loss:.3f}), "
            f"validation_accuracy(p={validation_policy_accuracy:.3f}, v={validation_value_accuracy:.3f}, "
            f"f={validation_foundation_accuracy:.3f}, r={validation_revealed_accuracy:.3f}, "
            f"t={validation_talon_accuracy:.3f}, c={validation_cascade_accuracy:.3f})"
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


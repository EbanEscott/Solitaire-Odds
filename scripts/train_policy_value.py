"""
Train a policy–value network on Solitaire logs from the Java engine.

Usage (from engine project root):

    python3 scripts/train_policy_value.py /Users/ebo/Code/solitaire/engine/logs/game.log

You can also pass multiple log files:

    python3 scripts/train_policy_value.py /path/to/log1.log /path/to/log2.log
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from alphasolitaire.dataset import SolitaireStateDataset
from alphasolitaire.model import PolicyValueNet


def main(argv: List[str]) -> None:
    if not argv:
        print("Usage: python3 scripts/train_policy_value.py /path/to/game.log [more_logs.log ...]")
        raise SystemExit(1)

    log_paths = [Path(p) for p in argv]
    for p in log_paths:
        if not p.exists():
            print(f"Log file not found: {p}")
            raise SystemExit(1)

    dataset = SolitaireStateDataset(log_paths)
    if len(dataset) == 0:
        print("Dataset is empty; ensure the Java engine was run with -Dlog.episodes=true.")
        raise SystemExit(1)

    # Train/validation split (e.g., 90% / 10%).
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    # Peek at one sample for dimensions.
    sample_state, sample_policy, _ = dataset[0]
    state_dim = sample_state.shape[0]
    num_actions = sample_policy.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyValueNet(state_dim=state_dim, num_actions=num_actions)
    model.to(device)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    num_epochs = 5

    print(
        f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples "
        f"(state_dim={state_dim}, num_actions={num_actions}, device={device})"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_correct_policy = 0
        total_correct_value = 0
        total_examples = 0

        for states, policies, values in train_loader:
            states = states.to(device)
            # For the policy, use the one-hot target's argmax as class index.
            target_actions = policies.argmax(dim=-1).to(device)
            target_values = values.to(device)

            logits, value_logits = model(states)

            policy_loss = policy_loss_fn(logits, target_actions)
            value_loss = value_loss_fn(value_logits.squeeze(-1), target_values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics.
            with torch.no_grad():
                pred_actions = logits.argmax(dim=-1)
                policy_correct = (pred_actions == target_actions).sum().item()

                value_probs = torch.sigmoid(value_logits.squeeze(-1))
                value_pred = (value_probs >= 0.5).float()
                value_correct = (value_pred == target_values).sum().item()

            batch_size = states.size(0)
            total_examples += batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_correct_policy += policy_correct
            total_correct_value += value_correct

        avg_policy_loss = total_policy_loss / total_examples
        avg_value_loss = total_value_loss / total_examples
        train_policy_acc = total_correct_policy / total_examples
        train_value_acc = total_correct_value / total_examples

        # Validation.
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct_policy = 0
        val_correct_value = 0
        val_examples = 0

        with torch.no_grad():
            for states, policies, values in val_loader:
                states = states.to(device)
                target_actions = policies.argmax(dim=-1).to(device)
                target_values = values.to(device)

                logits, value_logits = model(states)

                p_loss = policy_loss_fn(logits, target_actions)
                v_loss = value_loss_fn(value_logits.squeeze(-1), target_values)

                pred_actions = logits.argmax(dim=-1)
                policy_correct = (pred_actions == target_actions).sum().item()

                value_probs = torch.sigmoid(value_logits.squeeze(-1))
                value_pred = (value_probs >= 0.5).float()
                value_correct = (value_pred == target_values).sum().item()

                batch_size = states.size(0)
                val_examples += batch_size
                val_policy_loss += p_loss.item() * batchSize
                val_value_loss += v_loss.item() * batchSize
                val_correct_policy += policy_correct
                val_correct_value += value_correct

        avg_val_policy_loss = val_policy_loss / val_examples
        avg_val_value_loss = val_value_loss / val_examples
        val_policy_acc = val_correct_policy / val_examples
        val_value_acc = val_correct_value / val_examples

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss(p={avg_policy_loss:.3f}, v={avg_value_loss:.3f}), "
            f"train_acc(p={train_policy_acc:.3f}, v={train_value_acc:.3f}) "
            f"- val_loss(p={avg_val_policy_loss:.3f}, v={avg_val_value_loss:.3f}), "
            f"val_acc(p={val_policy_acc:.3f}, v={val_value_acc:.3f})"
        )

    # Optionally save the final model parameters.
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
    main(sys.argv[1:])

#!/usr/bin/env python3
"""
Promote a trained checkpoint from policy_value_latest.pt to a versioned checkpoint.

Usage:
    python tools/promote_checkpoint.py --name level4 --description "Trained on L2+L3+L4 endgames"

This script:
1. Reads metadata from checkpoints/policy_value_latest.pt
2. Generates a timestamp-based filename: policy_value_level{N}_YYYYMMDDTHHMMSS.pt
3. Copies the checkpoint to the versioned filename
4. Creates a companion .md file with training stats and performance notes
5. Updates checkpoints/VERSIONS.md with the new entry
"""

import argparse
import shutil
import torch
from datetime import datetime
from pathlib import Path
import json
import sys


def load_checkpoint(checkpoint_path):
    """Load a PyTorch checkpoint and return the dict."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        sys.exit(1)


def extract_metadata(checkpoint):
    """Extract metadata dict from checkpoint, or create empty if not present."""
    return checkpoint.get('metadata', {})


def generate_timestamp():
    """Generate ISO format timestamp: YYYYMMDDTHHMMSS."""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def format_float(value, decimals=4):
    """Format a numeric value with specified decimals, or return 'N/A' if not a number."""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return "N/A"


def format_percent(value):
    """Format a numeric value as percentage, or return 'N/A' if not a number."""
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return "N/A"


def generate_markdown(checkpoint, metadata, level_name, description, timestamp):
    """Generate a markdown file documenting the checkpoint."""
    
    # Extract information with defaults
    arch = metadata.get('architecture', {})
    hyperparams = metadata.get('hyperparameters', {})
    final_metrics = metadata.get('final_metrics', {})
    data_sources = metadata.get('data_sources', [])
    
    timestamp_str = metadata.get('timestamp', 'Unknown')
    git_engine = metadata.get('git_commit_engine', 'Unknown')
    git_neural = metadata.get('git_commit_neural', 'Unknown')
    
    training_samples = metadata.get('training_samples', 'Unknown')
    validation_samples = metadata.get('validation_samples', 'Unknown')
    training_duration = metadata.get('training_duration_seconds', 'Unknown')
    
    # Format training duration
    if isinstance(training_duration, (int, float)) and training_duration != 'Unknown':
        hours = int(training_duration) // 3600
        minutes = (int(training_duration) % 3600) // 60
        duration_str = f"{hours}h {minutes}m"
    else:
        duration_str = str(training_duration)
    
    # Build markdown content
    md = f"""# Policy-Value Network: {level_name} Training

**Checkpoint:** `policy_value_{level_name}_{timestamp}.pt`  
**Trained:** {timestamp_str}  
**Purpose:** {description}

## Metrics

| Metric | Value |
|--------|-------|
| Training Policy Loss | {format_float(final_metrics.get('training_policy_loss'))} |
| Training Value Loss | {format_float(final_metrics.get('training_value_loss'))} |
| Training Policy Accuracy | {format_percent(final_metrics.get('training_policy_accuracy'))} |
| Training Value Accuracy | {format_percent(final_metrics.get('training_value_accuracy'))} |
| Validation Policy Loss | {format_float(final_metrics.get('validation_policy_loss'))} |
| Validation Value Loss | {format_float(final_metrics.get('validation_value_loss'))} |
| Validation Policy Accuracy | {format_percent(final_metrics.get('validation_policy_accuracy'))} |
| Validation Value Accuracy | {format_percent(final_metrics.get('validation_value_accuracy'))} |

## Architecture

- Hidden Dim: {arch.get('hidden_dim', 'N/A')}
- Layers: {arch.get('num_layers', 'N/A')}
- Batch Norm: {arch.get('batch_norm', 'N/A')}
- Residual: {arch.get('residual', 'N/A')}

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {hyperparams.get('epochs', 'N/A')} |
| Batch Size | {hyperparams.get('batch_size', 'N/A')} |
| Learning Rate | {hyperparams.get('learning_rate', 'N/A')} |
| Training Duration | {duration_str} |

## Dataset

- Training Samples: {training_samples}
- Validation Samples: {validation_samples}
- Data Sources:
"""
    
    if data_sources:
        for source in data_sources:
            md += f"  - `{source}`\n"
    else:
        md += "  - Unknown\n"
    
    md += f"""
## Reproducibility

- Engine Commit: `{git_engine}`
- Neural-Network Commit: `{git_neural}`
- Python Version: {metadata.get('python_version', 'Unknown')}
- PyTorch Version: {metadata.get('pytorch_version', 'Unknown')}
- Feature Dimension: {checkpoint.get('feature_dim', 'Unknown')}
- Action Space Size: {checkpoint.get('action_space_size', 'Unknown')}

## Notes

Generated by `tools/promote_checkpoint.py` on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC.

Add additional performance notes, test results, and observations below:

---

(Add any test results, win rates, or performance observations here)
"""
    
    return md


def update_versions_table(checkpoint_dir, level_name, timestamp, checkpoint_file):
    """Update VERSIONS.md with the new checkpoint entry."""
    versions_path = checkpoint_dir / "VERSIONS.md"
    
    checkpoint = torch.load(checkpoint_dir / "policy_value_latest.pt", map_location='cpu', weights_only=False)
    metadata = checkpoint.get('metadata', {})
    final_metrics = metadata.get('final_metrics', {})
    
    # Create VERSIONS.md if it doesn't exist
    if not versions_path.exists():
        versions_content = """# Model Versions

| Checkpoint | Level | Date | Policy Acc | Value Acc | Notes |
|-----------|-------|------|-----------|-----------|-------|
"""
    else:
        with open(versions_path, 'r') as f:
            versions_content = f.read()
    
    # Extract date from timestamp (YYYYMMDDTHHMMSS -> YYYY-MM-DD)
    date_str = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    
    # Create new row
    policy_accuracy = final_metrics.get('validation_policy_accuracy', 0.0)
    value_accuracy = final_metrics.get('validation_value_accuracy', 0.0)
    
    new_row = f"| `{checkpoint_file}` | {level_name} | {date_str} | {policy_accuracy:.2%} | {value_accuracy:.2%} | Promoted from latest |\n"
    
    # Append to table (insert before any trailing content)
    if "| `" in versions_content:
        # Table already exists, append to it
        lines = versions_content.split('\n')
        insert_idx = len(lines) - 1
        while insert_idx > 0 and lines[insert_idx].strip() == '':
            insert_idx -= 1
        lines.insert(insert_idx + 1, new_row)
        versions_content = '\n'.join(lines)
    else:
        # Just append
        versions_content += new_row
    
    with open(versions_path, 'w') as f:
        f.write(versions_content)
    
    print(f"Updated {versions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote a checkpoint from policy_value_latest.pt to a versioned checkpoint"
    )
    parser.add_argument(
        '--name',
        required=True,
        help='Level name (e.g., "level4", "level5"). Checkpoint will be named policy_value_level{N}_TIMESTAMP.pt'
    )
    parser.add_argument(
        '--description',
        default='Promoted checkpoint',
        help='Description of the checkpoint purpose (e.g., "Trained on L2+L3+L4 endgames")'
    )
    parser.add_argument(
        '--source',
        default='checkpoints/policy_value_latest.pt',
        help='Source checkpoint path (default: checkpoints/policy_value_latest.pt)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to neural-network/ directory
    script_dir = Path(__file__).parent.parent  # neural-network/
    source_path = script_dir / args.source
    checkpoint_dir = script_dir / 'checkpoints'
    
    print(f"Promoting checkpoint: {args.name}")
    print(f"Source: {source_path}")
    
    # Load source checkpoint
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(source_path)
    metadata = extract_metadata(checkpoint)
    
    # Generate timestamp
    timestamp = generate_timestamp()
    print(f"Timestamp: {timestamp}")
    
    # Generate filenames
    checkpoint_filename = f"policy_value_{args.name}_{timestamp}.pt"
    markdown_filename = f"policy_value_{args.name}_{timestamp}.md"
    
    dest_checkpoint_path = checkpoint_dir / checkpoint_filename
    dest_markdown_path = checkpoint_dir / markdown_filename
    
    print(f"Destination checkpoint: {checkpoint_filename}")
    print(f"Destination markdown: {markdown_filename}")
    
    # Copy checkpoint
    print("Copying checkpoint...")
    shutil.copy(source_path, dest_checkpoint_path)
    print(f"✓ Created {dest_checkpoint_path}")
    
    # Generate markdown
    print("Generating markdown...")
    md_content = generate_markdown(checkpoint, metadata, args.name, args.description, timestamp)
    with open(dest_markdown_path, 'w') as f:
        f.write(md_content)
    print(f"✓ Created {dest_markdown_path}")
    
    # Update VERSIONS.md
    print("Updating VERSIONS.md...")
    update_versions_table(checkpoint_dir, args.name, timestamp, checkpoint_filename)
    
    print("\n✓ Promotion complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the markdown file: {dest_markdown_path}")
    print(f"  2. Add test results and performance notes to the markdown")
    print(f"  3. Commit both files to Git:")
    print(f"     git add {dest_checkpoint_path} {dest_markdown_path} {checkpoint_dir / 'VERSIONS.md'}")
    print(f"     git commit -m 'Promote {checkpoint_filename}'")


if __name__ == '__main__':
    main()

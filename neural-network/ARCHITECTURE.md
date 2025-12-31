# Neural Network Architecture Guide

This document explains the configurable neural network design and why what we're exploring for Solitaire's planning problem.

## The Four Core Neural Network Families

Neural networks come in distinct families, each designed for different types of data structure. The key difference is what kind of **inductive bias** (built-in assumption) the architecture has. A grid-based CNN assumes spatial locality. A sequence-based Transformer assumes order matters. A graph-based GNN assumes relationships between entities matter. An MLP assumes none of these — it's pure function approximation.

| Property                  | MLP                  | CNN  | Transformer | GNN    |
| ------------------------- | -------------------- | ---- | ----------- | ------ |
| Data structure            | Vector (flat)        | Grid | Sequence    | Graph  |
| Fixed size                | Yes              | Yes  | No          | No     |
| Order-sensitive           | Only by encoding | Yes  | Yes         | No     |
| Relational inductive bias | None             | Weak | Medium      | Strong |

**MLP (Multi-Layer Perceptron):** Fully-connected layers on flat vectors with no special structure—works on anything but learns nothing about relationships. **CNN (Convolutional Neural Network):** Specialized for grids (like images or Go boards) where local spatial patterns repeat and translation invariance matters. **Transformer:** Built for sequences (text, time series) where each element can attend to any other and position is explicit. **GNN (Graph Neural Network):** Designed for graph-structured data where relationships between entities matter—learning flows through edges via message passing.

## Why Explore GNN for Solitaire

Solitaire is fundamentally a planning game where move value depends on what downstream consequences it enables. The hypothesis: tree structure from A* search encodes reasoning that flat statistics lose. GNN's relational inductive bias could preserve causal relationships that MLP flattening destroys. However, we don't know yet whether this actually matters in practice—visit counts alone might suffice, or MLP's capacity might be enough. This is why we're building both models. For full context on the hypothesis, design questions, and implementation details, see `NEURAL_NETWORK_PLAN.md`.

## Current Implementation Path

**Phase 1: MLP Baseline.** Train MLP (512 hidden, 3 layers) on easy levels to verify the network can learn Solitaire fundamentals from board state alone. Uses only `EPISODE_STEP` records. Checkpoint: `policy_value_mpl_v1.pt`. See `NEURAL_NETWORK_PLAN.md` for training setup.

**Phase 2: Tree Logging.** Add `EPISODE_TREE` records to episode logs capturing A* exploration (moves, visit counts, heuristics). Tree depth limited to current node + immediate children. Prepares data for GNN training. See `NEURAL_NETWORK_PLAN.md` for logging format.

**Phase 3: GNN Model.** Implement pure GNN treating board features as root node and move tree as subgraph. GNN can learn whether tree structure matters for move selection. Checkpoint: `policy_value_gnn_v1.pt`. See `NEURAL_NETWORK_PLAN.md` for architecture decisions.

**Phase 4: Multi-Model Support.** Maintain both MLP and GNN in codebase via `ModelFactory` to enable direct comparison. Compare performance on same test set to determine which approach is stronger. Prepare for self-play with the winning architecture.

**Phase 5: Self-Play Loop.** Network priors guide MCTS, MCTS discovers better move distributions, network retrains on MCTS data, versions improve iteratively. This is the ultimate goal where network and search co-evolve. See `NEURAL_NETWORK_PLAN.md` Phase 5 for full details.

## Architecture Configuration

### Basic Options

```bash
# Small (default, fast training)
python -m src.train_policy_value --hidden-dim 256 --num-layers 2 logs/episode.log

# Medium (balanced)
python -m src.train_policy_value --hidden-dim 512 --num-layers 3 logs/episode.log

# Large (for full game tree training)
python -m src.train_policy_value --hidden-dim 1024 --num-layers 3 logs/episode.log

# Extra Large (if you have millions of samples)
python -m src.train_policy_value --hidden-dim 2048 --num-layers 4 logs/episode.log
```

### Detailed Parameter Reference

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--hidden-dim` | 256 | 128-2048+ | Width of hidden layers. Larger = more capacity but needs more data. |
| `--num-layers` | 2 | 1-5+ | Depth of shared backbone. 3-4 recommended for complex patterns. |
| `--batch-norm` | False | - | Apply batch normalization after each layer (experimental). Helps with training stability. |
| `--residual` | False | - | Add residual connections (experimental). Useful for very deep networks (4+ layers). |
| `--epochs` | 5 | 1-50 | Training epochs. Increase if underfitting, decrease if overfitting. |
| `--batch-size` | 64 | 32-256 | Larger batches = faster training but less frequent updates. |
| `--learning-rate` | 1e-3 | 1e-5 to 1e-1 | Adam learning rate. Decrease if unstable, increase if slow to converge. |

## Size vs. Capacity Trade-offs

```
Configuration              | Parameters | Checkpoint MB | Training Speed | Data Needed
Small (256, 2)            | 169k       | 0.64          | ~30s/epoch     | ~10k samples
Medium (512, 3)           | 731k       | 2.79          | ~60s/epoch     | ~50k samples
Large (1024, 3)           | 2.5M       | 9.6           | ~200s/epoch    | ~200k samples
XL (2048, 3)              | 9.2M       | 35            | ~500s/epoch    | ~1M samples
Huge (2048, 4)            | 12.5M      | 47.5          | ~700s/epoch    | ~2M samples
```
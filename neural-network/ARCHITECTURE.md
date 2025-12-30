# Neural Network Architecture Guide

This document explains the configurable neural network design and how to use it for training on full game trajectories (the key to effective self-play).

## The Core Insight: Full Trajectory Training vs. Isolated States

**Old approach (limited):**
- Train on individual game states in isolation
- Each state → one label (policy + value)
- No context about the game trajectory
- Network struggles to understand move sequences

**New approach (powerful):**
- Train on **full game trajectories** (start → finish)
- Each state in trajectory labeled with:
  - **Policy**: the actual move taken (supervised from A* or MCTS)
  - **Value**: the final game outcome (1.0 = win, 0.0 = loss)
  - Optionally: bootstrapped value from V(next_state) for RL
- Network learns move dependencies and long-range planning
- This is exactly how AlphaGo trained: supervised first, then self-play

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

## For Full Game Tree Training (Recommended)

When training on entire game trajectories from MCTS or self-play:

**Phase 1: Bootstrap from A* (what you're doing now)**
```bash
# Use medium model: 512 hidden, 3 layers
# This captures enough complexity without overfitting on limited A* data
python -m src.train_policy_value \
  --hidden-dim 512 \
  --num-layers 3 \
  "engine/logs/episode*.log"
```

**Phase 2: Self-play loop (future)**
```bash
# Once you have 100k+ self-play samples, scale up
python -m src.train_policy_value \
  --hidden-dim 1024 \
  --num-layers 3 \
  --batch-size 128 \
  "self_play_logs/episodes_v*.log"
```

**Phase 3: Deep refinement (if converging poorly)**
```bash
# Add batch norm for deeper networks
python -m src.train_policy_value \
  --hidden-dim 1024 \
  --num-layers 4 \
  --batch-norm \
  "self_play_logs/episodes_v*.log"
```

## Trajectory-Aware Value Targets

The dataset now supports **two value target modes**:

### Mode 1: Trajectory Value (Current)
- Every step in an episode is labeled with the final game outcome
- Example: A 50-step game that wins → all 50 steps labeled with 1.0
- Simple, works well for supervised learning
- Enabled by default: `trajectory_config.use_trajectory_value=True`

### Mode 2: Bootstrapped Value (For Self-Play RL)
```python
trajectory_config = TrajectoryConfig(
    use_trajectory_value=False,
    use_bootstrapped_value=True,
    discount_factor=0.99,  # γ for discount
)
```
- Value target: `V(s) = r + γ * V(next_state)`
- Requires value network to be pre-trained
- Enables efficient RL when combined with MCTS
- Will be used once self-play loop is implemented

## Impact on Training

### Policy Head (move prediction)
- Learns: "What move did the AI take at each state in winning/losing games?"
- Trajectory helps: Network sees sequences of moves, learns combinations
- Better priors for MCTS (reduces search branching)

### Value Head (win prediction)
- Learns: "Is this state on a winning trajectory?"
- Trajectory helps: Network learns which game positions lead to success
- States appearing in both winning and losing games force the network to learn critical features

### Multi-task Heads (foundation, talon, cascading moves)
- Aux losses help the shared backbone learn general Solitaire concepts
- Reduces overfitting to policy/value alone

## Practical Training Tips

1. **Start small, scale gradually**
   - (256, 2): Quick experimentation, ~10-30 seconds per epoch
   - (512, 3): Good balance, ~1-2 minutes per epoch
   - Only go larger if you have 200k+ samples

2. **Monitor validation accuracy**
   - Policy accuracy < 70% → need more data or deeper network
   - Value accuracy > 95% → network is learning but policy is weak
   - Both > 80% → good model, ready for self-play

3. **Adjust learning rate for batch size**
   - Larger batches (128-256) may need higher learning rate (5e-4 to 1e-3)
   - Smaller batches (32) may need lower learning rate (5e-4 to 1e-4)

4. **For full game tree training**
   - Use `--batch-size 128` (more data per gradient step)
   - Use `--num-layers 3` or 4 (capture long-range dependencies)
   - Train for longer (10-20 epochs) on accumulated self-play data

5. **Checkpoint frequently**
   - Model saves to `checkpoints/policy_value_latest.pt`
   - In self-play loop, save versioned checkpoints: `v1.pt`, `v2.pt`, etc.
   - Compare win rates between versions to track improvement

## Next Steps: Self-Play Loop

Once you have a trained model:

1. **Run MCTS games** using the neural network for guidance
2. **Log each game trajectory** (all states and MCTS-selected moves)
3. **Retrain on the new data** with bootstrapped value targets
4. **Compare model versions** (v1 vs v2 win rates)
5. **Repeat** (reinforcement learning loop)

This is the full AlphaGo approach:
- Supervised bootstrap (A* → v0)
- Self-play refinement (v0 plays itself → v1)
- Continuous improvement (v1 plays itself → v2, etc.)

## References

- AlphaGo paper: Uses similar supervised→self-play pipeline
- AlphaZero: Pure self-play (no supervised phase)
- Your codebase: `TrainingOpponent` can generate seeded positions for RL games


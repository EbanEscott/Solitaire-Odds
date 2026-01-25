# Quick Start: Full Game Tree Training

## The Key Concept

You now have **trajectory-aware training**: every step in a game trajectory is labeled with the full game outcome. This is what makes self-play effective—the network learns not just individual moves, but movement sequences and long-range planning.

## Commands to Try

### 1. Train with Default Config (What You're Doing)
```bash
cd /Users/ebo/Code/solitaire/neural-network
source .venv/bin/activate

# Default: 256 hidden, 2 layers
python -m src.train_policy_value /Users/ebo/Code/solitaire/engine/logs/episode.log
```

### 2. Train with Medium Model (Recommended for Full Game Trees)
```bash
# 512 hidden, 3 layers = ~730k params, ~2.8 MB
# Good balance for 50k-200k samples
python -m src.train_policy_value \
  --hidden-dim 512 \
  --num-layers 3 \
  /Users/ebo/Code/solitaire/engine/logs/episode.log
```

### 3. Train with Large Model (Once Self-Play Generates 200k+ Samples)
```bash
# 1024 hidden, 3 layers = ~2.5M params, ~9.6 MB
python -m src.train_policy_value \
  --hidden-dim 1024 \
  --num-layers 3 \
  --batch-size 128 \
  "engine/logs/episode*.log"
```

### 4. Experiment with Deeper Networks (4+ Layers for Complex Sequences)
```bash
# 512 hidden, 4 layers = ~994k params, ~3.8 MB
python -m src.train_policy_value \
  --hidden-dim 512 \
  --num-layers 4 \
  "engine/logs/episode*.log"
```

## What's Different Now

### Before
- Each game step = one isolated training example
- No trajectory context
- Network learns: "What's the best move here?"
- Limited effectiveness for complex sequences

### After
- Full game trajectory = multiple connected examples
- Network learns: "Given the entire game sequence, what move was taken, and does this path lead to victory?"
- Example: A 50-step winning game = 50 training pairs, all labeled with 1.0 (win)
- Network sees patterns: "Foundation moves early → better endgame outcomes"

## Why This Enables Self-Play

1. **Supervised bootstrap** (current): Train on A* games to learn good move patterns
2. **Self-play loop** (coming): 
   - MCTS uses the network as prior
   - Play games with MCTS
   - Log full trajectories with MCTS-selected moves
   - Retrain on those trajectories
   - Network improves → MCTS gets better priors → better moves → cycle

The trajectory labels make step 3 work: the network can see that "this move sequence worked in self-play."

## Testing Your Setup

```bash
# Generate a few test episodes (if not already done)
cd /Users/ebo/Code/solitaire/engine
./gradlew test --tests ai.games.results.AStarPlayerResultsTest "-Dlog.episodes=true"

# Switch to Python environment
cd ../neural-network
source .venv/bin/activate

# Quick test with small model
python -m src.train_policy_value \
  --hidden-dim 256 \
  --num-layers 2 \
  --epochs 2 \
  /Users/ebo/Code/solitaire/engine/logs/episode.log
```

Expected output:
```
Loading 1 file(s)...
Total: X episode(s) loaded
Building action space...
→ Y unique actions
Model Architecture: hidden_dim=256, num_layers=2, batch_norm=False, residual=False
Model Size: 169,XXX total parameters, 169,XXX trainable
Estimated checkpoint size: 0.64 MB
Training: 2 epochs, batch_size=64, lr=0.001
Epoch 1/2 - ...
```

## Architecture Recommendations by Dataset Size

| Samples | Model Config | Command |
|---------|--------------|---------|
| < 10k | 256, 2 layers | `--hidden-dim 256 --num-layers 2` |
| 10k-50k | 512, 3 layers | `--hidden-dim 512 --num-layers 3` |
| 50k-200k | 1024, 3 layers | `--hidden-dim 1024 --num-layers 3` |
| 200k+ | 2048, 3-4 layers | `--hidden-dim 2048 --num-layers 4` |

## Next Steps

1. **Run training with different architectures** and compare validation accuracy
2. **Monitor the value head** (should learn the game outcome very well)
3. **Monitor the policy head** (should improve with more trajectory data)
4. **Once MCTS is working**, implement self-play loop to generate unlimited training data

## Files Changed

- `neural-network/src/model.py`: Now supports `hidden_dim`, `num_layers`, `batch_norm`, `residual`
- `neural-network/src/dataset.py`: Now supports trajectory-aware value targets
- `neural-network/src/train_policy_value.py`: Command-line args for architecture tuning
- `neural-network/ARCHITECTURE.md`: Detailed configuration guide


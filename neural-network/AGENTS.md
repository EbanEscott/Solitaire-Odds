# Agents – Neural Network Module

## Module Overview

This `neural-network/` directory contains AlphaSolitaire: a policy–value network trained on solitaire game trajectories.

**Current Focus:** Phase 1 — Verify configurable architecture works, establish baseline on A* episodes

See [NEURAL_NETWORK_PLAN.md](NEURAL_NETWORK_PLAN.md) for the full 4-phase roadmap with details, timeline, and open questions.

## Python Code Guidelines

When contributing to this module:

- Keep modules importable from project root: `python -m src.<module>`
- Use relative imports within `src/` package
- Avoid heavy dependencies beyond `requirements.txt`
- Focus on Solitaire modeling (no unrelated utilities)

## Quick Start

Train the network with configurable architecture:

```bash
cd neural-network

# Generate A* episodes (or use existing logs/)
cd ../engine
./gradlew run --args="astar --num-games 100 --output ../neural-network/logs/episode.log"

# Train with custom architecture
cd ../neural-network
python -m src.train_policy_value \
  --hidden-dim 512 \
  --num-layers 3 \
  --batch-size 32 \
  --epochs 20 \
  logs/episode.log
```

Available flags:
- `--hidden-dim` (default 256): Network width (try 256–2048+)
- `--num-layers` (default 3): Network depth (try 1–5+)
- `--batch-size` (default 32): Batch size
- `--epochs` (default 20): Training epochs
- `--learning-rate` (default 0.001): Learning rate
- `--batch-norm`: Enable batch norm
- `--residual`: Enable residual connections

## Network Architecture

`PolicyValueNet` is a 2-head PyTorch network:

| Config | Params | Checkpoint Size |
|--------|--------|-----------------|
| 256 hidden, 3 layers | 1.3M | 5MB |
| 512 hidden, 3 layers | 3.2M | 12MB |
| 1024 hidden, 5 layers | 12M+ | 50MB |

Both heads share a trunk of configurable layers. Use larger models only with sufficient training data (10k+ samples recommended).

## Roadmap

| Phase | Goal | Dependencies |
|-------|------|---------------|
| 1 | Baseline on A* episodes | None, can start now |
| 2 | Unified GameTree class | Refactor A* & MCTS |
| 3 | Log search trees in episodes | Phase 2 complete |
| 4 | Self-play loop (MCTS + network) | Phase 3 + MCTS diagnostics |

**Phase 1 Success Criteria:** Policy acc > 70%, Value acc > 95%

**Blocker:** MCTS currently 0% win rate — needs diagnostics before Phase 4

See [NEURAL_NETWORK_PLAN.md](NEURAL_NETWORK_PLAN.md) for:
- What each phase involves
- Why trajectory-aware training needs search context
- Open questions (tree JSON format, visit count labels, bootstrapped values, etc.)

## Key Insight

Current training: `State → Network → Move`  
Future training: `State + Search Context → Network → Move Distribution`

The real improvement comes from training the network to see what search discovered, then using those improved priors to make search better. This creates the feedback loop that makes AlphaGo work.

We're not doing that yet because episodes don't log search trees. Phase 3 fixes that.

## Engine Integration

For AlphaSolitaire integration details, see:
- `engine/src/main/java/ai/games/player/ai/alpha/README.md`
- `engine/AGENTS.md`
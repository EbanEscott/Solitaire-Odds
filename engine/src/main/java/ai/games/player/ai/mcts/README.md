# Monte Carlo Tree Search Player Design

## Overview

The Monte Carlo Player uses Monte Carlo Tree Search (MCTS) to choose the next move in Klondike Solitaire.

It maintains a persistent tree for the duration of a game:

- **`root`**: created once at game start (`root == null` triggers initialisation)
- **`current`**: the node representing the current game position; advanced after each move
- **Tree reuse**: statistics under `current` are reused within a turn; the tree is updated each turn

## Architecture

### Single Persistent Game Tree

- Each node stores a cloned `Solitaire` snapshot.
- Children are keyed by the exact move command string returned by `LegalMovesHelper.listLegalMoves()`.

### Stale State Handling (Critical)

Like the A* player, this player must handle the fact that a real move can reveal cards.
That means:

1. At the start of `nextCommand()`, refresh `current.state` with a fresh `solitaire.copy()`.
2. **Invalidate children** of `current` (clear `current.children` and reset its MCTS stats).

Rationale: a card reveal changes the true game state, so previously simulated child states and their rollouts are no longer reliable.

## MCTS Algorithm

Each decision runs a fixed number of iterations:

- **Selection**: walk down the tree choosing children by UCT (UCB for trees)
- **Expansion**: add one child for an unexplored legal move
- **Simulation**: run a short random rollout (bounded length)
- **Backpropagation**: accumulate reward and visit counts back to the root

### UCT Policy

A child is selected by:

$$
UCT = \bar{x} + c \sqrt{\frac{\ln(N)}{n}}
$$

- $\bar{x}$: mean reward of the child
- $N$: parent visits
- $n$: child visits
- $c$: exploration constant (`EXPLORATION_CONSTANT`)

## Move Application

Moves are applied uniformly in one helper to avoid divergent parsing:

- `turn`
- `move <from> <to>` (3 tokens)
- `move <from> <mid> <to>` (4 tokens)

This is important because simulations and expansion must apply moves exactly the same way as expected-state computation.

## Quitting and Failure Modes

- `quit` is excluded from MCTS exploration.
- If no legal moves exist, `nextCommand()` returns `null`.

## Implementation Files

| File | Description |
|------|-------------|
| `MonteCarloPlayer.java` | Main player and MCTS loop, with stale-state invalidation and move application |
| `MonteCarloTreeNode.java` | Tree node with MCTS statistics and heuristic evaluation |

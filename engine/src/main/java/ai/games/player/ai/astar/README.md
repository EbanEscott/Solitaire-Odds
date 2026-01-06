# A* Search Player Design

## Overview

The A* Search Player uses a persistent game tree with true A* search (f = g + h) to find optimal move sequences in Solitaire. The tree persists across all moves in a game, enabling knowledge reuse from previous turns.

## Architecture

### Single Unified Game Tree

- **`root`**: The tree root, created when the game starts (`root == null` triggers initialization)
- **`current`**: The node representing the current game position, advances after each move
- **Tree persistence**: The tree is never rebuilt; explored branches remain available
- **State storage**: Each node stores a `Solitaire` state in PLAN mode (face-down cards as UNKNOWN)

### Stale State Handling

When `nextCommand()` is called, the real game may have revealed cards that were UNKNOWN in our tree:

1. **Refresh `current.state`** with a fresh `solitaire.copy()` from the actual game
2. **Invalidate children**: Clear `current.children` and remove descendants from `openSet` and `bestG`
3. **Re-explore**: A* naturally re-expands children with accurate states

This preserves cycle detection (ancestor chain intact) while ensuring fresh exploration from `current`.

### Search Budget

- **`NODE_BUDGET = 1024`**: Maximum node expansions per `nextCommand()` call
- Configurable constant; can be increased for more thorough search on powerful machines

## A* Search Algorithm

### True A* with f = g + h

- **g(n)**: Path cost from `current` (number of moves taken)
- **h(n)**: Heuristic estimate of remaining cost to win (lower = closer to goal)
- **f(n)**: Total estimated cost = g + h
- **Priority queue**: Nodes expanded in order of lowest f-score

### Heuristic Components (h)

```
h = (52 - foundationCards)      // Base: minimum moves to win
  + (2 × faceDownCards)         // Penalty: must be revealed
  + (0.5 × stockCards)          // Penalty: less accessible
  - (3 × emptyTableauColumns)   // Bonus: strategic flexibility
```

### Probability Weighting for UNKNOWN Cards

In PLAN mode, face-down tableau cards and unturned stock cards appear as UNKNOWN:

- When a move targets an UNKNOWN destination, estimate P(success)
- **Probability** = (compatible cards in unknown pool) / (total unknowns)
- **Adjusted f-score**: `f = g + h/p` (lower probability → higher penalty)

## Pruning Strategies

| Strategy | Description |
|----------|-------------|
| **Quit moves** | Never expanded during search |
| **Ping-pong prevention** | Don't immediately reverse the previous move |
| **Useless king moves** | Skip T→T king moves that don't reveal cards |
| **Duplicate paths** | Skip states already reached via better g-cost (`bestG` map) |
| **Terminal dead-ends** | Skip stuck states that aren't wins |

## Cycle Detection and Quitting

### How Cycles Are Handled

- **Detection**: `isCycleDetected()` checks if the current state matches an ancestor with no progress
- **Pruning**: Cycling branches are marked as `pruned` to prevent re-exploration
- **Example**: Repeatedly turning the stockpile without making other moves creates a cycle

### When the Player Quits

The player returns `"quit"` when the **tree is exhausted**:

- All branches from `current` are either pruned, terminal (stuck), or fully explored
- The `openSet` becomes empty with no winning path found
- This naturally handles unwinnable games: cycles get pruned, dead-ends get skipped, eventually nothing remains

**No separate cycle counter needed** — tree exhaustion is the quit condition.

## Implementation Files

| File | Description |
|------|-------------|
| `AStarPlayer.java` | Main player with `nextCommand()` containing inline A* search loop |
| `AStarTreeNode.java` | Extends `TreeNode` with g, h, f scores and `Comparable<AStarTreeNode>` |
| `AStarPlayerTest.java` | Tests for correctness, termination, and pruning behaviour |

## Key Methods

### `nextCommand(Solitaire solitaire, String moves, String feedback)`

1. **Initialize tree** if `root == null`
2. **Refresh `current.state`** with fresh `planningCopy()`
3. **Invalidate children** of `current` (clear from tree, `openSet`, `bestG`)
4. **Run A* loop** up to `NODE_BUDGET` expansions:
   - Pop lowest-f node from `openSet`
   - If won, trace back to find first move from `current`
   - Generate children via `LegalMovesHelper.listLegalMoves()`
   - Apply pruning rules; mark cycles as pruned
   - Add valid children to `openSet` and `bestG`
5. **Select move**: Best path found, or `"quit"` if `openSet` empty
6. **Advance `current`** to chosen child node

### `applyMove(Solitaire solitaire, String move)`

Single private helper for applying "turn" and "move X Y" commands to cloned states.
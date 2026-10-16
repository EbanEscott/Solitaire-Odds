# Monte Carlo Tree Search Player Design

## Overview

The Monte Carlo player currently uses flat Monte Carlo evaluation with UCB allocation to choose the next move in Klondike Solitaire. It evaluates each immediate candidate with bounded depth-first searches on sampled full-information boards. UCB allocates samples at the root; the internal DFS branches do not retain UCT statistics.

It maintains a persistent tree for the duration of a game:

- **`root`**: created once at game start (`root == null` triggers initialisation)
- **`current`**: the node representing the current game position; advanced after each move
- **Tree reuse**: statistics under `current` are reused within a turn; the tree is updated each turn

## Architecture

### Single Persistent Game Tree

- Each retained candidate stores a planning board, reused across evaluations through apply/rollback. Neither evaluator creates board copies inside its search.
- Legal moves are generated directly as `Move` objects via `LegalMovesHelper.listMoves()` and applied without parsing commands. Children retain command-string keys for compatibility with the shared tree API.

### Stale State Handling (Critical)

Like the A* player, this player must handle the fact that a real move can reveal cards.
That means:

1. At the start of `nextCommand()`, refresh `current.state` with a fresh `solitaire.planningCopy()`.
2. **Invalidate children** of `current` (clear `current.children` and reset its MCTS stats).

Rationale: a card reveal changes the true game state, so previously simulated child states and their rollouts are no longer reliable.

## MCTS Algorithm

Each decision evaluates every immediate legal move except quit:

- **Expansion**: create one child and planning-board copy per candidate.
- **Initial sampling**: sample each candidate once. Exclude candidates returning the rejection sentinel; zero-reward samples remain eligible.
- **Allocation**: share `64 × eligible candidates` evaluations, including initial samples, using UCB. Each evaluation samples a complete hidden arrangement, searches it with DFS, then rolls back all moves and assignments. The default search budget is 128 visits.
- **Statistics**: accumulate eligible evaluation rewards and visits on both the candidate and current decision node. Rejected candidates retain one rejection statistic but do not contribute to the decision node's totals.
- **Selection**: return the candidate with the highest mean reward, breaking exact ties by an immediate win and then the higher existing board progress heuristic.

### UCB Allocation

The player uses the node's UCT helper to select the next candidate to simulate:

$$
UCT = \bar{x} + c \sqrt{\frac{\ln(N)}{n}}
$$

- $\bar{x}$: mean reward of the child
- $N$: parent visits
- $n$: child visits
- $c$: exploration constant, initially `sqrt(2)`

Every eligible candidate receives an initial sample before UCB allocation. Ties during allocation follow legal-move order. The exploration bonus affects simulation allocation only; the final action still maximises mean reward, with exact ties preferring an immediate win and then the higher existing board progress heuristic. The tie-break uses the restored candidate planning board, so it does not use sampled hidden identities or change evaluation rewards. A single eligible candidate retains its 64-rollout allowance for this change. With no eligible candidates, allocation ends immediately.

### Sampled Depth-First Search

`simulateDepthFirst()` samples all UNKNOWN slots through `Solitaire.sampleUnknownCards()` before exploring. Assignments use only the PLAN pool, never the actual hidden deal, and remain fixed throughout backtracking. Face-up counts and movement rules still apply.

The player owns an explicit stack of frames containing remaining moves, incoming move, state key and history depth. It applies a move, explores deeper, then restores the checkpoint before trying a sibling. Search uses legal-generation order and the existing inverse, useless-king and sibling-symmetry filters. A path-local key set rejects cycles; there is no global transposition pruning. Neither the retained candidate node nor its statistics is used as mutable traversal state.

The starting position and every attempted successor, including cycle rejections, count against `montecarlo.search.visits` (default 128). A solved board is recognised on the final permitted visit. A `finally` block restores the candidate's original checkpoint on every exit, including exceptions.

| Outcome | Reward | Meaning |
| --- | --- | --- |
| Won | `1.0` | Found a win in this sampled arrangement. |
| Exhausted search | `0.0` | No winning continuation remained under the search's pruning rules. |
| Visit-limit cutoff | `0.5` | Search remains unresolved; this value is not an estimated win probability. |
| Rejected starting candidate | `-1.0` | Quit, a useless king shuffle, or a repeated starting state. |

Only candidate rejection stops further samples. An eligible candidate whose samples all score zero remains selectable. Solutions assume full information within each sampled world, so the aggregate can be optimistic about decisions that must precede actual reveals.

### Random Rollout Comparator

Set `-Dmontecarlo.evaluator=random` to use the previous `simulateRandom()` policy; the default is `dfs`. Random rollouts sample revealed identities as needed and follow one path of at most 128 accepted moves. Their rewards remain `1.0` for wins, `0.0` for exhausted continuations and the existing normalised progress heuristic in `[0, 1)` for cutoffs. Their cycle check retains the previous two-occurrence threshold. Both policies use the same UCB allocator and mutable candidate boards.

Set `-Dmontecarlo.search.visits=<positive integer>` to tune the DFS visit budget independently. A visit budget bounds explored positions across all branches, rather than moves along a single trajectory.

## Move Application

Moves are applied uniformly in one helper to avoid divergent parsing:

- `turn`
- `move <from> <to>` (3 tokens)
- `move <from> <mid> <to>` (4 tokens)

This is important because simulations and expansion must apply moves exactly the same way as expected-state computation.

## Quitting and Failure Modes

- `quit` is excluded from MCTS exploration.
- If no eligible candidate remains, `nextCommand()` returns `quit`.

## Implementation Files

| File | Description |
|------|-------------|
| `MonteCarloPlayer.java` | Main player and MCTS loop, with stale-state invalidation and move application |
| `MonteCarloTreeNode.java` | Tree node with MCTS statistics and heuristic evaluation |

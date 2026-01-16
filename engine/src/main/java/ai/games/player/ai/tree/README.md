
# Search-tree invalidation vs reuse (future optimisation)

## The problem

In the current implementations we invalidate (cut off/remove) large portions of the search tree as the game progresses. This happens because the tree is built in the presence of hidden information (UNKNOWN cards), and when the real game reveals one of those UNKNOWNs, many previously-generated child states become stale. The current approach is safe and simple, but it throws away potentially valuable work: even if only one card reveal occurred (or no reveal occurred at all), we can end up discarding entire subtrees “just in case”, which increases total nodes expanded and makes both A* and MCTS slower than they need to be.

## Direction of a solution

The optimisation idea is to keep more of the tree and reconcile it with new information.

Instead of blanket invalidation, we want a mechanism where an already-built node/subtree can be “updated” (or proven inconsistent and pruned) when a reveal happens.

Conceptually:

- A reveal is an *observation* arriving from the real game.
- Each node/subtree implicitly encodes assumptions about UNKNOWN cards (either explicitly via guesses, or implicitly via masked state).
- When an observation arrives, we should be able to:
	- keep branches whose assumptions are still consistent with the observation;
	- discard only the inconsistent branches;
	- preserve search statistics/caches where it is still valid to do so.

This likely needs a node-level “reveal reconciliation” hook (something like  a `revealUnknown()`), and corresponding support in node subclasses (e.g. A* nodes and Monte Carlo nodes) so they can refresh any cached fields derived from the stored state.

## What makes this tricky (trade-offs)

This improves performance by reusing work, but it adds complexity and correctness risk:

- We need a reliable way to detect whether a reveal actually happened between two consecutive game states (some moves reveal nothing).
- We need to identify *which* UNKNOWN became known (UNKNOWN-by-location / UNKNOWN-by-slot matters); if UNKNOWN cards are not distinguishable, “match reveal to assumption” becomes ambiguous.
- In PLAN mode, guesses and tracking structures (e.g. `UnknownCardGuess` / `UnknownCardTracker`) likely become part of correctness: a reveal should match one of the guessed possibilities for a branch to remain valid.
- Any cached values inside nodes (A* scores, MCTS visit counts, etc.) may need updating or invalidation when the underlying information changes.
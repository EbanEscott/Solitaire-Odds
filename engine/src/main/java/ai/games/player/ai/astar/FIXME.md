# A* Player Architecture Issues

## Hidden Information Leak in Lookahead Search

### Problem
The A* player's lookahead search applies moves to copies of the game state to evaluate candidate moves and detect cycles. When moves are applied to these copies, face-down cards in the tableau are revealed. This means:

1. The player gains knowledge about which cards will be revealed at future positions in the game
2. The heuristic evaluation is based on game states with perfect information about revealed cards
3. The search guidance (which branches to explore deeper) is influenced by this future information
4. The player makes decisions in the real game based on information it learned during lookahead but shouldn't have

This is a form of **look-ahead cheating** - the player is effectively playing with perfect information during search, even though it only uses aggregate metrics (card counts) rather than inspecting individual card identities.

### Real Solution Needed
We need a mechanism that allows A* to:
1. Search the game tree for tactical decisions
2. Apply moves to understand state transitions and cycle detection
3. **BUT** keep future card reveals hidden/unknown during search evaluation

This likely requires:
- A separation between "search state" (with revealing moves) and "evaluation state" (with hidden information)
- Possibly a "masked" evaluation function that doesn't benefit from revealed cards
- Or a fundamentally different search approach that respects the information barrier

### Impact
This affects both `evaluate()` heuristic guidance and potentially the `isCycleDetected()` logic if it's based on states with revealed cards.

### Priority
**High** - This is a correctness issue, not a performance optimization. The A* player may be winning games it shouldn't be able to win.

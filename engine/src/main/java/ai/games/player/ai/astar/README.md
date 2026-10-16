# A* Search Player

A* keeps the chosen game path between decisions and performs bounded best-first search from the latest player observation. Each decision uses `Solitaire.planningCopy()`, so unseen tableau and stock identities are masked. Previously observed stock cards retain their identities and order.

## Unknown card outcomes

Before expanding legal moves, a node resolves any revealed UNKNOWN slots. `expandReveal()` makes one child per remaining unseen identity for the next slot. Each child calls `Solitaire.assumeCard()`, which assigns the card and excludes it from later assignments on that branch. Sibling boards remain independent.

With three unseen cards, a particular first assignment has probability `1/3`, a second has conditional probability `1/2`, and the last has probability `1`. The complete sequence has probability `1/6`. Chance nodes carry no game command and do not increase the move count. Tracing a chosen path back to the current position returns the actual move that preceded those outcomes.

## Search ranking and limits

- `g`: number of game moves since the current observation.
- `h`: existing heuristic based on foundations, hidden tableau cards, stock size, and empty columns, clamped to zero.
- `p`: cumulative probability of the speculative assignments on the path.
- `f = (g + h + 1) / p - 1`: ranking score; lower is preferred. With no assumptions, this reduces to `g + h`. The extra one keeps uncertainty penalised at zero-heuristic positions and goals.

Duplicate positions retain the best ranking score. The search still allows at most 1,024 node expansions and 5,000 open nodes. Chance branches consume this budget too; drawing three cards from a large unseen pool can consume much of it. This is a search heuristic, not a full expected-win calculation or an optimality guarantee.

## Refresh after actual play

Each decision refreshes `current` from `planningCopy()`, resets its move cost and probability, and discards its speculative descendants. Real reveals therefore replace prior assumptions. Known moves continue through the shared tree pruning and board movement code.

`PlanningSearchTest` covers cumulative probabilities, sibling isolation, entry through planning copies, and refresh after an actual reveal. Historical win-rate and timing results have not been rerun for this behaviour.

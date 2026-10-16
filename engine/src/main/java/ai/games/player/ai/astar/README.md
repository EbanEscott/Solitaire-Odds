# A* Search Player

A* keeps the actual chosen path between decisions and performs bounded best-first search from the latest observation. `Solitaire.planningCopy()` masks unseen tableau and stock identities while retaining previously observed stock cards. Search policy stays in `AStarPlayer`; `AStarTreeNode` holds board, scoring, and expansion data.

## Known cards before reveals

Known positions are ordered by `f = g + h`, with lower `h` breaking ties. Newly exposed UNKNOWN slots stop a branch and enter a separate deferred queue. The initial phase spends up to 512 expansions on known cards and can retain at most 2,500 nodes, reserving capacity for reveals.

The remaining budget starts with a deferred reveal, then alternates up to 16 known-position expansions with another reveal. Each decision allows at most 1,024 expansions and 5,000 retained nodes, including pruned children. Expansions that cannot fit a complete successor group leave the node as a heuristic leaf. They do not discard the last few outcomes in card-enumeration order.

## Unknown outcomes and probabilities

A reveal event assigns every currently exposed UNKNOWN slot before continuing game moves. With three unseen cards, there are six ordered outcomes: successive assignments have conditional probabilities `1/3`, `1/2`, and `1`, so each complete outcome has probability `1/6`. Assignment uses `Solitaire.assumeCard()` without replacement and does not increase `g`.

Events with at most eight joint outcomes are enumerated exactly. Larger events use eight independent, uniform joint samples. Cards cannot repeat within an outcome, but independent samples may produce the same outcome. Samples use a random generator seeded from the planning state key, never hidden identities, so a repeated observed position receives repeatable samples.

Each outcome stores two separate quantities:

- Cumulative probability of its exact assignment sequence, including earlier reveals.
- Conditional weight for averaging this reveal: exact probability for enumeration, or `1/8` for a sampled event.

Equal sample weights approximate the full outcome distribution. Multiplying them by the exact sequence probability again would incorrectly discount the event a second time. Eight samples can miss rare outcomes; this is a bounded approximation.

## Choosing a move

Values are backed up through the tree:

- At a decision, choose the continuation with the lowest estimated cost.
- At a reveal, average all retained outcome values using their conditional weights.
- At an unexpanded leaf, use `g + h`; unexplored outcomes retain an estimate.
- At a win, use `g`.
- At an exhausted non-winning decision, use a finite failure cost of `1000 + g`, so one bad outcome does not automatically veto a reveal.

Only a complete root action's backed-up value determines selection. Finding a winning path in one speculative outcome does not establish that the action wins. Values are heuristic costs, not calibrated win probabilities or an optimality guarantee. The player quits when no viable root action remains or all estimated costs reach the failure threshold.

Shared inverse, king, cycle and sibling pruning still applies to action branches. The former global transposition pruning is removed: dropping a position reached in another chance branch would remove that outcome's contribution without backing up its value. The retained-node bound limits the additional work.

## Refresh and verification

After a real move, the current node is refreshed from the new observation. Its speculative descendants are discarded, move cost resets to zero, and probability resets to one. Prior actual positions remain as ancestors for pruning.

`PlanningSearchTest` covers exact probabilities, complete stock samples, deferred expansion, expected-value selection, heuristic fallback when an outcome group cannot fit, and observation refresh. Current result samples are recorded separately from historical full-information results in the root README.

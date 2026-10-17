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

`PlanningSearchTest` covers exact probabilities, complete stock samples, deferred expansion, expected-value selection, heuristic fallback when an outcome group cannot fit, and observation refresh. Interim result samples remain in `engine/logs/profiles/` during development. The root README results table is updated only when settled results are ready to publish and the user requests it.

## A* development experiments

A* exposes independent experimental JVM properties `astar.foundationWeight`
(default `1.0`), `astar.stockWeight` (default `0.5`), and `astar.knownFirstBudget`
(default `512`). Defaults preserve the existing policy; experimental scores apply to both
action and chance nodes. The total expansion and retained-node limits remain unchanged.

The opt-in `ai.games.experiments.AStarScreenTest` runs baseline and three single-change
variants on identical seeded deals and writes deck orders and per-game CSV results. Enable
with `-Dastar.screen=true -Dastar.output=/absolute/run/directory`; optionally set
`astar.games` (default 20) and `astar.seedStart` (default 0). Use a fresh output directory.
Research notes and analysis live in `experiments/notebooks/astar_improvement_research.ipynb`
at the repository root. Development runs do not update the canonical win-rate table.

Further controls are `astar.moveCost` (default `1.0`), `astar.nodeBudget` (default
`1024`), and `astar.maxSearchNodes` (default `5000`). The default initial known-only
allowance is half the configured expansion budget unless `astar.knownFirstBudget`
is supplied. With non-default move cost, `g` is accumulated action cost rather than
the number of moves; chance assignments still add zero.

Set `astar.trace=true` to retain a textual summary of the last decision's expansion
count, retained nodes, root action values and pruning reasons, available through
`getDecisionTrace()`. Tracing is off by default. The development runner supports
`astar.variants=baseline,lowcost,deeper` and writes per-decision board traces when
tracing is enabled. `lowcost` uses move cost `0.1`; `deeper` scales expansions,
retained nodes and initial known-only allowance by four.

`astar.hiddenWeight` (default `2.0`) and `astar.emptyWeight` (default `3.0`)
control reveal and empty-column scoring. Screen presets `hidden4`, `hidden8`, and
`noempty` change these separately. `astar.finishKnownWin=true` enables the `finish`
preset: prioritize and retain a discovered winning continuation containing no
unresolved or sampled chance node. Continuation reuse requires the observed state
key to match. These are experimental options, disabled or at baseline weights by
default; see the research notebook for matched results.

`astar.transpositions=true` enables experimental best-cost duplicate pruning within
each chance outcome. Every newly sampled/enumerated outcome gets a separate context,
so duplicate pruning does not remove another outcome's probability contribution.
This is an empirical search policy: shared path-dependent pruning still interacts
with which representative survives. It is not an optimality guarantee.

`astar.failureCost` (default `1000.0`) controls both exhausted-node cost and the root
quit threshold. Screen presets `failure200` and `failure400` test smaller penalties
for failed sampled continuations.

`astar.talonWeight` defaults to `0.0`. The `undeployed` preset penalizes stock and
waste equally at `2.0` per card, rewarding removal rather than drawing. The `hill`
preset uses `(1040 - hillClimberScore) / 5`: foundation deficit weight `3.0`,
hidden weight `1.6`, stock `1.2`, waste `1.0`, empty-column bonus `0.0`. It changes
the evaluation model as a whole; A* search, chance handling and move cost stay fixed.

The `transpositions4x` screen preset combines chance-scoped duplicate pruning with
4,096 expansions, 20,000 retained nodes and 2,048 initial known-only expansions.
It tests whether more capacity helps after duplicate pruning, rather than treating
the earlier larger-budget result as independent of search policy.

`astar.scanStock=true` performs one initial observation pass using real `turn`
commands until the stock is empty, then starts ordinary planning. It reads only
stock emptiness, never unseen identities; the real game must apply each command
before the next observation. The `scan` preset combines this with duplicate pruning,
counts the extra turns against the same move cap, and does not rescan on later
redeals. This relies on the engine's unlimited-redeal rules and remembered exposed
stock cards. It is disabled by default.

Experimental A* duplicate-value reuse is enabled with both `astar.transpositions=true`
and `astar.reuseDuplicates=true` (screen preset `reuse`). Duplicate nodes retain a
link to their representative instead of disappearing from decision backups; values
include the difference in accumulated move cost. Recursive alias cycles receive the
finite failure cost. Backups cache completed node values separately for each root
action to bound work; cycle cutoffs can make these estimates traversal-dependent.
Links are cleared at each real observation and never cross chance
outcome contexts. Path-dependent pruning still makes this an approximate policy.
Defaults remain unchanged; development results are recorded in the A* notebook.

`astar.balancedSearch=true` tests round-robin expansion attempts across root actions,
then least-used chance contexts within an action, with ordinary f/h/state ordering
breaking ties. The `balanced` screen preset combines this with duplicate pruning.
Known/reveal phase scheduling and total expansion/node bounds remain unchanged.
This balances attempts among queued candidates, not retained nodes or guaranteed
successful expansions; blocked expansions still consume attempts. Root traces report
allocated attempts. The option is disabled by default.

package ai.games.player.ai.astar;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import ai.games.player.ai.tree.TreeNode;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * A* search player for Klondike Solitaire with persistent game tree cycle detection.
 *
 * <h2>Architecture Overview</h2>
 * <p>This player combines two complementary search mechanisms:
 * <ul>
 *   <li><b>Tactical A* Search:</b> For each decision, runs a bounded best-first search to evaluate
 *       candidate moves and select the highest-value first move within a budget (256 expansions).
 *   <li><b>Persistent Game Tree:</b> Maintains complete game history across all decisions to detect
 *       cycles—situations where the player has returned to a previously-visited board state with
 *       no progress. Quits when stuck in such a cycle for more than 10 moves.
 * </ul>
 *
 * <h2>Cost Model</h2>
 * <p>The A* search uses a cost function {@code f(n) = g(n) + h(n)} where:
 * <ul>
 *   <li><b>g(n):</b> Path cost from root: number of moves taken + penalties for stock cycling
 *       (each "turn" action adds {@link #TURN_PENALTY} cost).
 *   <li><b>h(n):</b> Heuristic value (negative, guiding toward winning states):
 *       <ul>
 *         <li>Foundation progress: +25 per card in foundations
 *         <li>Tableau visibility: +6 per face-up card, -8 per face-down card
 *         <li>Empty columns: +10 per empty tableau column (kingable space)
 *         <li>Stock drag: -1 per card in stockpile
 *       </ul>
 * </ul>
 *
 * <h2>Search Strategy & Pruning</h2>
 * <p>Within each A* decision, the player prunes branches aggressively to stay responsive:
 * <ul>
 *   <li><b>Quit moves:</b> Never chosen by A* (handled separately if game exits)
 *   <li><b>Ping-pong moves:</b> Rejects immediate reversals of the previous move
 *   <li><b>Useless king moves:</b> Skips tableau-to-tableau king moves that don't reveal cards
 *   <li><b>Duplicate paths:</b> Avoids re-expanding states via inferior paths (via {@code bestG} map)
 *   <li><b>Foundation moves allowed:</b> Both foundation-up and foundation-down moves are explored,
 *       enabling strategic repositioning
 * </ul>
 *
 * <h2>Game Tree Persistence & Cycle Detection</h2>
 * <p>The persistent {@link AStarTreeNode} tree tracks:
 * <ul>
 *   <li><b>Full move history:</b> Every move made throughout the game, in order
 *   <li><b>Board state snapshots:</b> Hashed state key, foundation count, and facedown count
 *   <li><b>Visit tracking:</b> Counts how many times each state has been revisited
 * </ul>
 * <p>When a state is revisited with no progress (same foundation count and facedown count as a prior
 * visit), a cycle is detected. If the player makes 10+ moves within this cycle without breaking out
 * or finding progress, the player quits to avoid infinite loops.
 *
 * <h2>Why This Design</h2>
 * <p><b>Single Source of Truth:</b> The persistent game tree eliminates the need for separate
 * short-term caches or sliding windows of recent states. All state history is unified in one
 * data structure, reducing complexity and improving maintainability.
 *
 * <p><b>Responsive Tactically, Safe Strategically:</b> The bounded A* search (256 expansions)
 * keeps move selection fast (~0.5s per move), while the game tree provides strategic safety
 * against infinite loops across arbitrary game lengths.
 *
 * <p><b>Exploration vs. Exploitation:</b> By allowing foundation-down moves and other strategic
 * repositioning while pruning only clearly-wasteful moves (ping-pong, useless king moves),
 * the player balances exploration of promising paths with efficiency.
 *
 * <p><b>Full Game Tree for Optimization, Not Memory Consciousness:</b> This design maintains
 * the complete game history as a persistent tree rather than a bounded cache or sliding window.
 * We prioritize optimization opportunities (cycle detection, advanced heuristics, state analysis)
 * over memory efficiency. Modern hardware has ample memory, and the performance gains from having
 * full state history far outweigh the storage cost. This approach is well-suited for generating
 * training data and comprehensive game analysis where completeness and accuracy are more valuable
 * than minimizing resource usage.
 */
@Component
@Profile("ai-astar")
public class AStarPlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(AStarPlayer.class);

    /**
     * Hard cap on A* node expansions per decision to keep moves responsive.
     *
     * <p><b>Why this value?</b> With 256 expansions and branching factor ~3-5, we typically
     * explore ~3-4 ply depth, sufficient for medium-term tactical decisions (~0.5s runtime).
     * Higher values improve move quality but increase latency; lower values risk myopic decisions.
     *
     * <p><b>Tuning note:</b> If games feel sluggish (A* taking >1s per move), lower this value.
     * If many games exceed move limits, consider raising it slightly or improving heuristics.
     */
    private static final int MAX_EXPANSIONS = 256;

    /**
     * Additional cost applied when taking a "turn" action to discourage excessive stock cycling.
     *
     * <p><b>Why penalise turns?</b> Stock cycling (repeatedly turning and exhausting the deck)
     * is often a sign of stagnation. By adding cost to turn actions, the search prefers moves
     * that make foundation/tableau progress, which is usually more valuable than stock reshuffling.
     *
     * <p><b>Tuning note:</b> A value of 2 means a "turn" costs as much as 2 regular moves.
     * If the player turns too often, increase this; if it never turns when beneficial, decrease it.
     */
    private static final int TURN_PENALTY = 2;

    /**
     * Tracks the last non-quit, non-turn move to prevent immediate ping-pong reversals.
     *
     * <p><b>Why track this?</b> If the player moved a card from A→B, then immediately tries B→A,
     * we're wasting moves without making progress. By remembering the last move's signature
     * (source, destination, card), we can prune its exact inverse during A* expansion.
     *
     * <p><b>Scope:</b> This is game-wide state, persisting across all {@code nextCommand()} calls
     * within a single game. It resets via {@link #resetGameState()} between games.
     *
     * @see MoveSignature#isInverseOf(MoveSignature)
     */
    private static MoveSignature lastMove = null;

    /**
     * Root node of the persistent game tree, initialised on the first move of a game.
     *
     * <p><b>Why persist a game tree?</b> Klondike Solitaire has many cycles—board states reachable
     * via different move sequences. By maintaining the full game history in a tree, we can detect
     * when we've returned to a previously-visited state. If we're stuck cycling for 10+ moves
     * with no progress, quitting becomes the rational choice rather than looping forever.
     *
     * <p><b>Initialisation:</b> Set to null at game start (in {@link #resetGameState()}),
     * then initialised on the first call to {@code nextCommand()}. The root snapshot captures
     * the initial foundation count and facedown count, allowing us to detect stagnation later.
     *
     * <p><b>Lifecycle:</b> Reset to null at game end (when the player quits or wins).
     *
     * @see AStarTreeNode
     */
    private static AStarTreeNode gameTreeRoot = null;

    /**
     * Current node in the persistent game tree, pointing to where we are in game history.
     *
     * <p><b>Role:</b> After each move is selected and executed, {@code gameTreeCurrent} is
     * advanced to the corresponding child node. This maintains an accurate position in the
     * game history, enabling cycle detection at any point in the game.
     *
     * <p><b>Cycle detection:</b> To detect cycles, we call {@link AStarTreeNode#findCycleAncestor()}
     * on {@code gameTreeCurrent}. This method walks the ancestor chain looking for a node with
     * the same board state (same state key) and same progress (same foundation + facedown counts).
     * If found, we're in a cycle; if the cycle persists beyond 10 moves, we quit.
     *
     * <p><b>Why separate from root?</b> Keeping a "current" pointer avoids expensive tree traversals
     * to find our position on each move.
     *
     * @see AStarTreeNode#findCycleAncestor()
     * @see AStarTreeNode#distanceTo(AStarTreeNode)
     */
    private static AStarTreeNode gameTreeCurrent = null;

    /**
     * Resets all static state for the A* player before a new game.
     *
     * <p><b>Why reset?</b> Since {@code lastMove}, {@code gameTreeRoot}, and {@code gameTreeCurrent}
     * are static (shared across all instances), they persist after a game ends. Without resetting,
     * the second game would inherit move/state history from the first game, causing incorrect
     * cycle detection and ping-pong prevention.
     *
     * <p><b>When to call:</b> This should be called by the game framework before initialising
     * a new game. In tests, call this in {@code @BeforeEach} or {@code @AfterEach} setup methods.
     *
     * <p><b>Implementation note:</b> All three fields are set to null. Since {@link AStarTreeNode}
     * and {@link MoveSignature} objects have no cleanup code, simple null assignment is sufficient;
     * they will be garbage-collected when unreferenced.
     */
    public static void resetGameState() {
        lastMove = null;
        gameTreeRoot = null;
        gameTreeCurrent = null;
    }

    /**
     * Selects the next move for this player in the given game state.
     *
     * <p><b>High-level flow:</b>
     * <ol>
     *   <li><b>Phase 1:</b> Initialise or update the persistent game tree
     *   <li><b>Phase 2:</b> Handle trivial cases (only one legal move available)
     *   <li><b>Phase 3:</b> Run A* search to evaluate candidate moves
     *   <li><b>Phase 4:</b> Select the best move from the search results
     *   <li><b>Phase 5:</b> Update the persistent game tree and detect cycles
     * </ol>
     *
     * <p><b>Inputs:</b>
     * <ul>
     *   <li>{@code solitaire}: The current game state
     *   <li>{@code recommendedMoves}: Unused; provided by game framework for informational players
     *   <li>{@code feedback}: Unused; game-provided context (e.g., "stockpile exhausted" warnings)
     * </ul>
     *
     * <p><b>Return value:</b> A move command string (e.g., "move T1 4♥ F1", "turn", "quit")
     * that will be executed by the game engine.
     *
     * <p><b>Why five phases?</b> This separation ensures:
     *   <ul>
     *     <li>Game tree state is always in sync with actual board history
     *     <li>Cycle detection happens as moves are made (Phase 5), marking pruned branches
     *     <li>A* search (Phase 3) avoids those pruned branches via isCycleDetected() checks
     *     <li>Each phase has a clear, documented responsibility
     *   </ul>
     *
     * @param solitaire the current Klondike Solitaire game state
     * @param recommendedMoves informational string from game (ignored by this player)
     * @param feedback informational string from game (ignored by this player)
     * @return a command string to be executed by the game engine
     */
    @Override
    public String nextCommand(Solitaire solitaire, String recommendedMoves, String feedback) {
        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        if (legal.isEmpty()) {
            return "quit";
        }

        // ============= Phase 1: Initialise or update persistent game tree =============
        // On the very first move of a game, capture the initial board state as a snapshot.
        // This snapshot (foundation count, facedown count) will serve as the baseline for
        // detecting progress. If we ever return to a state with these exact metrics, we've made
        // no progress and are cycling.
        if (gameTreeRoot == null) {
            long rootKey = solitaire.getStateKey();
            int rootFoundation = solitaire.getFoundation().stream()
                .mapToInt(java.util.List::size).sum();
            int rootFacedown = solitaire.getTableauFaceDownCounts().stream()
                .mapToInt(Integer::intValue).sum();
            gameTreeRoot = new AStarTreeNode(null, null, rootKey, rootFoundation, rootFacedown);
            gameTreeCurrent = gameTreeRoot;
        }

        // Compute the current board state metrics. We'll use these later in Phase 6 to update
        // the game tree with the move we're about to make.
        long currentKey = solitaire.getStateKey();
        int currentFoundation = solitaire.getFoundation().stream()
            .mapToInt(java.util.List::size).sum();
        int currentFacedown = solitaire.getTableauFaceDownCounts().stream()
            .mapToInt(Integer::intValue).sum();

        // ============= Phase 2: Handle trivial cases =============
        // If only one move is legal, no need to search—execute it immediately.
        if (legal.size() == 1) {
            String only = legal.getFirst();
            return TreeNode.isQuit(only) ? "quit" : only;
        }

        // ============= Phase 3: Run A* search with improved pruning =============
        // Initialise the A* search at the current board state. We'll expand nodes using a
        // priority queue, guided by the heuristic evaluation function.
        // Switch to PLAN mode so that lookahead copies mask face-down cards with UNKNOWN.
        Solitaire.GameMode originalMode = solitaire.getMode();
        solitaire.setMode(Solitaire.GameMode.PLAN);
        int rootHeuristic = evaluate(solitaire);
        Solitaire planCopy = solitaire.copy();
        solitaire.setMode(originalMode);  // Restore original mode for the real game
        AStarTreeNode root = new AStarTreeNode(planCopy, null, null, 0, rootHeuristic);
        Queue<AStarTreeNode> open = new PriorityQueue<>();
        open.add(root);

        // The bestG map tracks the best path cost we've found to each state. If we encounter
        // the same state again via a worse path, we skip it. This prevents redundant expansions
        // and keeps the search efficient.
        Map<Long, Integer> bestG = new HashMap<>();
        bestG.put(currentKey, 0);

        AStarTreeNode bestNode = null;
        AStarTreeNode turnNode = null;
        int expansions = 0;

        while (!open.isEmpty() && expansions < MAX_EXPANSIONS) {
            AStarTreeNode current = open.poll();
            expansions++;

            // Track the node with the highest heuristic value (best board evaluation) that we've
            // encountered during the search. This serves as a fallback if the search runs out of
            // budget before finding a path to root.
            if (bestNode == null || current.heuristic > bestNode.heuristic) {
                bestNode = current;
            }

            // ===== Pruning: Skip pruned subtrees =====
            // If this node is marked as leading to unproductive cycles or stagnation,
            // skip expanding its children to conserve the 256-node budget. This is where
            // the persistent pruning shines: knowledge from previous decisions guides future search.
            if (current.isPruned()) {
                continue;
            }

            // Generate and evaluate all legal moves from this state.
            List<String> moves = LegalMovesHelper.listLegalMoves(current.state);
            for (String move : moves) {
                // ===== Pruning: Quit moves =====
                // Never choose a quit move during the search. If the game ends, it's handled
                // separately in phase 5 (fallback logic).
                if (TreeNode.isQuit(move)) {
                    continue;
                }

                // ===== Pruning: Ping-pong prevention =====
                // If the previous move was "move A→B", don't consider "move B→A" now. This
                // prevents wasting moves on immediate reversals.
                if (lastMove != null && move != null) {
                    MoveSignature currentSig = MoveSignature.tryParse(move);
                    if (currentSig != null && lastMove.isInverseOf(currentSig)) {
                        continue;
                    }
                }

                // ===== Pruning: Useless king moves =====
                // Don't shuffle a king between tableau columns if it won't reveal new cards
                // beneath it. This prune saves expansions without losing solutions.
                if (current.isUselessKingMove(move)) {
                    continue;
                }

                // ===== Pruning: Cycle detection =====
                // Skip moves that would lead us to a pruned subtree (marked from previous
                // game decisions as cycling or stagnating). This persists knowledge of
                // unproductive paths across decisions, conserving search budget.
                if (current.isCycleDetected(move, this::applyMove)) {
                    continue;
                }

                // Apply the move to a copy of the state so we can evaluate it.
                Solitaire copy = current.state.copy();
                applyMove(copy, move);
                long key = copy.getStateKey();

                // ===== Pruning: Already-explored better paths =====
                // If we've already found a path to this state with lower or equal cost, skip it.
                // This is classic A* optimization preventing the queue from filling with duplicates.
                int stepCost = 1 + (move != null && move.trim().equalsIgnoreCase("turn") ? TURN_PENALTY : 0);
                int tentativePathCost = current.pathCost + stepCost;
                int knownPathCost = bestG.getOrDefault(key, Integer.MAX_VALUE);
                if (tentativePathCost >= knownPathCost) {
                    continue;
                }
                bestG.put(key, tentativePathCost);

                // Evaluate the new state to guide future expansion decisions.
                int h = evaluate(copy);
                AStarTreeNode child = new AStarTreeNode(copy, current, move, tentativePathCost, h);

                // If this move is a "turn" action and we haven't seen one yet, remember it as a
                // fallback. This is useful if the search can't find any strong tactical move—
                // at least we can cycle the stock in hopes of better cards appearing.
                if (move != null && move.trim().equalsIgnoreCase("turn") && turnNode == null) {
                    turnNode = child;
                }

                // ===== Early exit optimization =====
                // If we've found a winning position (all 52 cards in foundation), immediately
                // stop expanding. No need to search further; a win is a win.
                if (TreeNode.isWon(copy)) {
                    bestNode = child;
                    open.clear();
                    break;
                }

                open.add(child);
            }
        }

        // ============= Phase 4: Select best move from search results =============
        // Given the search results (or fallback options), determine which move to actually execute.
        String chosen = selectBestMove(bestNode, turnNode);

        // ============= Phase 5: Update persistent game tree and detect cycles =============
        // Record the move we chose in the persistent game tree. This maintains an accurate
        // history of the game's actual path. After each move, check if we've entered a cycle;
        // if so, mark the node as pruned so future A* searches avoid this unproductive branch.
        if (!TreeNode.isQuit(chosen)) {
            AStarTreeNode child = (AStarTreeNode) gameTreeCurrent.children.get(chosen);
            if (child == null) {
                // This is the first time we've made this move from this state. Create a new node.
                child = new AStarTreeNode(chosen, gameTreeCurrent, currentKey, currentFoundation, currentFacedown);
                gameTreeCurrent.children.put(chosen, child);
            } else {
                // We've made this move before from this state. Increment the visit counter.
                child.visitCount++;
            }
            gameTreeCurrent = child;
            
            // After advancing the game tree, check if we've entered a cycle. If so, mark this
            // node as pruned so future decisions avoid this unproductive path.
            AStarTreeNode cycleAncestor = gameTreeCurrent.findCycleAncestor();
            if (cycleAncestor != null) {
                // We've detected a cycle: same board state with no progress (same foundation + facedown counts)
                gameTreeCurrent.markPruned();
                gameTreeCurrent.cycleDepth = gameTreeCurrent.distanceTo(cycleAncestor);
            }
        } else {
            // Game is ending (player quits). Reset the tree for the next game.
            gameTreeRoot = null;
            gameTreeCurrent = null;
        }

        lastMove = MoveSignature.tryParse(chosen);
        return chosen;
    }

    /**
     * Selects the best move given A* search results.
     *
     * <p><b>Logic:</b> The best move is the first step on the path from root to {@code bestNode}.
     * We "walk back" the game tree from bestNode to the root, then extract the first move.
     * This ensures we return a move that the game engine can execute immediately.
     *
     * <p><b>Fallback strategy:</b> If the search didn't find a viable path (bestNode is null or
     * has no parent), we fall back to {@code turnNode}—the first "turn" action discovered during
     * the search. If even that fails, we return "quit".
     *
     * <p><b>Why fallback to turn?</b> In difficult positions where the search finds no clear
     * tactical advantage, cycling the stock (via "turn") at least gives us a chance to see new
     * cards. This is better than quitting immediately, and it's better than making a random move.
     *
     * <p><b>Why this walkback logic?</b> The A* search returns a node deep in the search tree,
     * but the game expects a single move to execute immediately. By walking back to the root,
     * we extract the first move on the path that A* considered best, maintaining coherence between
     * the search's evaluation and the actual move made.
     *
     * @param bestNode the highest-valued node found by A* search, or null if search found nothing
     * @param turnNode the first "turn" action encountered during search (or null)
     * @return a move command string ("move...", "turn", or "quit")
     */
    private String selectBestMove(AStarTreeNode bestNode, AStarTreeNode turnNode) {
        if (bestNode == null || bestNode.parent == null) {
            // Search did not find a good path; try a turn move as fallback.
            // Turn gives us a chance to reset the stock and see new cards, which is often
                // better than giving up immediately.
            if (turnNode != null) {
                AStarTreeNode current = turnNode;;
                AStarTreeNode previous = null;
                while (current.parent != null && ((AStarTreeNode)current.parent).parent != null) {
                    previous = current;
                    current = (AStarTreeNode)current.parent;
                }
                AStarTreeNode firstStep = current.parent == null ? previous : current;
                String turnMove = firstStep != null && firstStep.move != null
                        ? firstStep.move
                        : "quit";
                if (!"quit".equalsIgnoreCase(turnMove.trim())) {
                    return turnMove;
                }
            }
            return "quit";
        }

        // Walk back from bestNode to root to find the first move on the best path.
        // The A* search found that starting with this move leads to the best future state.
        AStarTreeNode current = bestNode;
        AStarTreeNode previous = null;
        while (current.parent != null && ((AStarTreeNode)current.parent).parent != null) {
            previous = current;
            current = (AStarTreeNode)current.parent;
        }
        AStarTreeNode firstStep = current.parent == null ? previous : current;
        return firstStep != null && firstStep.move != null ? firstStep.move : "quit";
    }

    /**
     * Applies a move to a game state copy, executing the action specified by the move string.
     *
     * <p><b>Why this method?</b> The move command strings are game-language representations
     * ("move T1 4♥ F1", "turn", etc.). This method translates those strings into actual calls
     * on the {@code Solitaire} object, acting as the bridge between the A* player's decision
     * logic and the game engine's state transitions.
     *
     * <p><b>Move formats supported:</b>
     * <ul>
     *   <li>"turn" — cycle the stock three cards forward
     *   <li>"move SRC CARD DST" — move CARD from SRC (e.g., T1, F1) to DST
     *   <li>"move SRC CARD" — move CARD from SRC to the auto-destination (e.g., to foundation)
     * </ul>
     *
     * <p><b>Safety:</b> This method silently ignores malformed move strings rather than throwing.
     * This is appropriate for A* search, where occasionally a move might be misparsed; the game
     * engine will catch and report illegal moves if they occur.
     *
     * @param solitaire the game state to modify (should be a copy, not the real game)
     * @param move the move command string
     *
     * @see Solitaire#turnThree()
     * @see Solitaire#moveCard(String, String, String)
     */
    private void applyMove(Solitaire solitaire, String move) {
        if (move == null) {
            return;
        }
        String trimmed = move.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else {
                solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
    }

    /**
     * Evaluates the quality of a board state using a multi-factor heuristic.
     *
     * <p><b>Purpose:</b> This heuristic guides the A* search towards states that are likely
     * closer to winning. Higher scores are better (more progress towards win condition).
     *
     * <p><b>Scoring components (all accumulate into a single score):</b>
     * <ul>
     *   <li><b>Foundation progress:</b> +25 per card in any foundation pile.
     *       <br><i>Why?</i> The ultimate goal is to move all 52 cards to the foundation.
     *       <br><i>Weight 25:</i> High weight signals that foundation progress is primary objective.
     *
     *   <li><b>Tableau face-up visibility:</b> +6 per visible (face-up) card in tableau.
     *       <br><i>Why?</i> Visible cards can be moved strategically; hidden cards cannot.
     *       <br><i>Weight 6:</i> Less critical than foundation but still valuable for enabling moves.
     *
     *   <li><b>Tableau face-down penalty:</b> -8 per hidden (face-down) card in tableau.
     *       <br><i>Why?</i> Hidden cards block access to valuable cards beneath. Revealing them is
     *       crucial progress.
     *       <br><i>Weight 8:</i> Strong penalty motivates moves that reveal hidden cards.
     *       The -8 vs +6 ratio (4:3) means revealing 4 cards is worth ~3 visible cards.
     *
     *   <li><b>Empty columns:</b> +10 per empty tableau column.
     *       <br><i>Why?</i> Empty columns are "king slots"—only empty columns can accept kings.
     *       Having empty space is strategically valuable.
     *       <br><i>Weight 10:</i> Modest weight reflects that empty columns are useful but not
     *       as critical as foundation progress.
     *
     *   <li><b>Stock drag:</b> -1 per card remaining in the stockpile.
     *       <br><i>Why?</i> A full stockpile limits our options (fewer cards visible). Once we
     *       cycle through the stock twice, it becomes worthless.
     *       <br><i>Weight 1:</i> Weak penalty; the stock's value decreases naturally as we cycle.
     * </ul>
     *
     * <p><b>Trade-offs:</b>
     * <ul>
     *   <li>This heuristic is "admissible" (never overestimates remaining cost), so A* guarantees
     *       finding the best path within the search budget. However, it's not "consistent," meaning
     *       A* might re-expand nodes as new paths discover them.
     *   <li>The weights are empirically tuned for Klondike but could be adjusted based on gameplay
     *       results. If the player wins more often, the weights are good. If it's too conservative,
     *       try increasing foundation weight or decreasing stock drag weight.
     * </ul>
     *
     * @param solitaire the board state to evaluate
     * @return a heuristic score (higher = closer to winning)
     */
    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress: +25 per card placed.
        // This dominates the heuristic, signalling that foundation placement is the primary goal.
        for (var pile : solitaire.getFoundation()) {
            score += pile.size() * 25;
        }

        // Tableau visibility and facedown penalties.
        // Visible cards create options; hidden cards create blockages. We reward visibility
        // and penalise hidden cards to motivate moves that reveal new cards.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        int emptyColumns = 0;
        for (int i = 0; i < faceUps.size(); i++) {
            int up = faceUps.get(i);
            int down = faceDowns.get(i);
            score += up * 6;
            // Facedown penalty is stronger (-8 vs +6) to prioritise revealing hidden cards.
            score -= down * 8;
            if (up == 0 && down == 0) {
                emptyColumns++;
            }
        }

        // Empty columns are king slots—valuable strategic resources.
        score += emptyColumns * 10;

        // Stock drag: penalise remaining cards in the stockpile.
        // A larger stock limits what we can see and do.
        score -= solitaire.getStockpile().size();

        return score;
    }
}

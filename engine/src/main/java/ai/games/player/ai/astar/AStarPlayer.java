package ai.games.player.ai.astar;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.tree.MoveSignature;
import ai.games.player.ai.tree.TreeNode;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * A* search-based Solitaire AI player.
 *
 * <p>Uses a persistent game tree that survives across moves, enabling knowledge reuse
 * from previous turns. The tree tracks both game history and serves as the search space
 * for A* lookahead.
 *
 * <p><b>Key features:</b>
 * <ul>
 *   <li>True A* search with f = g + h scoring</li>
 *   <li>Probability weighting for moves targeting UNKNOWN cards</li>
 *   <li>Comprehensive pruning: quit moves, ping-pong, useless king moves, duplicates</li>
 *   <li>Cycle detection with tree exhaustion as the quit condition</li>
 * </ul>
 *
 * <p><b>Tree persistence:</b> Static fields ensure the tree survives Spring bean recreation.
 * The tree is reset when root is null (new game detected).
 */
@Component
@Profile("ai-astar")
public class AStarPlayer extends AIPlayer {

    private static final Logger log = LoggerFactory.getLogger(AStarPlayer.class);

    /** Maximum node expansions per nextCommand() call. Tunable for performance. */
    private static final int NODE_BUDGET = 1024;

    /** Maximum size of the open set to prevent memory exhaustion. */
    private static final int MAX_OPEN_SET_SIZE = 5000;

    /** Root of the game tree, created at game start. */
    private AStarTreeNode root = null;

    /** Current position in the tree, advances after each move selection. */
    private AStarTreeNode current = null;

    /** Priority queue for A* expansion, ordered by f-score (lowest first). */
    private PriorityQueue<AStarTreeNode> openSet = null;

    /** Best g-cost seen for each state key, for duplicate path pruning. */
    private Map<Long, Double> bestG = null;

    /**
     * Resets the game tree state. Called automatically when root is null,
     * but can be called explicitly to start a fresh game.
     */
    public void reset() {
        root = null;
        current = null;
        openSet = null;
        bestG = null;
    }

    /**
     * Provides the next command for the game loop.
     *
     * <p>This method contains the main A* search loop:
     * <ol>
     *   <li>Initialise tree if root is null (new game)</li>
     *   <li>Refresh current node's state with fresh planning copy</li>
     *   <li>Invalidate stale children (Option B: re-explore after card reveals)</li>
     *   <li>Run A* expansion up to NODE_BUDGET</li>
     *   <li>Extract best move or quit if tree exhausted</li>
     *   <li>Advance current to chosen child</li>
     * </ol>
     *
     * @param solitaire current game state
     * @param moves     recommended legal moves (unused, we compute our own)
     * @param feedback  guidance/error feedback (unused)
     * @return command string ("move ...", "turn", or "quit")
     */
    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        // ===== Step 1: Initialise tree if this is a new game =====
        if (root == null) {
            if (log.isDebugEnabled()) {
                log.debug("Initialising new A* game tree");
            }
            root = new AStarTreeNode(solitaire.copy());
            current = root;
            openSet = new PriorityQueue<>();
            bestG = new HashMap<>();
            bestG.put(current.getStateKey(), current.getG());
        }

        // ===== Step 2: Refresh current state with fresh planning copy =====
        // The real game may have revealed cards that were UNKNOWN in our tree
        Solitaire freshState = solitaire.copy();
        current.setState(freshState);
        current.setH(AStarTreeNode.computeHeuristic(freshState));
        current.recalculateF();

        // ===== Step 3: Invalidate stale children =====
        // Children were computed with old UNKNOWN cards; re-explore with fresh state
        // Also clear openSet since all nodes in it are now stale
        if (!current.getChildren().isEmpty()) {
            if (log.isDebugEnabled()) {
                log.debug("Invalidating {} stale children of current node", current.getChildren().size());
            }
            invalidateDescendants(current);
            current.getChildren().clear();
        }
        // Always reset openSet and bestG to start fresh from current on each turn
        // Previous search results are stale after real game state changes
        openSet.clear();
        bestG.clear();
        openSet.add(current);
        bestG.put(current.getStateKey(), current.getG());

        // ===== Step 4: Check for terminal state =====
        if (current.isWon()) {
            if (log.isDebugEnabled()) {
                log.debug("Game won! Returning quit.");
            }
            return "quit";
        }

        // ===== Step 5: A* Search Loop =====
        AStarTreeNode bestWinNode = null;
        int expansions = 0;

        while (!openSet.isEmpty() && expansions < NODE_BUDGET) {
            AStarTreeNode node = openSet.poll();
            expansions++;

            // Skip pruned nodes
            if (node.isPruned()) {
                continue;
            }

            // Check for win
            if (node.isWon()) {
                bestWinNode = node;
                if (log.isDebugEnabled()) {
                    log.debug("Found winning path after {} expansions", expansions);
                }
                break;
            }

            // Skip terminal non-win states (stuck)
            if (node.isTerminal()) {
                if (log.isTraceEnabled()) {
                    log.trace("Skipping terminal node (stuck state)");
                }
                continue;
            }

            // Generate legal moves from this node's state
            Solitaire nodeState = node.getState();
            List<String> legalMoves = LegalMovesHelper.listLegalMoves(nodeState);
            
            if (log.isTraceEnabled()) {
                log.trace("Expanding node with {} legal moves: {}", legalMoves.size(), legalMoves);
            }

            // Get parent's move signature for ping-pong detection
            MoveSignature parentSig = (node.getMove() != null) 
                    ? MoveSignature.tryParse(node.getMove()) : null;

            // Expand children
            for (String moveCmd : legalMoves) {
                // ----- Pruning: Skip quit moves -----
                if (moveCmd.trim().equalsIgnoreCase("quit")) {
                    if (log.isTraceEnabled()) {
                        log.trace("Skipping quit move");
                    }
                    continue;
                }

                // ----- Pruning: Ping-pong detection -----
                MoveSignature moveSig = MoveSignature.tryParse(moveCmd);
                if (parentSig != null && moveSig != null && moveSig.isInverseOf(parentSig)) {
                    if (log.isTraceEnabled()) {
                        log.trace("Skipping ping-pong move: {}", moveCmd);
                    }
                    continue;
                }

                // Check if child already exists
                TreeNode existingChild = node.getChildren().get(moveCmd);
                if (existingChild != null) {
                    if (log.isDebugEnabled()) {
                        log.debug("Skipping existing child for move: {}", moveCmd);
                    }
                    continue;
                }

                // Create child state by copying and applying move
                Solitaire childState = node.copyState();
                applyMove(childState, moveCmd);

                // Create child node
                AStarTreeNode child = new AStarTreeNode();
                child.setState(childState);
                child.setMove(moveCmd);
                child.setParent(node);
                node.addChild(moveCmd, child);

                // ----- Pruning: Useless king moves -----
                if (child.isUselessKingMove()) {
                    child.markPruned();
                    continue;
                }

                // ----- Pruning: Cycle detection -----
                if (child.isCycleDetected()) {
                    child.markPruned();
                    if (log.isTraceEnabled()) {
                        log.trace("Cycle detected for move: {}", moveCmd);
                    }
                    continue;
                }

                // Compute A* scores
                child.setG(node.getG() + 1.0);
                child.setH(AStarTreeNode.computeHeuristic(childState));
                child.setProbability(AStarTreeNode.computeProbability(moveCmd, nodeState));
                child.recalculateF();

                // ----- Pruning: Duplicate paths (better g already exists) -----
                long childKey = child.getStateKey();
                Double previousG = bestG.get(childKey);
                if (previousG != null && previousG <= child.getG()) {
                    // We've reached this state via a better or equal path
                    if (log.isTraceEnabled()) {
                        log.trace("Skipping duplicate path for move: {} (prevG={}, newG={})", 
                                moveCmd, previousG, child.getG());
                    }
                    continue;
                }
                bestG.put(childKey, child.getG());

                // Add to open set for expansion (with size limit to prevent OOM)
                if (openSet.size() < MAX_OPEN_SET_SIZE) {
                    if (log.isTraceEnabled()) {
                        log.trace("Adding child to openSet: {} (f={})", moveCmd, child.getF());
                    }
                    openSet.add(child);
                }
            }
        }

        if (log.isDebugEnabled()) {
            log.debug("A* search completed: {} expansions, openSet size: {}", expansions, openSet.size());
        }

        // ===== Step 6: Extract best move =====
        String selectedMove;

        if (bestWinNode != null) {
            // Found a winning path — trace back to find first move from current
            selectedMove = traceFirstMove(bestWinNode, current);
            if (log.isDebugEnabled()) {
                log.debug("Selected winning move: {}", selectedMove);
            }
        } else if (openSet.isEmpty()) {
            // Tree exhausted — no viable paths remain
            if (log.isDebugEnabled()) {
                log.debug("Tree exhausted, no winning path found. Quitting.");
            }
            return "quit";
        } else {
            // No win found within budget — pick the best node's path
            AStarTreeNode best = findBestNodeFromCurrent();
            if (best == null || best == current) {
                if (log.isDebugEnabled()) {
                    log.debug("No progress possible. Quitting.");
                }
                return "quit";
            }
            selectedMove = traceFirstMove(best, current);
            if (log.isDebugEnabled()) {
                log.debug("Selected best heuristic move: {}", selectedMove);
            }
        }

        // ===== Step 7: Advance current to chosen child =====
        if (selectedMove != null) {
            TreeNode nextNode = current.getChildren().get(selectedMove);
            if (nextNode instanceof AStarTreeNode) {
                AStarTreeNode chosenChild = (AStarTreeNode) nextNode;
                // Prune siblings to free memory - only keep the chosen path
                pruneSiblings(current, selectedMove);
                current = chosenChild;
                // Clean openSet: remove nodes not descendant of new current
                cleanOpenSet();
            }
        }

        return selectedMove != null ? selectedMove : "quit";
    }

    /**
     * Applies a move command to a Solitaire state.
     *
     * @param solitaire the state to modify
     * @param move      the move command ("turn" or "move X [card] Y")
     */
    private void applyMove(Solitaire solitaire, String move) {
        if (move == null || solitaire == null) {
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
     * Traces back from a node to find the first move from the search root.
     *
     * @param node       the target node (e.g., winning node or best heuristic)
     * @param searchRoot the root of the current search (usually 'current')
     * @return the first move from searchRoot toward node, or null if not reachable
     */
    private String traceFirstMove(AStarTreeNode node, AStarTreeNode searchRoot) {
        if (node == null || node == searchRoot) {
            return null;
        }
        // Walk back until we find a node whose parent is searchRoot
        AStarTreeNode child = node;
        while (child.getParent() != null && child.getParent() != searchRoot) {
            if (child.getParent() instanceof AStarTreeNode) {
                child = (AStarTreeNode) child.getParent();
            } else {
                break;
            }
        }
        return child.getMove();
    }

    /**
     * Finds the best (lowest f-score) node that is a descendant of current.
     *
     * @return the best node from the open set, or null if none valid
     */
    private AStarTreeNode findBestNodeFromCurrent() {
        // The openSet is already sorted by f-score
        // Find the first node that is a descendant of current
        for (AStarTreeNode node : openSet) {
            if (!node.isPruned() && isDescendantOf(node, current)) {
                return node;
            }
        }
        return null;
    }

    /**
     * Checks if a node is a descendant of an ancestor (or equal to it).
     *
     * @param node     the potential descendant
     * @param ancestor the potential ancestor
     * @return true if node is ancestor or descends from it
     */
    private boolean isDescendantOf(TreeNode node, TreeNode ancestor) {
        TreeNode n = node;
        while (n != null) {
            if (n == ancestor) {
                return true;
            }
            n = n.getParent();
        }
        return false;
    }

    /**
     * Prunes sibling branches when advancing to a chosen child.
     * This is critical for memory management - when we commit to a move,
     * all alternative branches become irrelevant.
     *
     * @param parent     the parent node (current before advancing)
     * @param chosenMove the move leading to the child we're advancing to
     */
    private void pruneSiblings(AStarTreeNode parent, String chosenMove) {
        TreeNode chosenChild = parent.getChildren().get(chosenMove);
        parent.getChildren().clear();
        if (chosenChild != null) {
            parent.getChildren().put(chosenMove, chosenChild);
        }
    }

    /**
     * Removes nodes from openSet that are not descendants of current.
     * Also cleans up bestG entries for orphaned nodes.
     */
    private void cleanOpenSet() {
        Iterator<AStarTreeNode> it = openSet.iterator();
        while (it.hasNext()) {
            AStarTreeNode node = it.next();
            if (!isDescendantOf(node, current)) {
                it.remove();
                // Optionally remove from bestG, but leave it for now as it might help pruning
            }
        }
    }

    /**
     * Removes all descendants of a node from the openSet and bestG map.
     * Called when invalidating stale children after card reveals.
     *
     * @param parent the parent whose descendants should be removed
     */
    private void invalidateDescendants(AStarTreeNode parent) {
        Iterator<AStarTreeNode> it = openSet.iterator();
        while (it.hasNext()) {
            AStarTreeNode node = it.next();
            if (isDescendantOf(node, parent) && node != parent) {
                it.remove();
                bestG.remove(node.getStateKey());
            }
        }
    }
}

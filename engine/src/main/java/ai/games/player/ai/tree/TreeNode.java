package ai.games.player.ai.tree;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for tree nodes used in game search algorithms.
 *
 * This abstract class provides common functionality and tree structure shared by different search strategies:
 * - AlphaSolitaire uses tree nodes for Monte Carlo Tree Search (MCTS)
 * - AStarPlayer uses tree nodes for A* search with persistent game tree tracking
 *
 * <p><b>Common tree structure:</b>
 * <ul>
 *   <li>{@code parent}: Points to the parent node (null if this is the root)
 *   <li>{@code state}: The Solitaire game state at this node
 *   <li>{@code stateKey}: Hashed state key for quick comparisons
 *   <li>{@code children}: Map from move string to child nodes
 * </ul>
 *
 * <p><b>Subclass specialization:</b>
 * Each subclass adds its own fields for specialized search logic (MCTS priors, A* costs, etc.)
 * and implements its own tree expansion and evaluation methods.
 */
public abstract class TreeNode {

    /**
     * Parent node in the tree (null if this is the root).
     */
    public TreeNode parent;

    /**
     * The game state at this node.
     * May be null for game tree nodes that only track state history (not full state).
     */
    public Solitaire state;

    /**
     * Hashed state key for quick comparisons and lookups.
     * Derived from the state; null state yields 0L.
     */
    public long stateKey;

    /**
     * Children: map from move string to resulting child node.
     * Provides a unified interface for accessing child nodes across different search strategies.
     */
    public final Map<String, TreeNode> children = new HashMap<>();

    /**
     * Pruned flag: when true, indicates this subtree should not be explored further.
     * This persists across game decisions—once we mark a branch as unproductive (e.g., leads to cycles),
     * future lookahead searches will skip it, saving exploration budget for more promising paths.
     */
    public boolean pruned = false;

    /**
     * Cycle depth: how many moves deep into a detected cycle are we?
     * Set when cycle detection finds a cycle; used to understand cycle severity.
     * A larger depth means we're wasting more moves in the cycle pattern.
     */
    public int cycleDepth = 0;

    /**
     * Protected constructor for subclasses.
     */
    protected TreeNode() {
        this.parent = null;
        this.state = null;
        this.stateKey = 0L;
        this.pruned = false;
        this.cycleDepth = 0;
    }

    /**
     * Set the parent node of this node.
     *
     * @param parent the parent node, or null if this is the root
     */
    public void setParent(TreeNode parent) {
        this.parent = parent;
    }

    /**
     * Get the parent node of this node.
     *
     * @return the parent node, or null if this is the root
     */
    public TreeNode getParent() {
        return parent;
    }

    /**
     * Get the game state at this node.
     *
     * @return the Solitaire state, or null if not available
     */
    public Solitaire getState() {
        return state;
    }

    /**
     * Set the game state at this node.
     *
     * @param state the Solitaire state
     */
    protected void setState(Solitaire state) {
        this.state = state;
        this.stateKey = state != null ? state.getStateKey() : 0L;
    }

    /**
     * Get the state key for this node.
     *
     * @return the hashed state key
     */
    public long getStateKey() {
        return stateKey;
    }

    /**
     * Set the state key directly (useful for game tree nodes that don't have full state).
     *
     * @param stateKey the state key to set
     */
    protected void setStateKey(long stateKey) {
        this.stateKey = stateKey;
    }

    /**
     * Check if the game is won from the given state.
     *
     * @param solitaire the game state
     * @return true if all 52 cards are in the foundation piles
     */
    public static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    /**
     * Clamp a value to the range [0.0, 1.0].
     *
     * @param v the value to clamp
     * @return the clamped value
     */
    protected static double clamp01(double v) {
        if (v < 0.0) {
            return 0.0;
        }
        if (v > 1.0) {
            return 1.0;
        }
        return v;
    }

    /**
     * Check if this node or any of its ancestors is marked as pruned.
     *
     * <p><b>Why check ancestors?</b> If a parent node is pruned (marked as leading to unproductive
     * paths), then all its descendants are also effectively pruned. By walking up the tree,
     * we can quickly determine if we're in a pruned subtree without examining every node.
     *
     * @return true if this node or any ancestor is marked pruned, false otherwise
     */
    public boolean isPruned() {
        TreeNode current = this;
        while (current != null) {
            if (current.pruned) {
                return true;
            }
            current = current.parent;
        }
        return false;
    }

    /**
     * Mark this node as pruned, indicating its subtree should not be explored further.
     *
     * <p><b>When to call:</b> After cycle detection detects that a particular state-transition
     * path leads to a cycle or stagnation, mark the node as pruned. This prevents future game
     * decisions from re-exploring the same unproductive pattern.
     *
     * <p><b>Memory across decisions:</b> The game tree is persistent across all moves in a game.
     * Once a node is marked pruned, it stays pruned for the remainder of the game. This creates
     * a "learning" effect: the more moves the player makes, the more unproductive paths it avoids.
     */
    public void markPruned() {
        this.pruned = true;
    }

    /**
     * Checks whether a move string is a "quit" command.
     *
     * @param move the move command string
     * @return true if the command is "quit" (case-insensitive), false otherwise
     */
    public static boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    /**
     * Create a copy of the current game state for exploration.
     *
     * @return a new Solitaire instance with the same board configuration
     */
    public abstract Solitaire copyState();

    /**
     * Check if this node is terminal (no moves or game won).
     * Terminal nodes may not need further evaluation; their value might be definite.
     *
     * @return true if no moves are available or the game is won
     */
    public abstract boolean isTerminal();

    /**
     * Prunes moves that shift a king between tableau columns without revealing new cards.
     *
     * <p><b>What does this prune?</b> If there are no face-down cards beneath a king in one
     * tableau column, moving that king to another empty column won't reveal anything and is
     * purely lateral. Such moves don't make progress and waste search budget.
     *
     * <p><b>Why prune this?</b> The search has a fixed budget. By eliminating
     * obviously-wasteful moves early, we keep the budget for more promising candidates.
     *
     * <p><b>What moves does this NOT prune?</b>
     * <ul>
     *   <li>King moves to the foundation (different columns, may enable other moves)
     *   <li>King moves when there ARE face-down cards beneath (revealing is valuable)
     *   <li>Non-king moves of any type
     * </ul>
     *
     * <p><b>Implementation:</b>
     * <ol>
     *   <li>Parse the move string to extract source column and card
     *   <li>Verify both source and destination are tableau columns
     *   <li>Find the card being moved and check it's a king
     *   <li>Examine the source column's face-down count
     *   <li>Return true (prune) only if the move is T→T, is a king, and has no face-downs
     * </ol>
     *
     * @param move the move command string to evaluate
     * @return true if the move is a useless king shuffle (should be pruned), false otherwise
     */
    public boolean isUselessKingMove(String move) {
        if (move == null || state == null) {
            return false;
        }
        String trimmed = move.trim();
        String[] parts = trimmed.split("\\s+");
        if (parts.length < 3) {
            return false;
        }
        if (!parts[0].equalsIgnoreCase("move")) {
            return false;
        }
        String from = parts[1];
        String dest = parts[parts.length - 1].toUpperCase();
        
        // Only prune tableau-to-tableau king moves. Allow king-to-foundation moves since
        // they might enable other plays or be part of a winning sequence.
        if (!from.startsWith("T") || !dest.startsWith("T")) {
            return false;
        }
        
        int pileIndex;
        try {
            pileIndex = Integer.parseInt(from.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return false;
        }
        List<List<Card>> visibleTableau = state.getVisibleTableau();
        if (pileIndex < 0 || pileIndex >= visibleTableau.size()) {
            return false;
        }

        String cardToken = parts[2];
        List<Card> tableauPile = visibleTableau.get(pileIndex);
        if (tableauPile == null || tableauPile.isEmpty()) {
            return false;
        }

        // Find the specific card being moved and verify it's a king.
        Card moving = null;
        for (Card c : tableauPile) {
            if (cardToken.equalsIgnoreCase(c.shortName())) {
                moving = c;
                break;
            }
        }
        if (moving == null || moving.getRank() != Rank.KING) {
            return false;
        }

        // Check the face-down count beneath this pile. If it's zero, revealing nothing means
        // the move doesn't make progress and can be pruned.
        List<Integer> faceDowns = state.getTableauFaceDownCounts();
        if (pileIndex < 0 || pileIndex >= faceDowns.size()) {
            return false;
        }
        int facedownCount = faceDowns.get(pileIndex);
        return facedownCount == 0;
    }

    /**
     * Checks whether a specific move would lead to a pruned subtree (marked as cycling or stagnating).
     *
     * <p><b>Purpose:</b> This method queries the persistent game tree to determine if a move
     * would land us in a subtree that has been marked {@code pruned}. If so, we skip the move
     * during search expansion to avoid re-exploring known-bad paths.
     *
     * <p><b>How it works:</b>
     * <ol>
     *   <li>Apply the move to a copy of the current state
     *   <li>Look up the resulting board state in the persistent game tree (starting from root)
     *   <li>If found and marked pruned, return true (prune this move)
     *   <li>Otherwise, return false (move is safe to explore)
     * </ol>
     *
     * <p><b>Why this matters:</b> The game tree persists across all game decisions. As the game
     * progresses, nodes are marked {@code pruned} when they lead to cycles. During search,
     * this method prevents the search from re-exploring those same unproductive branches,
     * conserving search budget for paths that haven't been ruled out.
     *
     * @param move the move command string to check
     * @param applyMoveFunction a function to apply moves to state copies
     * @return true if the move would land in a pruned subtree, false otherwise
     */
    public boolean isCycleDetected(String move, ApplyMoveFunction applyMoveFunction) {
        if (state == null) {
            return false;
        }
        
        // Apply the move to see where it leads
        Solitaire copy = state.copy();
        applyMoveFunction.apply(copy, move);
        long resultKey = copy.getStateKey();
        
        // Find the root of the tree (walk up parent chain)
        TreeNode root = this;
        while (root.parent != null) {
            root = root.parent;
        }
        
        // Search the persistent game tree for a node matching this state
        TreeNode node = root.findNodeByStateKey(resultKey);
        
        // If we found the node and it's marked pruned (or any ancestor is), skip this move
        return node != null && node.isPruned();
    }

    /**
     * Searches the persistent game tree starting from this node for a node with a given state key.
     *
     * <p><b>Note:</b> This is a depth-first search through the game tree. For games with many
     * moves, this could be slow. Optimisations like a state-key map could improve performance,
     * but for now we favour simplicity and correctness.
     *
     * @param targetKey the state key to search for
     * @return the node with matching state key, or null if not found
     */
    public TreeNode findNodeByStateKey(long targetKey) {
        if (this.stateKey == targetKey) {
            return this;
        }
        // Search children
        for (TreeNode childNode : this.children.values()) {
            TreeNode found = childNode.findNodeByStateKey(targetKey);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    /**
     * Functional interface for applying moves to state copies.
     * Used by isCycleDetected to apply moves in a pluggable way.
     */
    @FunctionalInterface
    public interface ApplyMoveFunction {
        void apply(Solitaire solitaire, String move);
    }
}

package ai.games.player.ai.tree;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
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
    protected TreeNode parent;

    /**
     * The game state at this node.
     * May be null for game tree nodes that only track state history (not full state).
     */
    protected Solitaire state;

    /**
     * Hashed state key for quick comparisons and lookups.
     * Derived from the state; null state yields 0L.
     */
    protected long stateKey;

    /**
     * Children: map from move string to resulting child node.
     * Provides a unified interface for accessing child nodes across different search strategies.
     */
    protected final Map<String, TreeNode> children = new HashMap<>();

    /**
     * Pruned flag: when true, indicates this subtree should not be explored further.
     * This persists across game decisions—once we mark a branch as unproductive (e.g., leads to cycles),
     * future lookahead searches will skip it, saving exploration budget for more promising paths.
     */
    protected boolean pruned = false;


    /**
     * Current move being evaluated. Set before calling isCycleDetected() or isUselessKingMove().
     * These methods evaluate the move in the context of this node's state.
     */
    protected String move = null;

    /**
     * Protected constructor for subclasses.
     */
    protected TreeNode() {
        this.parent = null;
        this.state = null;
        this.stateKey = 0L;
        this.pruned = false;
        this.move = null;
    }

    /**
     * String representation of the node for debugging.
     *
     * @return a string summarizing the node's state key and number of children
     */
    public String toString() {
        return "TreeNode[stateKey=" + stateKey + ", move=" + move + ", children=" + children.size() + "]";
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
    public void setState(Solitaire state) {
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
     * Get the move string that led to this node.
     *
     * @return the move string
     */
    public String getMove() {
        return move;
    }   

    /**
     * Set the move string that led to this node.
     *
     * @param move the move string
     */
    public void setMove(String move) {
        this.move = move;
    }

    /**
     * Applies a move command to a Solitaire state.
     *
     * @param move the move command ("turn" or "move X [card] Y")
     */
    public void applyMove(String move) {
        if (move == null || state == null) {
            throw new IllegalArgumentException("Move and state cannot be null");
        }
        String trimmed = move.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            state.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                state.moveCard(parts[1], parts[2], parts[3]);
            } else {
                state.moveCard(parts[1], null, parts[2]);
            }
        }
    }

    /**
     * Get the children map (move string to child node).
     *
     * @return the map of children nodes
     */
    public Map<String, TreeNode> getChildren() {
        return children;
    }

    /**
     * Add a child node for the given move.
     * 
     * @param move the move string
     * @param child the child node
     */
    public void addChild(String move, TreeNode child) {
        if(move == null || child == null) {
            throw new IllegalArgumentException("Move and child cannot be null");
        }

        children.put(move, child);
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
     * Set the pruned flag for this node.
     *
     * @param pruned true to mark as pruned, false otherwise
    */
    public void setPruned(boolean pruned) {
        this.pruned = pruned;
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
     * Create a copy of the current game state for exploration.
     *
     * @return a new Solitaire instance with the same board configuration, or null if state is null
     */
    public Solitaire copyState() {
        return state != null ? state.copy() : null;
    }

    /**
     * Check if the game is won from the given state.
     *
     * @return true if all 52 cards are in the foundation piles
     */
    public boolean isWon() {
        int total = 0;
        for (var pile : state.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    /**
     * Check if this node is terminal (no moves or game won).
     * Terminal nodes may not need further evaluation; their value might be definite.
     *
     * @return true if no moves are available or the game is won
     */
    public boolean isTerminal() {
        if (state == null) {
            return true;  // Null state is considered terminal
        }
        // Check if game is already won
        if (isWon()) {
            return true;
        }
        // Check if there are any legal moves available
        return LegalMovesHelper.listLegalMoves(state).isEmpty();
    }

    /**
     * Checks whether a move string is a "quit" command.
     *
     * @param move the move command string
     * @return true if the command is "quit" (case-insensitive), false otherwise
     */
    public boolean isQuit() {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    /**
     * Checks whether a move string is a "turn" command.
     *
     * @return true if the command starts with "turn" (case-insensitive), false otherwise
     */
    public boolean isTurn() {
        return move != null && move.trim().toLowerCase().startsWith("turn");
    }

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
     *   <li>Parse the move string (from this.move) to extract source column and card
     *   <li>Verify both source and destination are tableau columns
     *   <li>Find the card being moved and check it's a king
     *   <li>Examine the source column's face-down count
     *   <li>Return true (prune) only if the move is T→T, is a king, and has no face-downs
     * </ol>
     *
     * <p><b>Before calling:</b> Set {@code this.move} to the move command string to evaluate.
     *
     * @return true if the move is a useless king shuffle (should be pruned), false otherwise
     */
    public boolean isUselessKingMove() {
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
     * Checks whether a move would create a repeating cycle (ping-ponging).
     *
     * <p><b>Purpose:</b> Detect when a move would return to a state we've visited before,
     * AND that state is part of a repeating pattern. A single return to a state is fine
     * (exploration), but when the same sequence repeats twice, that's wasteful ping-ponging.
     *
     * <p><b>How it works:</b>
     * <ol>
     *   <li>Apply the move (from this.move) to a copy of the current state to get the resulting state key
     *   <li>Count how many ancestors in the path have that same state key
     *   <li>If 2 or more ancestors have it, the cycle pattern has repeated at least twice
     *   <li>Return true (ping-ponging detected)
     * </ol>
     *
     * <p><b>Why we don't check intermediate moves:</b> The state key is a Zobrist hash that
     * fully encodes the entire game state (card positions, face-up/face-down counts, stock, talon).
     * Two nodes with identical state keys represent identical boards and will have identical legal moves.
     * Therefore, the path taken to reach a state doesn't matter—identical states always lead to
     * identical futures. Counting occurrences of the state key is sufficient.
     *
     * <p><b>Example:</b>
     * <pre>
     * Pos 0: State X (key=123)
     * Pos 1: Move A → State Y
     * Pos 2: Move B → State Z
     * Pos 3: Move C → State X (key=123) ← First return to X (1 occurrence)
     * Pos 4: Move A → State Y
     * Pos 5: Move B → State Z
     * Pos 6: Move C → State X (key=123) ← Second return to X (2 occurrences = PING-PONG!)
     * </pre>
     *
     * <p><b>Before calling:</b> Set {@code this.move} to the move command string to check.
     *
     * @return true if this move would cause a repeating cycle (2+ occurrences), false otherwise
     */
    public boolean isCycleDetected() {
        if (state == null || move == null) {
            return false;
        }
        
        // Get the state key for the current tree node
        long resultKey = state.getStateKey();
        
        // Count how many ancestors have this state key
        int occurrences = 0;
        TreeNode current = parent;
        while (current != null) {
            if (current.stateKey == resultKey) {
                occurrences++;
                // If we've found this state 2+ times, the cycle repeats (ping-pong)
                if (occurrences >= 2) {
                    return true;
                }
            }
            current = current.parent;
        }
        
        // State appears 0 or 1 time in ancestors—no repeating cycle
        return false;
    }

    /**
     * Get the list of cards unknown to the player at this node during lookahead (PLAN mode).
     *
     * <p><b>Purpose:</b> During PLAN mode lookahead, cards that haven't been revealed to the
     * player (face-down cards under tableau piles, unseen stock cards) are represented as
     * {@link Card#UNKNOWN}. This method returns the list of actual cards that are currently
     * unknown to the player, allowing search algorithms to understand what information the
     * player doesn't have access to at this point in the game tree.
     *
     * <p><b>Usage:</b> When executing search (A*, MCTS) in PLAN mode, call this method to
     * determine which cards the search should treat as hidden when evaluating board positions.
     *
     * @return the list of cards unknown to the player at this node state, or an empty list if
     *         the state is null or no unknown cards are tracked
     */
    public List<Card> getUnknownCards() {
        if (state == null) {
            return List.of();
        }
        return state.getUnknownCards();
    }
}

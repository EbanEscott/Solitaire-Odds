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
     * Why this node was pruned.
      *
      * <p>When pruning affects behaviour or performance, it is otherwise difficult to tell whether a
      * branch was never explored or was explored and then eliminated. Recording a reason makes
      * debugging and search tuning much easier.
     */
    protected PruneReason pruneReason = PruneReason.NONE;

    /**
     * Optional debugging notes explaining the prune (kept null by default).
      *
      * <p>A reason enum is often enough, but in practice you sometimes need context (e.g. the command
      * that triggered pruning, or a short explanation). This field is deliberately nullable so we
      * don't allocate strings for every node.
     */
    protected String pruneNotes = null;


    /**
     * Current move being evaluated. Set before calling isCycleDetected() or isUselessKingMove().
     * These methods evaluate the move in the context of this node's state.
     */
    protected Move move = null;

    /**
     * Protected constructor for subclasses.
     */
    protected TreeNode() {
        this.parent = null;
        this.state = null;
        this.stateKey = 0L;
        this.pruned = false;
        this.pruneReason = PruneReason.NONE;
        this.pruneNotes = null;
        this.move = null;
    }

    /**
     * String representation of the node for debugging.
     *
     * @return a string summarizing the node's state key and number of children
     */
    public String toString() {
        return "TreeNode[stateKey="
                + stateKey
                + ", move="
                + getMove()
                + ", pruned="
                + pruned
                + ", pruneReason="
                + pruneReason
                + ", children="
                + children.size()
                + "]";
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
        if(state == null) {
            throw new IllegalArgumentException("State cannot be null");
        }
        this.state = state;
        this.stateKey = state.getStateKey();
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
     * Get the move that led to this node.
     *
     * @return the structured move (or null if no recognised move was applied)
     */
    public Move getMove() {
        return move;
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

        // Parse and store a structured representation for fast comparisons.
        // Catch parsing failures so applyMove remains permissive (unknown commands do not mutate state).
        this.move = null;
        try {
            this.move = Move.tryParse(move);
        } catch (IllegalArgumentException e) {
            return;
        }

        if (this.move.isTurn()) {
            state.turnThree();
            this.stateKey = state.getStateKey();
            return;
        }

        if (this.move.isMove()) {
            String fromCode = this.move.from() != null ? this.move.from().toCode() : null;
            String toCode = this.move.to() != null ? this.move.to().toCode() : null;
            String cardToken = this.move.card() != null ? this.move.card().shortName() : null;
            state.moveCard(fromCode, cardToken, toCode);
        }

        // Keep our cached key aligned with the mutated state.
        this.stateKey = state.getStateKey();
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

        child.setParent(this);
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
        if (!pruned) {
            this.pruneReason = PruneReason.NONE;
            this.pruneNotes = null;
        } else {
            // Preserve any existing reason, otherwise mark as manual.
            if (this.pruneReason == null || this.pruneReason == PruneReason.NONE) {
                this.pruneReason = PruneReason.MANUAL;
            }
        }
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
        markPruned(PruneReason.MANUAL, null);
    }

    /**
     * Mark this node as pruned with an explicit reason.
     *
     * <p>This gives call sites a consistent pruning mechanism while leaving behind a machine-readable
     * explanation that helps with tuning and post-mortems.
     */
    public void markPruned(PruneReason reason) {
        markPruned(reason, null);
    }

    /**
     * Mark this node as pruned with an explicit reason and optional notes.
     *
     * <p>Notes are an escape hatch for rare cases where a reason alone isn't enough to understand
     * the prune during debugging.
     */
    public void markPruned(PruneReason reason, String notes) {
        this.pruned = true;
        this.pruneReason = (reason != null) ? reason : PruneReason.MANUAL;
        this.pruneNotes = notes;
    }

    /**
     * Gets the reason this node was pruned.
     *
     * <p>This exposes pruning decisions to tests, logs, and future debugging tooling without
     * requiring ad-hoc string inspection.
     */
    public PruneReason getPruneReason() {
        return pruneReason;
    }

    /**
     * Gets any prune notes recorded for debugging.
     *
     * <p>Notes are optional and may be null; this accessor makes it easy to surface them in logs or
     * debugging UIs when present.
     */
    public String getPruneNotes() {
        return pruneNotes;
    }

    /**
     * Runs all prune detectors for this node and prunes it if any match.
      *
      * <p>Pruning previously had multiple call sites each manually invoking detectors and setting
      * flags. Centralising the policy here keeps pruning behaviour consistent across different search
      * algorithms and makes it much harder to forget a detector.
     *
     * @return true if the node is now pruned, false otherwise
     */
    public boolean doPruning() {
        if (pruned) {
            return true;
        }

        if (isInverseOfParentMove()) {
            markPruned(PruneReason.INVERSE_OF_PARENT_MOVE, move != null ? move.toCommandString() : null);
            return true;
        }

        if (isUselessKingMove()) {
            markPruned(PruneReason.USELESS_KING_MOVE, move != null ? move.toCommandString() : null);
            return true;
        }

        if (isCycleDetected()) {
            markPruned(PruneReason.CYCLE_DETECTED, move != null ? move.toCommandString() : null);
            return true;
        }

        if (isSimilarSibling()) {
            markPruned(PruneReason.SIMILAR_SIBLING, move != null ? move.toCommandString() : null);
            return true;
        }

        return false;
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
        return move != null && move.isQuit();
    }

    /**
     * Checks whether a move string is a "turn" command.
     *
     * @return true if the command starts with "turn" (case-insensitive), false otherwise
     */
    public boolean isTurn() {
        return move != null && move.isTurn();
    }

    /**
     * Checks whether this node's move is the exact inverse of its parent's move.
     *
     * <p><b>Purpose:</b> Prevent immediate ping-ponging where we make a move and then undo it
     * on the very next step.
     *
     * <p><b>Definition:</b> Two moves are considered inverses if they swap source and destination
     * piles. When both moves explicitly specify a card token, the card must also match.
     *
     * @return true if this move is the inverse of the parent's move, false otherwise
     */
    public boolean isInverseOfParentMove() {
        if (parent == null || parent.move == null || move == null) {
            return false;
        }
        if (!parent.move.isMove() || !move.isMove()) {
            return false;
        }

        Move.PileRef parentFrom = parent.move.from();
        Move.PileRef parentTo = parent.move.to();
        Move.PileRef from = move.from();
        Move.PileRef to = move.to();

        if (parentFrom == null || parentTo == null || from == null || to == null) {
            return false;
        }

        boolean swapped = from.equals(parentTo) && to.equals(parentFrom);
        if (!swapped) {
            return false;
        }

        // If both moves specify a card, require an exact match.
        if (parent.move.card() != null && move.card() != null) {
            return parent.move.card().equals(move.card());
        }

        return true;
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
     *   <li>Use the structured {@link Move} (from {@code this.move}) to extract source/destination piles
     *   <li>Verify both source and destination are tableau columns
     *   <li>Resolve the moved card (explicit card token, or inferred from the pre-move state)
     *   <li>Confirm the moved card is a king (and, when explicit, that it exists in the source pile)
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

        if (!move.isMove()) {
            return false;
        }

        Move.PileRef from = move.from();
        Move.PileRef to = move.to();
        if (from == null || to == null) {
            return false;
        }

        // Only prune tableau-to-tableau king moves.
        // Allow king-to-foundation moves since they might enable other plays or be part of a winning sequence.
        if (from.type() != Move.PileType.TABLEAU || to.type() != Move.PileType.TABLEAU) {
            return false;
        }

        // Classification must be based on the *pre-move* position.
        // In search (MCTS/A*), nodes typically store the state *after* applying `move`.
        // If we inspect `state` here, the source pile may already be empty, causing
        // single-card king shuffles (the common case) to be missed.
        Solitaire referenceState = (parent != null && parent.state != null) ? parent.state : state;

        int pileIndex = from.index();
        List<List<Card>> visibleTableau = referenceState.getVisibleTableau();
        if (pileIndex < 0 || pileIndex >= visibleTableau.size()) {
            return false;
        }
        List<Card> tableauPile = visibleTableau.get(pileIndex);
        if (tableauPile == null || tableauPile.isEmpty()) {
            return false;
        }

        // Resolve the moved card and verify it's a king.
        // If the move explicitly specifies a card, be conservative and confirm that card exists
        // in the source pile in the pre-move state.
        if (move.card() != null) {
            Move.CardRef card = move.card();
            if (card.rank() != Rank.KING) {
                return false;
            }

            boolean found = false;
            for (Card c : tableauPile) {
                if (c != null && c.getRank() == card.rank() && c.getSuit() == card.suit()) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        } else {
            Card inferred = inferMovingCard(referenceState, from);
            if (inferred == null || inferred.getRank() != Rank.KING) {
                return false;
            }
        }

        // Check the face-down count beneath this pile. If it's zero, revealing nothing means
        // the move doesn't make progress and can be pruned.
        List<Integer> faceDowns = referenceState.getTableauFaceDownCounts();
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

    /**
     * Checks whether this node's move is effectively equivalent to any other child move
     * under the same parent.
     *
     * <p><b>Purpose:</b> Some legal moves create multiple destination choices that lead to
     * strategically identical outcomes (symmetry). For example:
     * <ul>
     *   <li>Moving an Ace to any empty foundation pile (F1–F4)</li>
     *   <li>Moving a King to any empty tableau pile (T1–T7)</li>
     * </ul>
     *
     * <p>This method compares the current node against its siblings (other children of its
     * parent) and returns true if at least one sibling represents a similar move.
     */
    public boolean isSimilarSibling() {
        if (parent == null || move == null) {
            return false;
        }

        // IMPORTANT: Symmetry must be evaluated against the pre-move state.
        // Tree nodes typically store post-move state, so prefer parent.state where available.
        Solitaire referenceState = (parent.state != null) ? parent.state : state;
        if (referenceState == null) {
            return false;
        }

        MoveSymmetryKey thisKey = symmetryKeyForMove(move, referenceState);
        if (thisKey == null) {
            return false;
        }

        for (Map.Entry<String, TreeNode> entry : parent.children.entrySet()) {
            TreeNode sibling = entry.getValue();
            if (sibling == null || sibling == this) {
                continue;
            }

            Move siblingMove = resolveSiblingMove(sibling, entry.getKey());
            if (siblingMove == null) {
                continue;
            }

            MoveSymmetryKey siblingKey = symmetryKeyForMove(siblingMove, referenceState);
            if (thisKey.equals(siblingKey)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Resolves a sibling node's {@link Move} for symmetry comparison.
     *
      * <p>The canonical path is that child nodes call {@link #applyMove(String)} which stores a
      * structured {@link Move}. However, some code (including tests and legacy callers) may populate
      * {@link #children} directly with a move string key and a node that never parsed it. This helper
      * keeps that fallback contained so {@link #isSimilarSibling()} remains readable.
     */
    private static Move resolveSiblingMove(TreeNode sibling, String fallbackCommand) {
        if (sibling != null && sibling.move != null) {
            return sibling.move;
        }

        // Fallback for callers that populated parent.children directly without applyMove().
        try {
            return Move.tryParse(fallbackCommand);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    /**
     * The symmetry class for a move.
     *
     * <p>We keep symmetries as a small, explicit set of rules (not wildcard strings) so it's clear
     * exactly which move families we consider strategically equivalent.
     */
    private enum MoveSymmetryKind {
        ACE_TO_EMPTY_FOUNDATION,
        KING_TO_EMPTY_TABLEAU
    }

    /**
     * Canonical key representing the equivalence class for a move under symmetry.
     *
      * <p>Two moves are considered "similar siblings" when they represent the same strategic choice,
      * differing only by which symmetric destination pile was chosen. For that, we keep:
     * <ul>
     *   <li>{@code kind}: which symmetry rule matched</li>
     *   <li>{@code from}: the source pile (the strategic decision starts here)</li>
     *   <li>{@code card}: the moved card (to avoid over-grouping different moves)</li>
     * </ul>
     */
    private record MoveSymmetryKey(MoveSymmetryKind kind, Move.PileRef from, Move.CardRef card) {}

    /**
     * Computes the symmetry key for a move, or null if the move has no symmetric-equivalence rule.
     *
      * <p>This is the single place where we encode "which moves are symmetric" in a way that is
      * semantic (based on piles/cards and board state), not syntactic (string parsing).
     */
    private static MoveSymmetryKey symmetryKeyForMove(Move move, Solitaire referenceState) {
        if (move == null || referenceState == null || !move.isMove()) {
            return null;
        }

        Move.PileRef from = move.from();
        Move.PileRef to = move.to();
        if (from == null || to == null) {
            return null;
        }

        Move.CardRef card = resolveCardForMove(move, referenceState, from);
        if (card == null) {
            return null;
        }

        MoveSymmetryKey key;
        if ((key = aceToEmptyFoundationKey(from, to, card, referenceState)) != null) {
            return key;
        }
        if ((key = kingToEmptyTableauKey(from, to, card, referenceState)) != null) {
            return key;
        }

        return null;
    }

    /**
     * Resolves the card being moved.
     *
      * <p>The engine sometimes emits 3-token moves like {@code move W F1} without an explicit card
      * token. For symmetry classification we still need to know whether that implicit move is, for
      * example, an Ace-to-foundation.
     */
    private static Move.CardRef resolveCardForMove(Move move, Solitaire referenceState, Move.PileRef from) {
        Move.CardRef card = move.card();
        if (card != null) {
            return card;
        }

        Card inferred = inferMovingCard(referenceState, from);
        if (inferred == null) {
            return null;
        }
        return new Move.CardRef(inferred.getRank(), inferred.getSuit());
    }

    /**
     * Symmetry rule: moving an Ace to any empty foundation pile is equivalent.
     *
     * <p>Empty foundations are symmetric; choosing F1 vs F2 for the first Ace doesn't change future
     * possibilities in a meaningful way, so we keep only one representative.
     */
    private static MoveSymmetryKey aceToEmptyFoundationKey(
            Move.PileRef from, Move.PileRef to, Move.CardRef card, Solitaire referenceState) {
        if (to.type() != Move.PileType.FOUNDATION || card.rank() != Rank.ACE) {
            return null;
        }

        int idx = to.index();
        List<List<Card>> foundations = referenceState.getFoundation();
        if (idx < 0 || idx >= foundations.size()) {
            return null;
        }
        if (!foundations.get(idx).isEmpty()) {
            return null;
        }

        return new MoveSymmetryKey(MoveSymmetryKind.ACE_TO_EMPTY_FOUNDATION, from, card);
    }

    /**
     * Symmetry rule: moving a King (or king-led run) to any empty tableau column is equivalent.
     *
     * <p>Empty tableau columns are symmetric; if multiple empty columns exist, the first placement
     * choice is usually interchangeable and explodes branching without adding value.
     */
    private static MoveSymmetryKey kingToEmptyTableauKey(
            Move.PileRef from, Move.PileRef to, Move.CardRef card, Solitaire referenceState) {
        if (to.type() != Move.PileType.TABLEAU || card.rank() != Rank.KING) {
            return null;
        }

        int idx = to.index();
        List<List<Card>> tableau = referenceState.getVisibleTableau();
        if (idx < 0 || idx >= tableau.size()) {
            return null;
        }
        if (!tableau.get(idx).isEmpty()) {
            return null;
        }

        return new MoveSymmetryKey(MoveSymmetryKind.KING_TO_EMPTY_TABLEAU, from, card);
    }

    /**
     * Infers the moved card for move strings that omit an explicit card token.
     *
      * <p>Some engine commands intentionally omit the card for brevity. For classification and
      * pruning, we can infer the card as "the top card" of the source pile in the pre-move position.
     */
    private static Card inferMovingCard(Solitaire referenceState, Move.PileRef from) {
        if (referenceState == null || from == null) {
            return null;
        }

        return switch (from.type()) {
            case WASTE -> {
                List<Card> talon = referenceState.getTalon();
                yield talon.isEmpty() ? null : talon.get(talon.size() - 1);
            }
            case STOCK -> null;
            case FOUNDATION -> {
                int idx = from.index();
                List<List<Card>> foundations = referenceState.getFoundation();
                if (idx < 0 || idx >= foundations.size()) {
                    yield null;
                }
                List<Card> pile = foundations.get(idx);
                yield pile.isEmpty() ? null : pile.get(pile.size() - 1);
            }
            case TABLEAU -> {
                int idx = from.index();
                List<List<Card>> tableau = referenceState.getVisibleTableau();
                if (idx < 0 || idx >= tableau.size()) {
                    yield null;
                }
                List<Card> pile = tableau.get(idx);
                yield pile.isEmpty() ? null : pile.get(pile.size() - 1);
            }
        };
    }
}

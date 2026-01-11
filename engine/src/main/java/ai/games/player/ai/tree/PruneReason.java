package ai.games.player.ai.tree;

/**
 * Reason a {@link TreeNode} was pruned.
 *
 * <p>Pruning is used to prevent exploring obviously-unproductive branches (e.g. ping-ponging
 * moves, cycles, or known-useless moves). Recording a reason helps debug why a node was
 * excluded from further search.
 */
public enum PruneReason {
    /** No pruning has been applied. */
    NONE,

    /**
     * Pruned by explicit caller request (not a detector).
     *
     * <p><b>Why this exists:</b> Some algorithms prune for control-flow reasons (e.g. committing to
     * a chosen line) and we still want that to be distinguishable from heuristic pruning.
     */
    MANUAL,

    /**
     * The move immediately undoes the parent's move (ping-pong).
     *
     * <p><b>Why this exists:</b> Immediate undo moves waste search budget and commonly create
     * short oscillations that never make progress.
     */
    INVERSE_OF_PARENT_MOVE,

    /**
     * The move is a known-useless king shuffle between tableau columns.
     *
     * <p><b>Why this exists:</b> These moves do not reveal new information or change constraints
     * when no facedown cards are freed, so they burn budget without improving the position.
     */
    USELESS_KING_MOVE,

    /**
     * The move returns to an already-repeated state key pattern.
     *
     * <p><b>Why this exists:</b> Repeating the same state multiple times indicates a cycle and
     * exploring deeper from that loop is typically unproductive.
     */
    CYCLE_DETECTED,

    /**
     * The move is symmetric to an existing sibling move and is redundant.
     *
     * <p><b>Why this exists:</b> Symmetry creates multiple equivalent branches; keeping only one
     * representative reduces branching factor while preserving strategic coverage.
     */
    SIMILAR_SIBLING,
}

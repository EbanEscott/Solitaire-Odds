package ai.games.player.ai.tree;

/**
 * Reason a {@link TreeNode} was pruned.
 *
 * <p>Pruning is used to prevent exploring obviously-unproductive branches (e.g. ping-ponging
 * moves, cycles, or known-useless moves). Recording a reason helps debug why a node was
 * excluded from further search.
 */
public enum PruneReason {
    NONE,
    MANUAL,
    INVERSE_OF_PARENT_MOVE,
    USELESS_KING_MOVE,
    CYCLE_DETECTED,
    SIMILAR_SIBLING,
}

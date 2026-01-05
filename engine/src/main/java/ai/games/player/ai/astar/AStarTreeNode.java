package ai.games.player.ai.astar;

import ai.games.game.Solitaire;
import ai.games.player.ai.tree.TreeNode;

/**
 * Unified node representing both the A* search tree (for lookahead decisions)
 * and the persistent game tree (for cycle detection and intelligent pruning across the full game).
 *
 * <p><b>Dual-purpose design:</b>
 * <ul>
 *   <li>A* search fields ({@code state}, {@code pathCost}, {@code heuristic}) track lookahead
 *       decisions within a single {@code nextCommand()} call, guiding tactical move selection.
 *   <li>Game tree fields ({@code move}, {@code parent}, {@code children}, {@code stateKey}) track
 *       the actual moves made throughout the entire game, enabling strategic cycle detection.
 * </ul>
 *
 * <p><b>What this enables:</b>
 * <ul>
 *   <li><b>Tactical A* search:</b> Uses game state evaluation to find strong moves quickly
 *   <li><b>State visit tracking:</b> Counts how many times each board state has been reached
 *   <li><b>Cycle detection:</b> From any leaf node, walks back up through parents identifying
 *       cycles—repeated patterns of returning to the same board state. Counts nested cycles
 *       to determine if the game is stuck in unproductive loops.
 *   <li><b>Intelligent pruning:</b> Marks unproductive subtrees ({@code pruned = true})
 *       so future decisions avoid them, saving exploration budget
 * </ul>
 *
 * <p><b>Why persistent pruning matters:</b>
 * Klondike has many potential cycles—move sequences that return to previous board states.
 * When Phase 2 (cycle detection) walks up from the current leaf and finds multiple nested cycles,
 * it marks the leaf as {@code pruned}. On the next game decision, when A* considers moves, it skips
 * expanding children of pruned nodes. This memory prevents the player from re-exploring the same
 * unproductive paths, concentrating the 256-node budget on moves that can break the cycle
 * or find new progress. Over a full game, this dramatically improves move quality and shortens
 * games that would otherwise loop indefinitely.
 */
public class AStarTreeNode extends TreeNode implements Comparable<AStarTreeNode> {
    // ============= A* Search Fields =============
    /** The Solitaire game state at this node (inherited from TreeNode). */
    // state field is inherited
    
    /** Path cost from root in the A* search. */
    public final int pathCost;
    
    /** Heuristic value (negative, so lower is better). */
    public final int heuristic;
    
    
    // ============= Game Tree Fields =============
    /** Parent node in the game tree (inherited from TreeNode). */
    // parent field is inherited
    
    /** Children: map from move string to resulting node (inherited from TreeNode). */
    // children field is inherited from base TreeNode
    
    /** The move that led to this state from parent (inherited from TreeNode, can be set before evaluating). */
    // move field is inherited
    
    /** Number of times this exact board state has been visited. */
    public int visitCount = 1;
    
    /** Game progress snapshot: foundation + facedown card counts at this node. */
    public int foundationCount;
    public int facedownCount;
    
    /** State key for quick lookup (inherited from TreeNode). */
    // stateKey field is inherited
    
    /**
     * Pruned flag (inherited from TreeNode): when true, indicates this subtree should not be explored further.
     * This persists across game decisions—once we mark a branch as unproductive (e.g., leads to cycles),
     * future lookahead searches will skip it, saving exploration budget for more promising paths.
     */
    // pruned field is inherited

    /**
     * Cycle depth (inherited from TreeNode): how many moves deep into a detected cycle are we?
     * Set when findCycleAncestor() detects a cycle; used to understand cycle severity.
     * A larger depth means we're wasting more moves in the cycle pattern.
     */
    // cycleDepth field is inherited

    /**
     * Constructor for A* search nodes.
     * Used when building lookahead trees within a single decision.
     */
    public AStarTreeNode(Solitaire state, AStarTreeNode parent, String move, int pathCost, int heuristic) {
        super();
        this.parent = parent;
        setState(state);  // This sets both state and stateKey from base class
        this.move = move;  // Set the inherited move field
        this.pathCost = pathCost;
        this.heuristic = heuristic;
        this.foundationCount = 0;
        this.facedownCount = 0;
    }

    /**
     * Constructor for game tree nodes (persistent game history).
     * Used when tracking moves throughout the entire game.
     */
    public AStarTreeNode(String move, AStarTreeNode parent, long stateKey, int foundationCount, int facedownCount) {
        super();
        this.move = move;  // Set the inherited move field
        this.parent = parent;
        this.foundationCount = foundationCount;
        this.facedownCount = facedownCount;
        
        // A* fields not used in game tree mode
        this.pathCost = 0;
        this.heuristic = 0;
    }

    /**
     * A* f-score: pathCost + heuristic (path cost plus heuristic).
     * Only meaningful for A* search nodes.
     */
    public int f() {
        // Higher heuristic (good board) should reduce cost, so subtract.
        return pathCost - heuristic;
    }

    /**
     * For use in A* priority queue: compare by f-score.
     */
    @Override
    public int compareTo(AStarTreeNode other) {
        return Integer.compare(this.f(), other.f());
    }

    /**
     * Walk up the tree to find how many times we've been in this exact state.
     * Counts all ancestor nodes with the same stateKey.
     */
    public int countVisitsToState() {
        int count = 1;  // This node
        AStarTreeNode p = (AStarTreeNode) parent;
        while (p != null) {
            if (p.getStateKey() == this.getStateKey()) {
                count++;
            }
            p = (AStarTreeNode) p.parent;
        }
        return count;
    }

    /**
     * Check if we're cycling: same state as an ancestor but no progress made.
     * Returns the ancestor node we're cycling to, or null if no cycle detected.
     */
    public AStarTreeNode findCycleAncestor() {
        AStarTreeNode p = (AStarTreeNode) parent;
        while (p != null) {
            if (p.getStateKey() == this.getStateKey() &&
                p.foundationCount == this.foundationCount &&
                p.facedownCount == this.facedownCount) {
                // Found the same board state with no progress
                return p;
            }
            p = (AStarTreeNode) p.parent;
        }
        return null;
    }

    /**
     * Get the distance (number of moves) to a given ancestor node.
     */
    public int distanceTo(AStarTreeNode ancestor) {
        int dist = 0;
        AStarTreeNode current = this;
        while (current != null && current != ancestor) {
            dist++;
            current = (AStarTreeNode) current.parent;
        }
        return current == ancestor ? dist : -1;
    }
}

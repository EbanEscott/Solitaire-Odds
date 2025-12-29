package ai.games.player.ai.astar;

import ai.games.game.Solitaire;
import java.util.HashMap;
import java.util.Map;

/**
 * Unified node representing both the A* search tree (for lookahead decisions)
 * and the persistent game tree (for cycle detection across the full game).
 *
 * <p>A* search fields track lookahead decisions within a single `nextCommand()` call.
 * Game tree fields track the actual moves made throughout the entire game.
 *
 * <p>This allows us to:
 * <ul>
 *     <li>Use A* heuristic search for tactical lookahead</li>
 *     <li>Track how many times we've visited a particular board state</li>
 *     <li>Detect cycles: same board state reached via different move sequences</li>
 *     <li>Detect stagnation: revisiting a state without making game progress</li>
 * </ul>
 */
public class GameTreeNode implements Comparable<GameTreeNode> {
    // ============= A* Search Fields =============
    /** The Solitaire game state at this node. */
    public final Solitaire state;
    
    /** Path cost from root in the A* search. */
    public final int pathCost;
    
    /** Heuristic value (negative, so lower is better). */
    public final int heuristic;
    
    
    // ============= Game Tree Fields =============
    /** The move that led to this state from parent (null for root). */
    public final String move;
    
    /** Parent node in the game tree. */
    public final GameTreeNode parent;
    
    /** Children: map from move string to resulting node. */
    public final Map<String, GameTreeNode> children = new HashMap<>();
    
    /** Number of times this exact board state has been visited. */
    public int visitCount = 1;
    
    /** Game progress snapshot: foundation + facedown card counts at this node. */
    public int foundationCount;
    public int facedownCount;
    
    /** State key for quick lookup. */
    public final long stateKey;

    /**
     * Constructor for A* search nodes.
     * Used when building lookahead trees within a single decision.
     */
    public GameTreeNode(Solitaire state, GameTreeNode parent, String move, int pathCost, int heuristic) {
        this.state = state;
        this.parent = parent;
        this.move = move;
        this.pathCost = pathCost;
        this.heuristic = heuristic;
        this.stateKey = state != null ? state.getStateKey() : 0L;
        this.foundationCount = 0;
        this.facedownCount = 0;
    }

    /**
     * Constructor for game tree nodes (persistent game history).
     * Used when tracking moves throughout the entire game.
     */
    public GameTreeNode(String move, GameTreeNode parent, long stateKey, int foundationCount, int facedownCount) {
        this.move = move;
        this.parent = parent;
        this.stateKey = stateKey;
        this.foundationCount = foundationCount;
        this.facedownCount = facedownCount;
        
        // A* fields not used in game tree mode
        this.state = null;
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
    public int compareTo(GameTreeNode other) {
        return Integer.compare(this.f(), other.f());
    }

    /**
     * Walk up the tree to find how many times we've been in this exact state.
     * Counts all ancestor nodes with the same stateKey.
     */
    public int countVisitsToState() {
        int count = 1;  // This node
        GameTreeNode p = parent;
        while (p != null) {
            if (p.stateKey == this.stateKey) {
                count++;
            }
            p = p.parent;
        }
        return count;
    }

    /**
     * Check if we're cycling: same state as an ancestor but no progress made.
     * Returns the ancestor node we're cycling to, or null if no cycle detected.
     */
    public GameTreeNode findCycleAncestor() {
        GameTreeNode p = parent;
        while (p != null) {
            if (p.stateKey == this.stateKey &&
                p.foundationCount == this.foundationCount &&
                p.facedownCount == this.facedownCount) {
                // Found the same board state with no progress
                return p;
            }
            p = p.parent;
        }
        return null;
    }

    /**
     * Get the distance (number of moves) to a given ancestor node.
     */
    public int distanceTo(GameTreeNode ancestor) {
        int dist = 0;
        GameTreeNode current = this;
        while (current != null && current != ancestor) {
            dist++;
            current = current.parent;
        }
        return current == ancestor ? dist : -1;
    }
}


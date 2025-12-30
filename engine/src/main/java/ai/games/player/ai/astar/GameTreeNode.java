package ai.games.player.ai.astar;

import ai.games.game.Solitaire;
import java.util.HashMap;
import java.util.Map;

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
     * Pruned flag: when true, indicates this subtree should not be explored further.
     * This persists across game decisions—once we mark a branch as unproductive (e.g., leads to cycles),
     * future lookahead searches will skip it, saving exploration budget for more promising paths.
     * 
     * <p><b>Why this is powerful:</b> With a 256-node expansion budget, avoiding known-bad branches
     * means more budget for paths that might break cycles or make progress. Over a full game,
     * this accumulates into significant improvements in move quality and shorter game lengths.
     */
    public boolean pruned = false;
    
    /**
     * Cycle depth: how many moves deep into a detected cycle are we?
     * Set when findCycleAncestor() detects a cycle; used to understand cycle severity.
     * A larger depth means we're wasting more moves in the cycle pattern.
     */
    public int cycleDepth = 0;

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

    /**
     * Check if this node or any of its ancestors is marked as pruned.
     *
     * <p><b>Why check ancestors?</b> If a parent node is pruned (marked as leading to unproductive
     * paths), then all its descendants are also effectively pruned. By walking up the tree,
     * we can quickly determine if we're in a pruned subtree without examining every node.
     *
     * <p><b>Usage in A* search:</b> Before expanding a node's children, check {@code isPruned()}.
     * If true, skip this node to save exploration budget for more promising branches.
     *
     * @return true if this node or any ancestor is marked pruned, false otherwise
     */
    public boolean isPruned() {
        GameTreeNode current = this;
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
     * <p><b>When to call:</b> After cycle detection (Phase 2 of {@code nextCommand()}) detects
     * that a particular state-transition path leads to a cycle or stagnation, mark the cycle node
     * as pruned. This prevents future game decisions from re-exploring the same unproductive pattern.
     *
     * <p><b>Memory across decisions:</b> The game tree is persistent across all moves in a game.
     * Once a node is marked pruned, it stays pruned for the remainder of the game. This creates
     * a "learning" effect: the more moves the player makes, the more unproductive paths it avoids,
     * and the better its move quality becomes.
     *
     * <p><b>Impact on A* search:</b> When A* expands nodes in Phase 4, it will skip children
     * of pruned nodes (via {@code isPruned()} check), conserving the 256-node budget for paths
     * that haven't been ruled out as unproductive.
     */
    public void markPruned() {
        this.pruned = true;
    }
}


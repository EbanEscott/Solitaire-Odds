package ai.games.player.ai.mcts;

import ai.games.player.ai.tree.TreeNode;
import java.util.List;

/**
 * Monte Carlo Tree Node specialisation for MCTS-based search.
 *
 * <p>This node type extends {@link TreeNode} with MCTS-specific evaluation, playout methods,
 * and UCB-based tree statistics.
 * It encapsulates:
 * <ul>
 *   <li>State evaluation for reward computation during playouts
 *   <li>Move application and game state simulation
 *   <li>Terminal state detection
 *   <li>State copying for exploration
 *   <li>UCB statistics: visits count, total reward accumulation
 *   <li>Tree structure: parent/child relationships for backpropagation
 * </ul>
 *
 * <p>These methods are used by MonteCarloPlayer to perform proper Monte Carlo Tree Search with
 * tree building, selection, expansion, simulation, and backpropagation phases.
 */
public class MonteCarloTreeNode extends TreeNode {

    // MCTS statistics
    private int visits = 0;
    private double totalReward = 0.0;

    // Maximum possible evaluation score for normalisation
    public static final double MAX_SCORE = 100.0;

    /**
     * Creates a new MonteCarloTreeNode.
     */
    public MonteCarloTreeNode() {
        super();
    }

    /**
     * State evaluation used as a playout reward.
     *
     * @param solitaire the game state to evaluate
     * @return the heuristic score for this state
     */
    public int evaluate() {
        if (state == null) {
            throw new IllegalStateException("Cannot evaluate null state");
        }

        int score = 0;

        // Foundation progress
        int foundationCards = 0;
        for (var pile : state.getFoundation()) {
            foundationCards += pile.size();
        }
        score += foundationCards * 4; // Weight foundation cards more heavily

        // Tableau visibility: reward face-ups
        int faceUps = 0;
        for(int count : state.getTableauFaceUpCounts())
            faceUps += count;
        score += faceUps;

        // Tableau visibility: reward flipping face-downs
        int faceDowns = 0;
        for(int count : state.getTableauFaceDownCounts())
            faceDowns += count;
        // Assume starting face-down count is 21
        int startingFaceDownCount = 21;
        score += (startingFaceDownCount - faceDowns) * 3; // Weight flipped cards

        if(score > MAX_SCORE) {
            throw new IllegalStateException("Evaluation score exceeds maximum: " + score);
        }
        return score;
    }

    /**
     * Updates MCTS statistics for this node.
     *
     * <p>Increments visit count and accumulates the reward. This is called during
     * backpropagation to update node statistics after a playout.
     *
     * @param reward the reward value from the playout
     */
    public void updateStats(double reward) {
        this.visits++;
        this.totalReward += reward;
    }

    /**
     * Gets the mean reward (average value) for this node.
     *
     * <p>Returns the total accumulated reward divided by visit count.
     * If visits is 0, returns 0.0 to avoid division by zero.
     *
     * @return the mean reward, or 0.0 if never visited
     */
    public double getMeanReward() {
        if (visits == 0) {
            return 0.0;
        }
        return totalReward / visits;
    }

    /**
     * Gets the UCT (Upper Confidence bound for Trees) value of this node.
     *
     * <p>The UCT value combines the mean reward with an exploration bonus:
     * UCT = mean + c * sqrt(ln(parent_visits) / visits)
     *
     * <p>If this node has never been visited (visits == 0), returns POSITIVE_INFINITY
     * to force exploration of unvisited nodes.
     *
     * @param c the exploration constant (typically sqrt(2))
     * @param parentVisits the visit count of the parent node
     * @return the UCT value, or POSITIVE_INFINITY if unvisited
     */
    public double getUctValue(double c, int parentVisits) {
        if (visits == 0) {
            return Double.POSITIVE_INFINITY;
        }
        double mean = getMeanReward();
        double bonus = c * Math.sqrt(Math.log(parentVisits) / visits);
        return mean + bonus;
    }

    /**
     * Gets the number of times this node has been visited.
     *
     * @return the visit count
     */
    public int getVisits() {
        return visits;
    }

    /**
     * Gets the total accumulated reward for this node.
     *
     * @return the sum of all rewards from playouts through this node
     */
    public double getTotalReward() {
        return totalReward;
    }
}

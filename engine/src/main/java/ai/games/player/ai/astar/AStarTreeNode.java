package ai.games.player.ai.astar;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.ai.tree.TreeNode;
import java.util.List;

/**
 * A* search tree node extending the base TreeNode with A* scoring fields.
 *
 * <p>Each node stores:
 * <ul>
 *   <li><b>g</b>: Path cost from the search root (number of moves taken)</li>
 *   <li><b>h</b>: Heuristic estimate of remaining cost to win (lower = closer to goal)</li>
 *   <li><b>f</b>: Ranking score {@code g + h} (used for queue ordering)</li>
 *   <li><b>probability</b>: Cumulative probability of speculative reveal outcomes</li>
 * </ul>
 *
 * <p>Nodes are compared by f-score for use in a priority queue, enabling A* expansion order.
 */
public class AStarTreeNode extends TreeNode implements Comparable<AStarTreeNode> {

    /** Path cost from search root (number of moves). */
    private double g;

    /** Heuristic estimate of remaining cost to win. */
    private double h;

    /** Known-position ordering cost: f = g + h. */
    private double f;

    /** Cumulative probability of all speculative card assignments on this search path. */
    private double probability;

    /** Whether all retained successors for this expansion have been generated. */
    private boolean expanded;

    /** Whether successors are alternative reveal outcomes rather than chosen actions. */
    private boolean chanceNode;

    /** Conditional exact probability or empirical weight within the parent reveal event. */
    private double outcomeWeight = 1.0;

    /**
     * Creates a new A* tree node with default scores.
     */
    public AStarTreeNode() {
        super();
        this.g = 0.0;
        this.h = 0.0;
        this.f = 0.0;
        this.probability = 1.0;
    }

    /**
     * Creates a new A* tree node with the given state.
     *
     * @param state the Solitaire game state at this node
     */
    public AStarTreeNode(Solitaire state) {
        super();
        setState(state);
        this.g = 0.0;
        this.h = computeHeuristic(state);
        this.probability = 1.0;
        this.f = g + h;
    }

    // ========== Getters and Setters ==========

    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }

    public double getH() {
        return h;
    }

    public void setH(double h) {
        this.h = h;
    }

    public double getF() {
        return f;
    }

    public void setF(double f) {
        this.f = f;
    }

    public double getProbability() {
        return probability;
    }

    public void setProbability(double probability) {
        this.probability = probability;
    }

    public boolean isExpanded() {
        return expanded;
    }

    public void setExpanded(boolean expanded) {
        this.expanded = expanded;
    }

    public boolean isChanceNode() {
        return chanceNode;
    }

    public void setChanceNode(boolean chanceNode) {
        this.chanceNode = chanceNode;
    }

    public double getOutcomeWeight() {
        return outcomeWeight;
    }

    public void setOutcomeWeight(double outcomeWeight) {
        this.outcomeWeight = outcomeWeight;
    }

    // ========== A* Scoring Methods ==========

    /**
     * Computes the heuristic estimate for the given state.
     *
     * <p>Heuristic components:
     * <ul>
     *   <li>Base: (52 - foundationCards) — minimum moves to win</li>
     *   <li>Penalty: +2 per face-down card — must be revealed before playing</li>
     *   <li>Penalty: +0.5 per stock card — less accessible than tableau</li>
     *   <li>Bonus: -3 per empty tableau column — strategic flexibility</li>
     * </ul>
     *
     * @param state the Solitaire state to evaluate
     * @return the heuristic value (lower = closer to goal)
     */
    public static double computeHeuristic(Solitaire state) {
        if (state == null) {
            return Double.MAX_VALUE;
        }

        // Count foundation cards (goal progress)
        int foundationCards = 0;
        for (List<Card> pile : state.getFoundation()) {
            foundationCards += pile.size();
        }

        // Count face-down cards across all tableau piles
        int faceDownCards = 0;
        for (int count : state.getTableauFaceDownCounts()) {
            faceDownCards += count;
        }

        // Count stock cards (less accessible)
        int stockCards = state.getStockpile().size();

        // Count empty tableau columns (strategic flexibility)
        int emptyColumns = 0;
        List<List<Card>> visibleTableau = state.getVisibleTableau();
        for (List<Card> pile : visibleTableau) {
            if (pile.isEmpty()) {
                emptyColumns++;
            }
        }

        // Compute heuristic: base + penalties - bonuses
        double h = (52 - foundationCards)       // Base: minimum moves to win
                 + (2.0 * faceDownCards)        // Penalty: face-down cards
                 + (0.5 * stockCards)           // Penalty: stock cards
                 - (3.0 * emptyColumns);        // Bonus: empty columns

        return Math.max(0.0, h);
    }

    /**
     * Recalculates the known-position ordering score from move cost and heuristic.
     * Probability contributes to reveal expectations in the player, not this ordering score.
     * Call this after updating any of the component values.
     */
    public void recalculateF() {
        this.f = g + h;
    }

    // ========== Comparable Implementation ==========

    /**
     * Compares nodes by f-score for priority queue ordering.
     * Lower f-scores receive higher priority under the search ranking heuristic.
     *
     * @param other the other node to compare against
     * @return negative if this node has lower f, positive if higher, 0 if equal
     */
    @Override
    public int compareTo(AStarTreeNode other) {
        return Double.compare(this.f, other.f);
    }

    // ========== String Representation ==========

    @Override
    public String toString() {
        return String.format("AStarTreeNode[g=%.2f, h=%.2f, f=%.2f, p=%.2f, move=%s, children=%d]",
                g, h, f, probability, move, children.size());
    }
}

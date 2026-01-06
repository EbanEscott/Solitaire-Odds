package ai.games.player.ai.astar;

import ai.games.game.Card;
import ai.games.game.Rank;
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
 *   <li><b>f</b>: Total estimated cost = g + h (used for priority queue ordering)</li>
 *   <li><b>probability</b>: Estimated probability of success when moving to UNKNOWN cards</li>
 * </ul>
 *
 * <p>Nodes are compared by f-score for use in a priority queue, enabling A* expansion order.
 */
public class AStarTreeNode extends TreeNode implements Comparable<AStarTreeNode> {

    /** Path cost from search root (number of moves). */
    private double g;

    /** Heuristic estimate of remaining cost to win. */
    private double h;

    /** Total estimated cost: f = g + h/probability. */
    private double f;

    /** Probability of success for moves targeting UNKNOWN cards (1.0 for known destinations). */
    private double probability;

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
     * Computes the probability of a move succeeding when targeting UNKNOWN cards.
     *
     * <p>When moving to a tableau pile topped by an UNKNOWN card, we estimate the probability
     * that the hidden card is compatible (allows the move). This is calculated as:
     * <pre>
     *   P(success) = (compatible cards in unknown pool) / (total unknowns)
     * </pre>
     *
     * <p>The probability is used to penalise risky moves: f = g + h/p.
     * Lower probability results in higher f-score, discouraging uncertain moves.
     *
     * @param move  the move command string to evaluate
     * @param state the current game state
     * @return probability between 0.0 and 1.0 (1.0 if destination is known)
     */
    public static double computeProbability(String move, Solitaire state) {
        if (move == null || state == null) {
            return 1.0;
        }

        // Parse the move to extract destination
        String[] parts = move.trim().split("\\s+");
        if (parts.length < 3 || !parts[0].equalsIgnoreCase("move")) {
            return 1.0; // Turn or other non-move commands
        }

        String dest = parts[parts.length - 1].toUpperCase();
        if (!dest.startsWith("T")) {
            return 1.0; // Foundation moves don't depend on unknown cards
        }

        // Get the destination tableau pile
        int pileIndex;
        try {
            pileIndex = Integer.parseInt(dest.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return 1.0;
        }

        List<List<Card>> visibleTableau = state.getVisibleTableau();
        if (pileIndex < 0 || pileIndex >= visibleTableau.size()) {
            return 1.0;
        }

        List<Card> destPile = visibleTableau.get(pileIndex);
        if (destPile.isEmpty()) {
            return 1.0; // Empty column: only kings can move, no unknown dependency
        }

        Card topCard = destPile.get(destPile.size() - 1);
        if (topCard.getRank() != Rank.UNKNOWN) {
            return 1.0; // Known card at destination
        }

        // Destination is UNKNOWN — estimate probability based on compatible cards
        List<Card> unknowns = state.getUnknownCards();
        if (unknowns.isEmpty()) {
            return 1.0;
        }

        // Parse the card being moved to determine compatibility requirements
        String cardToken = (parts.length == 4) ? parts[2].toUpperCase() : null;
        if (cardToken == null) {
            return 0.5; // Unknown what we're moving, assume 50% chance
        }

        // When destination is UNKNOWN, we can't know if the move will succeed.
        // Rough estimate: about half of unknowns will be compatible colour
        // and 1/13 will be the right rank. So ~1/26 chance per unknown.
        // Simplified: use a fixed probability to penalise uncertain moves.
        double p = 0.5; // Base probability for unknown destination
        return clamp01(p);
    }

    /**
     * Recalculates the f-score based on current g, h, and probability.
     * Call this after updating any of the component values.
     */
    public void recalculateF() {
        // Adjust f-score: lower probability penalises the move
        // f = g + h/p (when p is low, h/p is high, making f higher)
        double adjustedH = (probability > 0.0) ? h / probability : Double.MAX_VALUE;
        this.f = g + adjustedH;
    }

    // ========== Comparable Implementation ==========

    /**
     * Compares nodes by f-score for priority queue ordering.
     * Lower f-score = higher priority (closer to optimal path).
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

package ai.games.player.ai.alpha;

import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Single node in the Monte Carlo Tree Search (MCTS) tree for AlphaSolitaire.
 *
 * Implements the AlphaZero-style MCTS algorithm, combining neural network
 * policy priors with empirical value estimates to guide tree exploration.
 *
 * Architecture:
 * - Each node represents a game state (board position + legal moves)
 * - Priors are set from the policy head (move probabilities)
 * - Values are backed up from the value head or terminal states
 * - UCB1 (PUCT) selection balances exploration and exploitation
 *
 * The tree is ephemeral within a single nextCommand() call; it does not
 * persist across moves.
 */
public class TreeNode {

    /**
     * The game state at this node.
     * Each node has its own copy to allow parallel exploration without mutation.
     */
    private final Solitaire state;

    /**
     * List of legal moves available from this state.
     * Pruned to exclude "quit" moves from MCTS exploration (quit is always available).
     */
    private final List<String> moves;

    /**
     * Prior probabilities for each move from the policy head.
     * Normalized to sum to 1.0. Used in PUCT formula: p * sqrt(N) / (1 + n)
     */
    private final double[] priors;

    /**
     * Cumulative values for each move: sum of backpropagated rewards.
     */
    private final double[] valueSums;

    /**
     * Visit counts for each move: how many times each has been explored.
     */
    private final int[] visits;

    /**
     * Child nodes: one for each legal move (initially null, created on expansion).
     */
    private final TreeNode[] children;

    /**
     * Whether this node has been evaluated by the neural network.
     */
    private boolean evaluated;

    /**
     * Estimated win probability for this position (from value head or heuristic).
     */
    private double valueEstimate;

    /**
     * Construct a new tree node with the given state and legal moves.
     * The node is not yet evaluated; priors and value must be set via ensureEvaluated().
     *
     * @param state the game state at this node
     * @param moves the list of legal moves from this state
     */
    private TreeNode(Solitaire state, List<String> moves) {
        this.state = state;
        this.moves = moves;
        int n = moves.size();
        this.priors = new double[n];
        this.valueSums = new double[n];
        this.visits = new int[n];
        this.children = new TreeNode[n];
        this.evaluated = false;
        this.valueEstimate = 0.0;
    }

    /**
     * Create a root node for MCTS.
     * The root uses all legal moves, including "quit".
     *
     * @param state the current game state
     * @param legalMoves all legal moves from this state
     * @return a new root node
     */
    public static TreeNode createRoot(Solitaire state, List<String> legalMoves) {
        return new TreeNode(state, legalMoves);
    }

    /**
     * Create a child node during tree expansion.
     * The child uses all legal moves from the new state, including "quit".
     *
     * @param state the game state after applying a move
     * @param legalMoves all legal moves from this new state
     * @return a new child node
     */
    public static TreeNode createChild(Solitaire state, List<String> legalMoves) {
        return new TreeNode(state, legalMoves);
    }

    /**
     * Check if this node is terminal (no moves or game won).
     * Terminal nodes do not need neural network evaluation; their value is definite.
     *
     * @return true if no moves are available or the game is won
     */
    public boolean isTerminal() {
        return moves.isEmpty() || isWon(state);
    }

    /**
     * Get the estimated value for this position.
     * If not yet evaluated, returns a heuristic estimate.
     *
     * @return estimated win probability (0.0 = loss, 1.0 = win)
     */
    public double getValueEstimate() {
        if (!evaluated) {
            // If we have not yet contacted the neural service, fall back
            // to a heuristic view of the current state.
            return valueEstimateFromHeuristic();
        }
        return valueEstimate;
    }

    /**
     * Ensure this node has been evaluated by the neural network.
     * Fetches policy priors and value estimate from the service if not yet done.
     *
     * @param client the AlphaSolitaire HTTP client
     * @return the estimated value for this position
     */
    public double ensureEvaluated(AlphaSolitaireClient client) {
        if (evaluated) {
            return valueEstimate;
        }
        if (moves.isEmpty() || isWon(state)) {
            valueEstimate = isWon(state) ? 1.0 : 0.0;
            evaluated = true;
            return valueEstimate;
        }

        try {
            AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(state);
            AlphaSolitaireResponse response = client.evaluate(request);
            if (response != null) {
                // Extract policy priors from response
                Map<String, Double> priorByMove = new HashMap<>();
                List<AlphaSolitaireResponse.MoveScore> scored = response.getLegalMoves();
                if (scored != null) {
                    for (AlphaSolitaireResponse.MoveScore ms : scored) {
                        if (ms.getCommand() != null) {
                            priorByMove.put(ms.getCommand().trim(), ms.getProbability());
                        }
                    }
                }

                // Assign priors to our moves (normalize to sum to 1.0)
                double sum = 0.0;
                for (int i = 0; i < moves.size(); i++) {
                    String m = moves.get(i);
                    double p = priorByMove.getOrDefault(m.trim(), 0.0);
                    priors[i] = p;
                    sum += p;
                }
                if (sum > 0.0) {
                    for (int i = 0; i < priors.length; i++) {
                        priors[i] /= sum;
                    }
                } else {
                    // Uniform prior if no priors were found
                    double uniform = moves.isEmpty() ? 0.0 : 1.0 / moves.size();
                    for (int i = 0; i < priors.length; i++) {
                        priors[i] = uniform;
                    }
                }

                // Extract value estimate from value head
                valueEstimate = clamp01(response.getWinProbability());
            } else {
                valueEstimate = valueEstimateFromHeuristic();
            }
        } catch (Exception e) {
            valueEstimate = valueEstimateFromHeuristic();
        }

        evaluated = true;
        return valueEstimate;
    }

    /**
     * Estimate value using a heuristic when the neural service is unavailable.
     * Used as fallback and for unselected leaf nodes.
     *
     * @return estimated win probability based on board position
     */
    private double valueEstimateFromHeuristic() {
        if (isWon(state)) {
            return 1.0;
        }
        int score = AlphaSolitairePlayer.heuristicScore(state);
        double normalized = 1.0 / (1.0 + Math.exp(-score / 100.0));
        return normalized;
    }

    /**
     * Create a copy of the current game state for exploration.
     *
     * @return a new Solitaire instance with the same board configuration
     */
    public Solitaire copyState() {
        return state.copy();
    }

    /**
     * Get the move at the given index.
     *
     * @param index move index (0-based)
     * @return the move command string
     */
    public String moveAt(int index) {
        return moves.get(index);
    }

    /**
     * Get the child node at the given index (may be null if not yet expanded).
     *
     * @param index move index (0-based)
     * @return the child node, or null if not yet created
     */
    public TreeNode child(int index) {
        return children[index];
    }

    /**
     * Set the child node at the given index (called during tree expansion).
     *
     * @param index move index (0-based)
     * @param child the new child node
     */
    public void setChild(int index, TreeNode child) {
        children[index] = child;
    }

    /**
     * Update statistics for a move with a backpropagated value.
     *
     * @param moveIndex the index of the move to update
     * @param value the value to add to the move's cumulative sum
     */
    public void updateStats(int moveIndex, double value) {
        if (moveIndex < 0 || moveIndex >= moves.size()) {
            return;
        }
        valueSums[moveIndex] += value;
        visits[moveIndex] += 1;
    }

    /**
     * Select the most promising move to explore using the PUCT algorithm.
     * Balances exploitation (high Q-value) with exploration (high prior * sqrt(N)).
     *
     * Formula: Q(a) + cpuct * P(a) * sqrt(N) / (1 + N(a))
     *   - Q(a) = average value of the move (exploited rewards)
     *   - P(a) = prior probability from policy head
     *   - N = total visits to parent
     *   - N(a) = visits to this move
     *
     * @param cpuct the PUCT exploration constant (balance parameter)
     * @return index of the selected move, or -1 if no moves available
     */
    public int selectChildIndex(double cpuct) {
        int nMoves = moves.size();
        if (nMoves == 0) {
            return -1;
        }

        int totalVisits = 0;
        for (int n : visits) {
            totalVisits += n;
        }

        double bestScore = Double.NEGATIVE_INFINITY;
        int bestIndex = 0;

        for (int i = 0; i < nMoves; i++) {
            double p = priors[i];
            int n = visits[i];
            double q = n == 0 ? 0.0 : valueSums[i] / n;
            double u = cpuct * p * Math.sqrt(totalVisits + 1e-6) / (1 + n);
            double score = q + u;
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    /**
     * Select the best move to play based on visit counts alone (exploitation only).
     * Returns the move with the highest number of visits from MCTS.
     *
     * @return the best move command, or null if no moves available
     */
    public String bestMove() {
        int nMoves = moves.size();
        if (nMoves == 0) {
            return null;
        }

        int bestIndex = 0;
        int bestVisits = visits[0];

        for (int i = 1; i < nMoves; i++) {
            if (visits[i] > bestVisits) {
                bestVisits = visits[i];
                bestIndex = i;
            }
        }

        return moves.get(bestIndex);
    }

    /**
     * Get the visit count for a specific move by command string.
     *
     * @param move the move command string to look up
     * @return the number of times this move has been visited in MCTS
     */
    public int visitsForMove(String move) {
        if (move == null) {
            return 0;
        }
        String normalized = move.trim();
        for (int i = 0; i < moves.size(); i++) {
            if (normalized.equals(moves.get(i).trim())) {
                return visits[i];
            }
        }
        return 0;
    }

    /**
     * Clamp a value to the range [0.0, 1.0].
     *
     * @param v the value to clamp
     * @return the clamped value
     */
    private static double clamp01(double v) {
        if (v < 0.0) {
            return 0.0;
        }
        if (v > 1.0) {
            return 1.0;
        }
        return v;
    }

    /**
     * Check if the game is won from the given state.
     *
     * @param solitaire the game state
     * @return true if all 52 cards are in the foundation piles
     */
    private static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }
}

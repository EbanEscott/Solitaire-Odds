package ai.games.player.ai.alpha;

import ai.games.game.Solitaire;
import ai.games.player.ai.tree.TreeNode;
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
public class AlphaTreeNode extends TreeNode {

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
    private AlphaTreeNode(Solitaire state, List<String> moves) {
        super();
        setState(state);  // Use base class method to set state and stateKey
        this.moves = moves;
        int n = moves.size();
        this.priors = new double[n];
        this.valueSums = new double[n];
        this.visits = new int[n];
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
    public static AlphaTreeNode createRoot(Solitaire state, List<String> legalMoves) {
        return new AlphaTreeNode(state, legalMoves);
    }

    /**
     * Create a child node during tree expansion.
     * The child uses all legal moves from the new state, including "quit".
     *
     * @param state the game state after applying a move
     * @param legalMoves all legal moves from this new state
     * @return a new child node
     */
    public static AlphaTreeNode createChild(Solitaire state, List<String> legalMoves) {
        return new AlphaTreeNode(state, legalMoves);
    }

    /**
     * Check if this node is terminal (no moves or game won).
     * Terminal nodes do not need neural network evaluation; their value is definite.
     *
     * @return true if no moves are available or the game is won
     */
    @Override
    public boolean isTerminal() {
        if (moves.isEmpty()) {
            return true;
        }
        int total = 0;
        for (var pile : getState().getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    /**
     * Get the estimated value for this position.
     * This node must have been evaluated by the neural network first.
     *
     * @return estimated win probability (0.0 = loss, 1.0 = win)
     * @throws IllegalStateException if the node has not been evaluated by the neural service
     */
    public double getValueEstimate() {
        if (!evaluated) {
            throw new IllegalStateException("Node has not been evaluated by neural service; call ensureEvaluated() first");
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
        if (moves.isEmpty()) {
            valueEstimate = 0.0;
            evaluated = true;
            return valueEstimate;
        }
        int total = 0;
        for (var pile : getState().getFoundation()) {
            total += pile.size();
        }
        if (total == 52) {
            valueEstimate = 1.0;
            evaluated = true;
            return valueEstimate;
        }

        try {
            AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(getState());
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
                throw new IllegalStateException("Neural service returned null response; AlphaSolitairePlayer requires the neural service to be running");
            }
        } catch (Exception e) {
            throw new IllegalStateException("Failed to evaluate position with neural service; AlphaSolitairePlayer requires the neural service to be running", e);
        }

        evaluated = true;
        return valueEstimate;
    }

    /**
     * Create a copy of the current game state for exploration.
     *
     * @return a new Solitaire instance with the same board configuration
     */
    @Override
    public Solitaire copyState() {
        return getState().copy();
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
    public AlphaTreeNode child(int index) {
        if (index < 0 || index >= moves.size()) {
            return null;
        }
        String move = moves.get(index);
        return (AlphaTreeNode) children.get(move);
    }

    /**
     * Set the child node at the given index (called during tree expansion).
     *
     * @param index move index (0-based)
     * @param child the new child node
     */
    public void setChild(int index, AlphaTreeNode child) {
        if (index < 0 || index >= moves.size()) {
            return;
        }
        String move = moves.get(index);
        children.put(move, child);
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
}

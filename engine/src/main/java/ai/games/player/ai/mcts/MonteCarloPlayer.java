package ai.games.player.ai.mcts;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Monte Carlo Tree Search (MCTS) player for Klondike Solitaire.
 *
 * <h2>Algorithm Overview</h2>
 * <p>This player uses Monte Carlo Tree Search to evaluate moves. For each decision:
 * <ul>
 *   <li><b>Tree Building:</b> Runs multiple MCTS iterations (default: 256)
 *   <li><b>Move Selection:</b> Chooses the most-visited child node after tree building
 *   <li><b>Fallback:</b> If no children were explored, returns the first available move
 * </ul>
 *
 * <h2>MCTS Phases</h2>
 * <p>Each iteration of MCTS consists of four phases:
 * <ol>
 *   <li><b>Selection:</b> Navigate from root down the tree using UCB policy, expanding one new node
 *   <li><b>Expansion:</b> Create a child node for an unexplored move from the selected node
 *   <li><b>Simulation:</b> Run a random playout from the expanded node to a terminal state
 *   <li><b>Backpropagation:</b> Update visit counts and reward sums along the path back to root
 * </ol>
 *
 * <h2>Move Selection</h2>
 * <p>After MCTS completes, the root's children are evaluated:
 * <ul>
 *   <li>Most-visited child is selected (exploitation: best empirically-evaluated move)
 *   <li>If tie, first move in iteration order is chosen
 *   <li>If no children explored, returns first legal move (safety fallback)
 * </ul>
 *
 * <h2>Configuration</h2>
 * <p>Key parameters can be tuned for different playing strengths:
 * <ul>
 *   <li>{@code MCTS_ITERATIONS}: Number of tree-building iterations (default 256)
 *   <li>{@code EXPLORATION_CONSTANT}: UCB exploration weight (default {@link #EXPLORATION_CONSTANT})
 * </ul>
 */
@Component
@Profile("ai-mcts")
public class MonteCarloPlayer extends AIPlayer {

    private static final Logger log = LoggerFactory.getLogger(MonteCarloPlayer.class);

    /**
     * Number of MCTS iterations (tree-building loops) per decision.
     * Higher values = stronger play but slower decisions.
     */
    private static final int MCTS_ITERATIONS = 256;

    /**
     * Exploration constant for UCB formula: c * sqrt(ln(parent_visits) / child_visits).
     * sqrt(2) ≈ 1.414 is a common default balancing exploration vs exploitation.
     */
    private static final double EXPLORATION_CONSTANT = Math.sqrt(2);

    /**
     * Root of the persistent MCTS tree for this game session.
     */
    private MonteCarloTreeNode root;

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        log.debug(LegalMovesHelper.listLegalMoves(  solitaire).toString(    ));
        // Initialise root on first call
        if (root == null) {
            root = new MonteCarloTreeNode();
            root.setState(solitaire);
        }

        // Run MCTS tree building
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < MCTS_ITERATIONS; i++) {
            MonteCarloTreeNode leaf = selectAndExpand(root);
            log.debug("Selected and expanded node: {}", leaf);

            // Simulate and backpropagate
            if (leaf != null) {
                double reward = simulate(leaf);
                log.debug("Simulation reward: {}", reward);
                backpropagate(leaf, reward);
            }
        }
        long elapsedMs = System.currentTimeMillis() - startTime;

        // Select the best move from the root after selection, expansion, simulation, and backpropagation.
        String bestMove = null;
        MonteCarloTreeNode bestChild = null;
        int bestVisits = -1;
        for(String move: root.getChildren().keySet()) {
            MonteCarloTreeNode child = (MonteCarloTreeNode) root.getChildren().get(move);
            int visits = child.getVisits();
            if(visits > bestVisits) {
                bestVisits = visits;
                bestChild = child;
                bestMove = move;
            }
        }

        // Advance root to selected child for next decision
        root = bestChild;
        root.setState(solitaire);
        root.setParent(null);

        if(log.isDebugEnabled()) {
            log.debug(
                "MCTS decision: {} iterations in {}ms, selected move: {}",
                MCTS_ITERATIONS,
                elapsedMs,
                bestMove
            );
        }

        return bestMove;
    }

    /**
     * MCTS Phase 1 & 2: Selection and Expansion.
     *
     * <p>Navigates from root down the tree using UCB policy, stopping at the first
     * node with unexplored moves. Creates and returns a new child node for one of those moves.
     *
     * <p>If the given node is terminal (game won or no moves), returns null to skip this iteration.
     *
     * @param node current node to select/expand from
     * @return the newly-expanded child node, or null if terminal
     */
    private MonteCarloTreeNode selectAndExpand(MonteCarloTreeNode node) {
        MonteCarloTreeNode current = node;

        // Navigate down the tree using UCB until we find an unexplored move
        while (true) {
            Solitaire state = current.getState();
            if (state == null) {
                return null; // Safety check
            }

            // Check if we have won
            if (current.isWon()) {
                return null;
            }

            // Get legal moves from this state (but remove quit)
            List<String> moves = LegalMovesHelper.listLegalMoves(state);
            moves.removeIf(m -> m.equalsIgnoreCase("quit"));
            if (moves.isEmpty()) {
                return null;
            }

            // Find unexplored children
            boolean hasUnexplored = false;
            for (String move : moves) {
                if (!current.getChildren().containsKey(move)) {
                    hasUnexplored = true;
                    break;
                }
            }

            if (hasUnexplored) {
                // Expand: create a new child for the first unexplored move
                return expandFirstUnexploredMove(current, moves);
            } else {
                // All children explored; use UCB to select best child for deeper exploration
                MonteCarloTreeNode bestChild = selectBestChildByUCB(current, moves);
                if (bestChild == null) {
                    return null; // Safety fallback
                }
                current = bestChild;
            }
        }
    }

    /**
     * MCTS Phase 3: Simulation (Playout).
     *
     * <p>Runs a random playout from the given node to a terminal state.
     * Returns a reward based on the heuristic evaluation of the final state.
     *
     * <p>The reward is normalised: -1.0 for loss (no progress), 1.0 for win (all foundation cards).
     *
     * @param node the node to simulate from
     * @return reward in [-1.0, 1.0]
     */
    private double simulate(MonteCarloTreeNode node) {
        Solitaire state = node.getState();
        if (state == null) {
            throw new IllegalStateException("Cannot simulate from null state");
        }

        // Skip undesirable nodes
        if(node.isPruned() || node.isQuit() || node.isUselessKingMove() || node.isCycleDetected()) {
            return -1.0; // No rewards from undesirable nodes
        }

        Solitaire simulation = state.copy();
        int maxSteps = 16; // Prevent infinite loops in playout

        for (int step = 0; step < maxSteps; step++) {
            // Check won condition
            int foundationCards = 0;
            for (var pile : simulation.getFoundation()) {
                foundationCards += pile.size();
            }
            if (foundationCards == 52) {
                return 1.0; // Win
            }

            // Get legal moves (but remove quit)
            List<String> moves = LegalMovesHelper.listLegalMoves(simulation);
            moves.removeIf(m -> m.equalsIgnoreCase("quit"));
            if (moves.isEmpty()) {
                break; // No more moves
            }

            // Random move selection (uniformly from legal moves)
            int randomIndex = (int) (Math.random() * moves.size());
            String moveCommand = moves.get(randomIndex);
            
            // Parse and apply the move
            if (moveCommand.equalsIgnoreCase("turn")) {
                simulation.turnThree();
            } else if (moveCommand.toLowerCase().startsWith("move")) {
                String[] parts = moveCommand.split("\\s+");
                if (parts.length == 4) {
                    simulation.attemptMove(parts[1], parts[2], parts[3]);
                }
            }
        }

        // Evaluate final state using node's heuristic
        MonteCarloTreeNode evalNode = new MonteCarloTreeNode();
        evalNode.setState(simulation);
        int score = evalNode.evaluate();

        // Normalise score to [-1.0, 1.0]
        double MAX_SCORE = 3000.0; //Approx max achievable score
        double clamped = Math.max(-MAX_SCORE, Math.min(MAX_SCORE, score));
        return clamped / MAX_SCORE;
    }

    /**
     * MCTS Phase 4: Backpropagation.
     *
     * <p>Updates visit counts and accumulated rewards from the given node
     * up to the root, ensuring statistics propagate through the entire path.
     *
     * @param node the leaf node to backpropagate from
     * @param reward the reward to accumulate
     */
    private void backpropagate(MonteCarloTreeNode node, double reward) {
        MonteCarloTreeNode current = node;
        while (current != null) {
            current.updateStats(reward);
            current = (MonteCarloTreeNode) current.getParent();
        }
    }

    /**
     * Expand the first unexplored move from the given node.
     *
     * <p>Creates a child node by applying the move, initializes its state,
     * and links it back to the parent.
     *
     * @param parent the parent node
     * @param moves legal moves from parent's state
     * @return the newly-created child node
     */
    private MonteCarloTreeNode expandFirstUnexploredMove(
        MonteCarloTreeNode parent,
        List<String> moves
    ) {
        for (String moveCommand : moves) {
            if (!parent.getChildren().containsKey(moveCommand)) {
                // Create child
                Solitaire nextState = parent.getState().copy();
                
                // Apply the move
                if (moveCommand.equalsIgnoreCase("turn")) {
                    nextState.turnThree();
                } else if (moveCommand.toLowerCase().startsWith("move")) {
                    String[] parts = moveCommand.split("\\s+");
                    if (parts.length == 4) {
                        nextState.attemptMove(parts[1], parts[2], parts[3]);
                    }
                }

                MonteCarloTreeNode child = new MonteCarloTreeNode();
                child.setState(nextState);
                child.setMove(moveCommand);
                child.setParent(parent);

                parent.getChildren().put(moveCommand, child);
                return child;
            }
        }
        return null; // Should not reach here
    }

    /**
     * Select the best child node using UCB (Upper Confidence Bound).
     *
     * <p>Balances exploitation (high reward) with exploration (unvisited or less-visited nodes).
     * Returns the child with the highest UCB value.
     *
     * @param parent the parent node
     * @param moves legal moves from parent's state
     * @return the best child by UCB, or null if no children
     */
    private MonteCarloTreeNode selectBestChildByUCB(MonteCarloTreeNode parent, List<String> moves) {
        MonteCarloTreeNode best = null;
        double bestUcb = -Double.MAX_VALUE;

        for (String move : moves) {
            MonteCarloTreeNode child = (MonteCarloTreeNode) parent.getChildren().get(move);
            if (child != null) {
                double ucb = child.getUctValue(EXPLORATION_CONSTANT, parent.getVisits());
                if (ucb > bestUcb) {
                    bestUcb = ucb;
                    best = child;
                }
            }
        }

        return best;
    }
}

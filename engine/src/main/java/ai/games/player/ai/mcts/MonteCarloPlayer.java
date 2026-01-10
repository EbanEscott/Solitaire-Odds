package ai.games.player.ai.mcts;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Pure Monte Carlo Tree Search (MCTS) player for Klondike Solitaire.
 *
 * <h2>Algorithm Overview</h2>
 * <p>This player uses <b>Pure Monte Carlo search</b> (flat bandit evaluation): for each legal move,
 * it runs {@value #MCTS_ITERATIONS} random playouts to terminal states and selects the move with
 * the highest average reward. Unlike full MCTS with UCB-guided tree expansion, this approach
 * evaluates all moves equally without adaptive exploration—making it simpler but less efficient
 * for deep game trees.
 *
 * <h2>How it Works</h2>
 * <ol>
 *   <li><b>Enumerate moves:</b> Get all legal moves from the current position
 *   <li><b>Simulate:</b> For each move, play k random games to completion
 *   <li><b>Score:</b> Record the average heuristic evaluation from those playouts
 *   <li><b>Select:</b> Choose the move with the best average score
 * </ol>
 *
 * <p>This converges to optimal play as iterations → ∞, but requires many more samples than
 * UCB-based MCTS for the same decision quality.
 *
 * <h2>Configuration</h2>
 * <p>Key parameters can be tuned for different playing strengths:
 * <ul>
 *   <li>{@code MCTS_ITERATIONS}: Number of random playouts per move (default {@value #MCTS_ITERATIONS})
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
    private static final int MCTS_ITERATIONS = 64;

    /**
     * Maximum simulation steps per playout to avoid infinite loops.
     */
    private static final int MAX_SIMULATION_STEPS = 128;

    /**
     * Persisted MCTS tree root across turns.
     */
    private MonteCarloTreeNode root;

    /**
     * Node corresponding to the current Solitaire state.
     */
    private MonteCarloTreeNode current;

    /**
     * Expected state key after the previously selected move is applied.
     * Used to detect state drift when callers forget to apply the AI's chosen move.
     */
    private Long expectedNextStateKey;

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        // Validate that the caller applied the previously selected move.
        // In PLAN mode, card identities may be masked/guessed, so the state key is not stable;
        // we skip validation and clear the expectation.
        if (expectedNextStateKey != null) {
            if (solitaire.getMode() == Solitaire.GameMode.PLAN) {
                expectedNextStateKey = null;
            } else {
                long actualKey = solitaire.getStateKey();
                if (actualKey != expectedNextStateKey) {
                    throw new IllegalStateException(
                            "State drift detected: expected stateKey=" + expectedNextStateKey + " but was " + actualKey);
                }
                expectedNextStateKey = null;
            }
        }

        if (log.isTraceEnabled()) {
            log.trace(LegalMovesHelper.listLegalMoves(solitaire).toString());
        }

        // Initialise or update MCTS tree
        if (root == null || current == null) {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire.copy());
            node.setParent(null);
            this.root = node;
            this.current = node;
        } else {
            current.setState(solitaire.copy());
            current.getChildren().clear();
            current.resetStats();
        }

        if(log.isTraceEnabled()) {
            log.trace("MCTS starting. Root has {} children that include {}", current.getChildren().size(), current.getChildren().keySet());
        }

        // Phase 1 and 2: Selection and Expansion.
        long startTime = System.currentTimeMillis();
        List<String> legalMoves = LegalMovesHelper.listLegalMoves(current.getState());
        legalMoves.removeIf(m -> m.equalsIgnoreCase("quit"));
        for(String move : legalMoves) {
            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(current);
            current.addChild(move, child);
            child.setState(solitaire.copy());
            child.applyMove(move);

            // In pure MCTS, all children have the same number of simulations
            for (int i = 0; i < MCTS_ITERATIONS; i++) {
                double reward = simulate(child);
                if (log.isTraceEnabled()) {
                    log.trace("Simulation reward: {}", reward);
                }
                // Phase 4: Backpropagation (one level is needed only as children are directly under current)
                child.updateStats(reward);

                // If we hit the sentinel penalty (-1.0), the move is undesirable (quit/useless/cycle).
                // Running more rollouts for this move cannot improve its mean reward, so stop early.
                if (reward == -1.0) {
                    break;
                }
            }
        }
        long elapsedMs = System.currentTimeMillis() - startTime;
        if(log.isDebugEnabled()) {
            log.debug("MCTS completed. Root has {} children after {} iterations:", current.getChildren().size(), MCTS_ITERATIONS);
        }

        // Select best move
        String bestMove = null;
        MonteCarloTreeNode bestChild = null;
        double bestMeanReward = -1.0;
        for (String move : current.getChildren().keySet()) {
            MonteCarloTreeNode child = (MonteCarloTreeNode) current.getChildren().get(move);
            int visits = child.getVisits();
            double meanReward = child.getMeanReward();
            
            if(log.isDebugEnabled()) {
                log.debug("  '{}': visits={}, meanReward={}", move, visits, String.format("%.4f", meanReward));
            }
            
            if(meanReward > bestMeanReward) {
                bestMeanReward = meanReward;
                bestChild = child;
                bestMove = move;
            }
        }

        // If MCTS didn't expand anything (or we're in a terminal/no-moves situation), fall back safely.
        if (bestMove == null) {
            if(log.isDebugEnabled()) {
                log.debug("MCTS found no explored moves; falling back to quit.");
            }
            return "quit";
        }

        // Advance current to selected child for next decision (root stays at the initial game state)
        current = bestChild;

        // Record the state we expect to see on the next call (after the move is applied).
        expectedNextStateKey = current.getState() != null ? current.getState().getStateKey() : null;

        if(log.isDebugEnabled()) {
            log.debug("MCTS selected '{}' with {} visits (highest mean reward)", bestMove, bestMeanReward);
        }

        if(log.isTraceEnabled()) {
            log.trace(
                "MCTS decision: {} iterations in {}ms, selected move: {}",
                MCTS_ITERATIONS,
                elapsedMs,
                bestMove
            );
        }

        return bestMove;
    }

    /**
     * Phase 3: Simulation (Playout).
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
        if(log.isTraceEnabled()) {
            log.trace("Simulating move: {} from parent: {}", node.getMove(), node.getParent().getMove() != null ? node.getParent().getMove() : "ROOT");
        }

        // Skip quitting
        if(node.isQuit()) {
            if(log.isTraceEnabled()) {
                log.trace("Quitting reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        // Skip useless king move
        if(node.isUselessKingMove()) {
            if(log.isTraceEnabled()) {
                log.trace("Useless king move reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        // Skip cycle detected
        if(node.isCycleDetected()) {
            if(log.isTraceEnabled()) {
                log.trace("Cycle detected reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        // Randmonly play out a game from this node
        MonteCarloTreeNode leaf = new MonteCarloTreeNode();
        leaf.setState(node.getState().copy());
        leaf.setParent(node);
        // The possible moves from current state
        List<MonteCarloTreeNode> possibles = new ArrayList<>();
        int steps = 0;
        while (!leaf.isTerminal()) {
            List<String> legalMoves = LegalMovesHelper.listLegalMoves(leaf.getState());
            for(String move : legalMoves) {
                MonteCarloTreeNode possible = new MonteCarloTreeNode();
                possible.setParent(leaf);
                possible.setState(leaf.getState().copy());
                possible.applyMove(move);

                // Filter out undesirable moves
                if(!possible.isQuit() &&!possible.isCycleDetected() && !possible.isUselessKingMove()) {
                    possibles.add(possible);
                }
            }

            // Terminal state: no productive moves available (all filtered out).
            // Treat this as the end of the rollout and evaluate from here.
            if(possibles.isEmpty()) {
                break;
            }

            // Randomly select one of the possible moves
            int idx = (int)(Math.random() * possibles.size());
            MonteCarloTreeNode selected = possibles.get(idx);
            if(log.isTraceEnabled()) {
                List<String> moveStrings = possibles.stream().map(MonteCarloTreeNode::getMove).toList();
                log.trace("Move: {} [{}]", selected.getMove(), moveStrings);
            }
            possibles.clear();
            // Advance leaf but keep building the tree for cycle detection
            leaf.addChild(selected.getMove(), selected);
            leaf = selected;
            steps++;
            if(steps >= MAX_SIMULATION_STEPS) {
                if(log.isTraceEnabled()) {
                    log.trace("Maximum simulation steps {} reached, terminating playout.", MAX_SIMULATION_STEPS);
                }
                break;
            }   
        }


        // Get the score and normalise it
        double score = leaf.evaluate();
        double clamped = Math.max(-MonteCarloTreeNode.MAX_SCORE, Math.min(MonteCarloTreeNode.MAX_SCORE, score));
        double normalised = clamped / MonteCarloTreeNode.MAX_SCORE;
        
        if(log.isTraceEnabled()) {
            log.trace("Normalised reward: {}", String.format("%.4f", normalised));
        }
        
        return normalised;
    }
}

package ai.games.player.ai.mcts;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Pure Monte Carlo Tree Search (MCTS) player for Klondike Solitaire.
 *
 * <h2>Algorithm Overview</h2>
 * <p>This player uses <b>Pure Monte Carlo search</b> (flat bandit evaluation): for each legal move,
 * it runs up to {@value #MCTS_ITERATIONS} bounded random playouts and selects the move with
 * the highest average reward. Unlike full MCTS with UCB-guided tree expansion, this approach
 * evaluates all moves equally without adaptive exploration—making it simpler but less efficient
 * for deep game trees.
 *
 * <h2>How it Works</h2>
 * <ol>
 *   <li><b>Enumerate moves:</b> Get all legal moves from the current position
 *   <li><b>Simulate:</b> For each move, run bounded random playouts on a planning board
 *   <li><b>Score:</b> Record the average heuristic evaluation from those playouts
 *   <li><b>Select:</b> Choose the move with the best average score
 * </ol>
 *
 * <p>Revealed UNKNOWN identities are sampled without replacement in each rollout. Rewards
 * estimate the performance of the rollout policy; increasing the sample count does not
 * guarantee optimal play.
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
     * Expected planning position after the previously selected move is applied.
     * Used to detect state drift when callers forget to apply the AI's chosen move.
     */
    private Solitaire expectedNextState;

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        Solitaire planning = solitaire.planningCopy();
        // Predicted UNKNOWN slots may now have real identities, but known cards and pile shapes must match.
        if (expectedNextState != null && !expectedNextState.matchesObservation(planning)) {
            throw new IllegalStateException("State drift detected: the selected move was not applied");
        }
        expectedNextState = null;

        if (log.isTraceEnabled()) {
            log.trace(LegalMovesHelper.listLegalMoves(solitaire).toString());
        }

        // Initialise or update MCTS tree
        if (root == null || current == null) {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(planning);
            node.setParent(null);
            this.root = node;
            this.current = node;
        } else {
            current.setState(planning);
            current.getChildren().clear();
            current.resetStats();
        }

        if(log.isTraceEnabled()) {
            log.trace("MCTS starting. Root has {} children that include {}", current.getChildren().size(), current.getChildren().keySet());
        }

        // Phase 1 and 2: Selection and Expansion.
        long startTime = System.currentTimeMillis();
        List<Move> legalMoves = LegalMovesHelper.listMoves(current.getState());
        legalMoves.removeIf(Move::isQuit);
        for (Move move : legalMoves) {
            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(current);
            current.addChild(move.toCommandString(), child);
            child.setState(planning.copy());
            child.applyMove(move);

            // Reuse this candidate board for every rollout; simulate restores it before returning.
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
        expectedNextState = current.getState();

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
     * The candidate board is reused and rolled back before returning; no rollout copies are made.
     *
     * <p>The reward is normalised: -1.0 for loss (no progress), 1.0 for win (all foundation cards).
     *
     * @param node the node to simulate from
     * @return reward in [-1.0, 1.0]
     */
    private double simulate(MonteCarloTreeNode node) {
        return simulate(node, ThreadLocalRandom.current(), MAX_SIMULATION_STEPS);
    }

    // Explicit randomness and step budget also allow deterministic, bounded rollout tests.
    /**
     * Runs a bounded rollout using explicit randomness for move and reveal selection.
     * <p>
     * Samples revealed identities without replacement, then generates and prunes ordinary
     * legal moves. The candidate's board is reused and restored to its starting history depth
     * in a {@code finally} block, including when sampling or move selection throws.
     *
     * @param node the candidate node whose board is temporarily explored
     * @param random the generator used for move selection and speculative reveals
     * @param maxSteps the maximum number of accepted game moves, excluding assignments
     * @return the normalised heuristic reward, or {@code -1.0} for a pruned candidate
     */
    private double simulate(MonteCarloTreeNode node, RandomGenerator random, int maxSteps) {
        if(log.isTraceEnabled()) {
            log.trace(
                "Simulating move: {} from parent: {}",
                node.getMove(),
                node.getParent() != null && node.getParent().getMove() != null
                        ? node.getParent().getMove() : "ROOT");
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

        Solitaire board = node.getState();
        int rolloutStart = board.getMoveHistorySize();
        // This temporary node operates on the candidate's board. Its path stores keys only;
        // the retained candidate node keeps its original move, key, parent, and statistics.
        MonteCarloTreeNode position = new MonteCarloTreeNode(board);
        position.setParent(node);
        position.setMove(node.getMove());
        List<Long> pathKeys = new ArrayList<>();
        // Preserve the old rollout-root occurrence as well as the retained candidate ancestor.
        pathKeys.add(board.getStateKey());
        try {
            resolveRevealedCards(board, random);
            position.setState(board);
            for (int steps = 0; steps < maxSteps && !position.isWon(); steps++) {
                // Generate and prune against the current board before applying candidates.
                List<Move> legalMoves = LegalMovesHelper.listMoves(board);
                List<Move> candidates = new ArrayList<>(legalMoves.size());
                for (Move candidate : legalMoves) {
                    if (candidate.isQuit()
                            || position.isInverseOfLastMove(candidate)
                            || position.isUselessKingMove(candidate)
                            || position.isSimilarSibling(candidate, candidates)) {
                        continue;
                    }
                    candidates.add(candidate);
                }

                // Try candidates without replacement, reversing a rejected move before retrying.
                int stepStart = board.getMoveHistorySize();
                Move previousMove = position.getMove();
                Move selected = null;
                while (!candidates.isEmpty()) {
                    int index = random.nextInt(candidates.size());
                    Move candidate = candidates.get(index);
                    Move last = candidates.remove(candidates.size() - 1);
                    if (index < candidates.size()) {
                        candidates.set(index, last);
                    }

                    position.applyMove(candidate);
                    resolveRevealedCards(board, random);
                    position.setState(board);
                    if (position.isCycleDetected(pathKeys)) {
                        board.rollbackTo(stepStart);
                        position.setState(board);
                        position.setMove(previousMove);
                        continue;
                    }
                    selected = candidate;
                    break;
                }

                if (selected == null) {
                    break;
                }
                pathKeys.add(board.getStateKey());
                if (log.isTraceEnabled()) {
                    log.trace("Rollout move: {}", selected);
                }
            }

            double score = position.evaluate();
            double clamped = Math.max(-MonteCarloTreeNode.MAX_SCORE,
                    Math.min(MonteCarloTreeNode.MAX_SCORE, score));
            double normalised = clamped / MonteCarloTreeNode.MAX_SCORE;
            if (log.isTraceEnabled()) {
                log.trace("Normalised reward: {}", String.format("%.4f", normalised));
            }
            return normalised;
        } finally {
            // Also restore after a terminal position, exhausted candidates, or an exception.
            board.rollbackTo(rolloutStart);
        }
    }

    /**
     * Samples identities for all currently revealed UNKNOWN slots on the rollout board.
     * <p>
     * Each identity is chosen uniformly from the remaining unseen pool and committed through
     * {@link Solitaire#assumeCard(Card, Card)}. Later assignments cannot reuse that identity.
     * The enclosing rollout restores these assignments through the board's action history.
     *
     * @param board the board being simulated; GAME boards have no unresolved planning slots
     * @param random the generator used to sample identities without replacement
     * @throws IllegalStateException if an UNKNOWN slot remains but no unseen identity is available
     */
    private void resolveRevealedCards(Solitaire board, RandomGenerator random) {
        Card unknown;
        while ((unknown = board.nextUnresolvedCard()) != null) {
            List<Card> unseen = board.getUnknownCards();
            if (unseen.isEmpty()) throw new IllegalStateException("No unseen identity for a revealed slot");
            board.assumeCard(unknown, unseen.get(random.nextInt(unseen.size())));
        }
    }

}

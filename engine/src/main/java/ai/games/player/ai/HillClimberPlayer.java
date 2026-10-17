package ai.games.player.ai;

import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Stochastic hill-climbing over immediate successors only.
 *
 * <p>Evaluate each legal neighbor once using observed identities. Choose the highest
 * strict improvement, breaking ties with seeded randomness. Otherwise prefer up to
 * eight consecutive equal-score sideways moves, then take one random eligible local
 * perturbation. Perturbations are not restarts or backtracking and do not reset the deal.
 * The overall 5,000-action budget bounds exploration without imposing a stock-pass limit.
 *
 * <p>Real observed positions and attempted actions prevent repeated cycles. Predictions
 * containing unknown identities are not recorded as observations. Create one player per
 * game or call {@link #resetGame()} before reuse.
 */
@Component
@Profile("ai-hill")
public class HillClimberPlayer extends AIPlayer implements Player {
    /** Overall action budget, including improving, sideways, and perturbation moves. */
    private static final int MAX_STEPS_PER_GAME = 5000;
    /** Equal-score moves preferred before allowing a random local perturbation. */
    private static final int MAX_SIDEWAYS_STEPS = 8;
    /** Seed retained for reproducible resets. */
    private final long seed;
    /** Source of tie-breaking and local perturbation choices. */
    private final Random random;
    /** Actual positions observed in this game. */
    private final Set<Long> visitedStates = new HashSet<>();
    /** Actions already selected at each observed position, including unresolved predictions. */
    private final Map<Long, Set<String>> attemptedActions = new HashMap<>();
    /** Number of actions selected during this game. */
    private int stepsTaken;
    /** Number of consecutive preferred sideways moves since improvement or perturbation. */
    private int sidewaysSteps;

    /** Creates a player with a fresh seed. */
    public HillClimberPlayer() {
        this(System.nanoTime());
    }

    /** Creates a player whose tie-breaking and perturbations are reproducible. */
    public HillClimberPlayer(long seed) {
        /** Keep the initial seed alongside its mutable random stream. */
        this.seed = seed;
        this.random = new Random(seed);
    }

    /** Clears game history, action budgets, and seeded randomness before another deal. */
    public void resetGame() {
        /** Reset all instance-owned exploration state together. */
        visitedStates.clear();
        attemptedActions.clear();
        stepsTaken = 0;
        sidewaysSteps = 0;
        random.setSeed(seed);
    }

    /** Selects a local improvement, sideways step, or eligible perturbation without deeper search. */
    @Override
    public String nextCommand(Solitaire solitaire, String recommendedMoves, String feedback) {
        /** The global budget bounds random exploration without treating draws as failed progress. */
        if (stepsTaken >= MAX_STEPS_PER_GAME) {
            return "quit";
        }
        Solitaire board = solitaire.planningCopy();
        long key = board.getStateKey();
        visitedStates.add(key);
        Set<String> attempted = attemptedActions.computeIfAbsent(key, ignored -> new HashSet<>());
        int currentScore = evaluate(board);
        List<Neighbor> eligible = new ArrayList<>();
        List<Neighbor> best = new ArrayList<>();
        List<Neighbor> sideways = new ArrayList<>();
        int bestScore = currentScore;

        /** Generate, simulate, and score each eligible legal action only once. */
        for (Move move : LegalMovesHelper.listMoves(board)) {
            /** Quit is a termination decision; repeated observed actions cannot supply new exploration. */
            if (move.isQuit() || attempted.contains(move.toCommandString())) {
                continue;
            }
            Solitaire next = board.copy();
            applyMove(next, move);
            /** Reject no-ops and known real positions without storing speculative unknown states. */
            if (visitedStates.contains(next.getStateKey())) {
                continue;
            }
            Neighbor neighbor = new Neighbor(move.toCommandString(), evaluate(next));
            eligible.add(neighbor);
            /** Keep all strict best improvements so ties can be resolved reproducibly. */
            if (neighbor.score > bestScore) {
                bestScore = neighbor.score;
                best.clear();
            }
            /** Preserve all equally best improving neighbors for seeded tie-breaking. */
            if (neighbor.score == bestScore && bestScore > currentScore) {
                best.add(neighbor);
            }
            /** Sideways candidates are distinct from downhill local perturbations. */
            if (neighbor.score == currentScore) {
                sideways.add(neighbor);
            }
        }
        /** Inspect the complete eligible set before concluding that no usable neighbor exists. */
        if (eligible.isEmpty()) {
            return "quit";
        }

        Neighbor selected;
        /** Strict local improvements always take precedence over plateau exploration. */
        if (!best.isEmpty()) {
            selected = choose(best);
            sidewaysSteps = 0;
        } else if (!sideways.isEmpty() && sidewaysSteps < MAX_SIDEWAYS_STEPS) {
            /** Prefer a bounded run of neutral moves before permitting a downhill perturbation. */
            selected = choose(sideways);
            sidewaysSteps++;
        } else {
            /** Sample once from valid neighbors, never repeatedly from rejected legal moves. */
            selected = choose(eligible);
            sidewaysSteps = 0;
        }
        attempted.add(selected.command);
        stepsTaken++;
        return selected.command;
    }

    /** Selects uniformly from an already filtered nonempty list without rejection retries. */
    private Neighbor choose(List<Neighbor> neighbors) {
        /** Filtering is complete, so every possible selected index is valid. */
        return neighbors.get(random.nextInt(neighbors.size()));
    }

    /** Applies one structured legal command to an isolated planning copy. */
    private void applyMove(Solitaire board, Move move) {
        /** Configured stock turns include recycling when legal. */
        if (move.isTurn()) {
            board.turnStock();
        } else {
            /** All legal card categories, including worry-back, participate in the local neighborhood. */
            board.moveCard(move.from().toCode(), move.card() == null ? null : move.card().shortName(),
                    move.to().toCode());
        }
    }

    /**
     * Scores foundation deployment and physical tableau exposure using observable counts.
     *
     * <p>Weights remain 20 per foundation card, 5 per face-up tableau card, and minus 3
     * per face-down tableau card. The undeployed-card cost is minus 1 per stock or waste
     * card, so drawing and recycling alone are neutral rather than artificial progress.
     */
    private int evaluate(Solitaire board) {
        /** Deployment matters; merely relocating cards between stock and waste does not. */
        int score = board.getFoundation().stream().mapToInt(List::size).sum() * 20
                - board.getStockpile().size() - board.getTalon().size();
        /** Physical exposure remains useful even when thoughtful knowledge reveals all identities. */
        for (int count : board.getTableauFaceUpCounts()) {
            score += count * 5;
        }
        /** Face-down cards remain physically blocked even in thoughtful mode. */
        for (int count : board.getTableauFaceDownCounts()) {
            score -= count * 3;
        }
        return score;
    }

    /** Cached immediate action and score; no successor is expanded beyond this one ply. */
    private static final class Neighbor {
        /** Legal command that produces this neighbor. */
        private final String command;
        /** Evaluation computed once from the copied successor. */
        private final int score;

        /** Retains the evaluated neighbor for improvement and perturbation selection alike. */
        private Neighbor(String command, int score) {
            /** Cache only what selection needs, releasing copied boards after evaluation. */
            this.command = command;
            this.score = score;
        }
    }
}

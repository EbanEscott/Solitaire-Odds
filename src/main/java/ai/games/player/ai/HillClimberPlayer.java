package ai.games.player.ai;

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
 * Hill-climbing search player driven by the opaque {@link Solitaire#getStateKey()}.
 *
 * <p>Algorithm sketch:
 * <ul>
 *     <li>At each step, enumerate all legal moves using {@link LegalMovesHelper} (including "turn" and "quit").</li>
 *     <li>Simulate each non-quit move on a copy of the board and score the resulting state.</li>
 *     <li>Track best score seen for each {@code stateKey} to avoid revisiting strictly worse states.</li>
 *     <li>Choose a neighbour with the highest score; if no improving neighbour exists, perform a random
 *     restart by picking a random non-quit move.</li>
 *     <li>If several moves share the best score, pick one at random (deterministically for a fixed seed).</li>
 *     <li>Quit if no legal non-quit moves exist or a safety step limit is exceeded.</li>
 * </ul>
 *
 * <p>Backtracking is implemented logically via the {@code bestScoreByState} map: when we simulate a move that
 * leads into a state whose score is lower than the best known score for that {@code stateKey}, that neighbour
 * is discarded and the search continues exploring other branches from the current state. Random restarts use
 * the same neighbourhood but ignore improvement requirements, allowing escape from local maxima or plateaus.
 */
@Component
@Profile("ai-hill")
public class HillClimberPlayer extends AIPlayer implements Player {

    private static final int MAX_STEPS_PER_GAME = 5000;
    private static final int MAX_RESTARTS_PER_POSITION = 8;

    private final Random random;

    // Tracks the best score we have seen for a given state key across the current game.
    private final Map<Long, Integer> bestScoreByState = new HashMap<>();

    // Tracks states we've already visited to help guard against simple cycles.
    private final Set<Long> visitedStates = new HashSet<>();

    private int stepsTaken = 0;

    public HillClimberPlayer() {
        this(System.nanoTime());
    }

    public HillClimberPlayer(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        stepsTaken++;
        if (stepsTaken > MAX_STEPS_PER_GAME) {
            return "quit";
        }

        long stateKey = solitaire.getStateKey();
        int currentScore = evaluate(solitaire);

        int previousBest = bestScoreByState.getOrDefault(stateKey, Integer.MIN_VALUE);
        if (currentScore > previousBest) {
            bestScoreByState.put(stateKey, currentScore);
        }
        visitedStates.add(stateKey);

        List<String> legalMoves = LegalMovesHelper.listLegalMoves(solitaire);
        // If quit is the only move, honour it.
        if (legalMoves.isEmpty() || (legalMoves.size() == 1 && isQuit(legalMoves.getFirst()))) {
            return "quit";
        }

        String bestMove = pickBestImprovingMove(solitaire, stateKey, currentScore, legalMoves);
        if (bestMove != null) {
            return bestMove;
        }

        // Local maximum / plateau reached: perform a bounded number of random restarts from this position.
        for (int restart = 0; restart < MAX_RESTARTS_PER_POSITION; restart++) {
            String randomMove = pickRandomNonQuitMove(legalMoves);
            if (randomMove == null) {
                break;
            }
            long nextKey = simulateStateKey(solitaire, randomMove);
            if (nextKey == stateKey) {
                continue;
            }
            if (nextKey != 0L && visitedStates.contains(nextKey)) {
                continue;
            }
            return randomMove;
        }

        // Fallback: if no suitable restart move could be found, quit to avoid infinite loops.
        return "quit";
    }

    private String pickBestImprovingMove(Solitaire solitaire, long currentKey, int currentScore, List<String> legalMoves) {
        List<String> candidates = new ArrayList<>();
        int bestDelta = 0;

        for (String move : legalMoves) {
            if (isQuit(move)) {
                continue;
            }
            long nextKey = simulateStateKey(solitaire, move);
            if (nextKey == currentKey) {
                // No-op or cycle; ignore.
                continue;
            }
            if (nextKey != 0L && visitedStates.contains(nextKey)) {
                // Simple cycle detection using the Zobrist hash.
                continue;
            }
            int nextScore = simulateScore(solitaire, move);
            int bestKnownForNext = bestScoreByState.getOrDefault(nextKey, Integer.MIN_VALUE);

            // Only consider moves that strictly improve the state or the best-known score for that stateKey.
            boolean improvesCurrent = nextScore > currentScore;
            boolean improvesKnown = nextScore > bestKnownForNext;
            if (!improvesCurrent && !improvesKnown) {
                continue;
            }

            int delta = nextScore - currentScore;
            if (delta > bestDelta) {
                candidates.clear();
                candidates.add(move);
                bestDelta = delta;
            } else if (delta == bestDelta && delta > 0) {
                candidates.add(move);
            }

            if (nextKey != 0L && nextScore > bestKnownForNext) {
                bestScoreByState.put(nextKey, nextScore);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }
        return candidates.get(random.nextInt(candidates.size()));
    }

    private String pickRandomNonQuitMove(List<String> legalMoves) {
        List<String> filtered = new ArrayList<>();
        for (String move : legalMoves) {
            if (!isQuit(move)) {
                filtered.add(move);
            }
        }
        if (filtered.isEmpty()) {
            return null;
        }
        return filtered.get(random.nextInt(filtered.size()));
    }

    private boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    private int simulateScore(Solitaire solitaire, String move) {
        Solitaire copy = solitaire.copy();
        applyMove(copy, move);
        return evaluate(copy);
    }

    private long simulateStateKey(Solitaire solitaire, String move) {
        Solitaire copy = solitaire.copy();
        applyMove(copy, move);
        return copy.getStateKey();
    }

    private void applyMove(Solitaire solitaire, String move) {
        if (move == null) {
            return;
        }
        String trimmed = move.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else {
                solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
    }

    /**
     * State evaluation that leans on the Zobrist state key for indexing but
     * uses human-readable features for scoring.
     *
     * <p>Heuristic (higher is better):
     * <ul>
     *     <li>+20 per card on foundation (push toward completion)</li>
     *     <li>+5 per visible tableau card</li>
     *     <li>-3 per facedown tableau card</li>
     *     <li>-1 per stockpile card</li>
     * </ul>
     */
    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress.
        for (var pile : solitaire.getFoundation()) {
            score += pile.size() * 20;
        }

        // Tableau visibility: reward visible, penalise hidden.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        for (int i = 0; i < faceUps.size(); i++) {
            score += faceUps.get(i) * 5;
        }
        for (int i = 0; i < faceDowns.size(); i++) {
            score -= faceDowns.get(i) * 3;
        }

        // Stockpile drag: fewer buried cards is generally better.
        score -= solitaire.getStockpile().size();

        return score;
    }
}


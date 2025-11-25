package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Monte Carlo Tree Search (MCTS)-style player for Klondike Solitaire.
 *
 * <p>Simplified algorithm:
 * <ul>
 *     <li>At each decision point, enumerate all legal non-quit moves.</li>
 *     <li>For each move, run a fixed number of random playouts from the resulting state.</li>
 *     <li>Each playout is bounded by a maximum number of steps and uses random legal moves.</li>
 *     <li>At playout end, score the terminal state using a heuristic (foundation, visibility, stock size).</li>
 *     <li>Choose the move with the highest average playout score.</li>
 * </ul>
 *
 * <p>This is closer to Monte Carlo control than a full UCT-based tree, but follows the same spirit:
 * sampling futures from candidate moves and backing up average values to pick actions.
 */
@Component
@Profile("ai-mcts")
public class MonteCarloPlayer extends AIPlayer implements Player {

    private static final int PLAYOUTS_PER_MOVE = 16;
    private static final int MAX_PLAYOUT_STEPS = 200;

    // Track last real move across instances to help block simple ping-pong moves.
    private static MoveSignature lastMove = null;
    // Track visited engine states across the current game to avoid cycling.
    private static final java.util.Set<Long> visitedStates = new java.util.HashSet<>();

    private final Random random;

    public MonteCarloPlayer() {
        this(System.nanoTime());
    }

    public MonteCarloPlayer(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        long currentKey = solitaire.getStateKey();
        visitedStates.add(currentKey);

        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        if (legal.isEmpty()) {
            return "quit";
        }
        if (legal.size() == 1) {
            String only = legal.getFirst();
            return isQuit(only) ? "quit" : only;
        }

        List<String> candidateMoves = new ArrayList<>();
        for (String move : legal) {
            long nextKey = simulateStateKey(solitaire, move);
            // Skip no-op moves or moves that lead back into already visited states.
            if (nextKey == currentKey || (nextKey != 0L && visitedStates.contains(nextKey))) {
                continue;
            }
            if (!isQuit(move)
                    && !isFromFoundation(move)
                    && !isUselessKingMove(solitaire, move)
                    && !isInverseOfLast(move)) {
                candidateMoves.add(move);
            }
        }
        if (candidateMoves.isEmpty()) {
            return "quit";
        }

        // Greedy priority: if any safe move goes to foundation, take it immediately.
        List<String> toFoundation = new ArrayList<>();
        for (String move : candidateMoves) {
            if (isToFoundation(move)) {
                toFoundation.add(move);
            }
        }
        if (!toFoundation.isEmpty()) {
            return toFoundation.getFirst();
        }

        String bestMove = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (String move : candidateMoves) {
            double avgScore = evaluateMoveByPlayouts(solitaire, move);
            if (avgScore > bestScore) {
                bestScore = avgScore;
                bestMove = move;
            }
        }

        String chosen = bestMove != null ? bestMove : candidateMoves.get(0);
        lastMove = MoveSignature.tryParse(chosen);
        return chosen;
    }

    private double evaluateMoveByPlayouts(Solitaire root, String move) {
        double total = 0.0;
        for (int i = 0; i < PLAYOUTS_PER_MOVE; i++) {
            Solitaire copy = root.copy();
            applyMove(copy, move);
            total += runPlayout(copy);
        }
        return total / PLAYOUTS_PER_MOVE;
    }

    private double runPlayout(Solitaire solitaire) {
        for (int step = 0; step < MAX_PLAYOUT_STEPS; step++) {
            if (isWon(solitaire)) {
                // Large reward for completed game.
                return evaluate(solitaire) + 1000.0;
            }
            List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
            if (legal.isEmpty()) {
                break;
            }
            // Prefer non-quit moves; if none, accept quit.
            String move = pickRandomNonQuit(legal);
            if (move == null || isQuit(move)) {
                break;
            }
            applyMove(solitaire, move);
        }
        return evaluate(solitaire);
    }

    private String pickRandomNonQuit(List<String> legal) {
        List<String> nonQuit = new ArrayList<>();
        for (String m : legal) {
            if (!isQuit(m)
                    && !isFromFoundation(m)) {
                nonQuit.add(m);
            }
        }
        if (nonQuit.isEmpty()) {
            return null;
        }
        return nonQuit.get(random.nextInt(nonQuit.size()));
    }

    private long simulateStateKey(Solitaire solitaire, String move) {
        if (move == null) {
            return 0L;
        }
        Solitaire copy = solitaire.copy();
        applyMove(copy, move);
        return copy.getStateKey();
    }

    private boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    private boolean isInverseOfLast(String command) {
        if (lastMove == null || command == null) {
            return false;
        }
        MoveSignature current = MoveSignature.tryParse(command);
        if (current == null) {
            return false;
        }
        return lastMove.isInverseOf(current);
    }

    private boolean isFromFoundation(String move) {
        if (move == null) {
            return false;
        }
        String[] parts = move.trim().split("\\s+");
        if (parts.length < 2) {
            return false;
        }
        return parts[0].equalsIgnoreCase("move")
                && parts[1].toUpperCase().startsWith("F");
    }

    private boolean isToFoundation(String move) {
        if (move == null) {
            return false;
        }
        String[] parts = move.trim().split("\\s+");
        if (parts.length < 3) {
            return false;
        }
        String dest = parts[parts.length - 1].toUpperCase();
        return dest.startsWith("F");
    }

    /**
     * Prunes moves that shift a king from a tableau pile without revealing any new card.
     * These tend to just shuffle kings between columns and cause pointless ping-ponging.
     */
    private boolean isUselessKingMove(Solitaire solitaire, String move) {
        if (move == null) {
            return false;
        }
        String trimmed = move.trim();
        String[] parts = trimmed.split("\\s+");
        if (parts.length < 3) {
            return false;
        }
        if (!parts[0].equalsIgnoreCase("move")) {
            return false;
        }
        // Do not treat king moves to foundation as useless – those are progress.
        String dest = parts[parts.length - 1].toUpperCase();
        if (dest.startsWith("F")) {
            return false;
        }
        String from = parts[1];
        if (!from.startsWith("T")) {
            return false;
        }
        int pileIndex;
        try {
            pileIndex = Integer.parseInt(from.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return false;
        }
        if (pileIndex < 0 || pileIndex >= solitaire.getVisibleTableau().size()) {
            return false;
        }

        String cardToken = parts[2];
        List<Card> tableauPile = solitaire.getVisibleTableau().get(pileIndex);
        if (tableauPile == null || tableauPile.isEmpty()) {
            return false;
        }

        Card moving = null;
        for (Card c : tableauPile) {
            if (cardToken.equalsIgnoreCase(c.shortName())) {
                moving = c;
                break;
            }
        }
        if (moving == null || moving.getRank() != Rank.KING) {
            return false;
        }

        // If there are no facedown cards beneath this pile, moving the king won't reveal anything.
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        if (pileIndex < 0 || pileIndex >= faceDowns.size()) {
            return false;
        }
        int facedownCount = faceDowns.get(pileIndex);
        return facedownCount == 0;
    }

    private static final class MoveSignature {
        final String from;
        final String to;
        final String card;

        private MoveSignature(String from, String to, String card) {
            this.from = from;
            this.to = to;
            this.card = card;
        }

        static MoveSignature tryParse(String command) {
            if (command == null) {
                return null;
            }
            String[] parts = command.trim().split("\\s+");
            if (parts.length < 3) {
                return null;
            }
            if (!parts[0].equalsIgnoreCase("move")) {
                return null;
            }
            String from = parts[1].toUpperCase();
            String card = parts[2].toUpperCase();
            String to;
            if (parts.length == 4) {
                to = parts[3].toUpperCase();
            } else if (parts.length == 3) {
                to = "";
            } else {
                return null;
            }
            return new MoveSignature(from, to, card);
        }

        boolean isInverseOf(MoveSignature other) {
            if (other == null) {
                return false;
            }
            if (!this.card.equals(other.card)) {
                return false;
            }
            return this.from.equals(other.to) && this.to.equals(other.from);
        }
    }

    private boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
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
     * Heuristic similar to other AIs: reward foundation progress, visible tableau,
     * and lightly penalise hidden cards and stock size.
     */
    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress.
        for (var pile : solitaire.getFoundation()) {
            score += pile.size() * 25;
        }

        // Tableau visibility.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        for (int i = 0; i < faceUps.size(); i++) {
            score += faceUps.get(i) * 6;
            score -= faceDowns.get(i) * 4;
        }

        // Stock drag.
        score -= solitaire.getStockpile().size();

        return score;
    }
}

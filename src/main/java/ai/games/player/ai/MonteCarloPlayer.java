package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Monte Carlo (sampling-based) player for Klondike Solitaire.
 *
 * <p>Algorithm sketch (root-level multi-armed bandit):
 * <ul>
 *     <li>Enumerate legal root moves and prune obviously bad ones:
 *         <ul>
 *             <li>never move from foundation back to tableau</li>
 *             <li>avoid king moves that do not reveal any facedown card</li>
 *             <li>avoid immediate inverse of the last real move</li>
 *         </ul>
 *     </li>
 *     <li>Allocate a fixed playout budget and treat each root move as an "arm".</li>
 *     <li>Use UCB1 to choose which move to sample next: explore rarely-sampled moves,
 *     exploit high-reward ones.</li>
 *     <li>Each playout performs a short, biased-random simulation and returns a numeric reward
 *     based on win / foundation cards / visibility / stock size.</li>
 *     <li>Play the root move with the highest average reward.</li>
 * </ul>
 *
 * <p>This is "true Monte Carlo" in the sense that decisions are driven by sampled outcomes,
 * not by a purely deterministic heuristic; but the policy inside playouts is biased to favour
 * structural progress and to avoid pointless ping-ponging.
 */
@Component
@Profile("ai-mcts")
public class MonteCarloPlayer extends AIPlayer implements Player {

    // Overall playout budget per decision (spread across moves via UCB).
    private static final int MAX_ROOT_PLAYOUTS = 48;
    // Maximum depth for a single playout.
    private static final int MAX_PLAYOUT_STEPS = 60;
    // UCB exploration constant.
    private static final double UCB_C = Math.sqrt(2.0);

    // Track last real move across decisions to avoid simple two-move ping-pong.
    private static MoveSignature lastMove = null;

    private final Random random;

    public MonteCarloPlayer() {
        this(System.nanoTime());
    }

    public MonteCarloPlayer(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        if (legal.isEmpty()) {
            return "quit";
        }
        if (legal.size() == 1) {
            String only = legal.getFirst();
            return isQuit(only) ? "quit" : only;
        }

        // Filter out clearly undesirable moves.
        List<String> candidates = new ArrayList<>();
        for (String move : legal) {
            if (isQuit(move)) {
                continue;
            }
            if (isFromFoundation(move)) {
                continue;
            }
            if (isInverseOfLast(move)) {
                continue;
            }
            if (isUselessKingMove(solitaire, move)) {
                continue;
            }
            candidates.add(move);
        }

        if (candidates.isEmpty()) {
            return "quit";
        }

        // If we have direct moves to foundation among the filtered moves, still treat them
        // as high-value: MC will tend to agree, but we can short-circuit for speed.
        String foundationMove = pickFoundationMove(candidates);
        if (foundationMove != null) {
            lastMove = MoveSignature.tryParse(foundationMove);
            return foundationMove;
        }

        int moveCount = candidates.size();
        double[] totalReward = new double[moveCount];
        int[] visits = new int[moveCount];

        int totalPlayouts = 0;
        while (totalPlayouts < MAX_ROOT_PLAYOUTS) {
            int index = selectMoveByUcb(totalReward, visits, totalPlayouts, moveCount);
            String move = candidates.get(index);

            Solitaire copy = solitaire.copy();
            applyMove(copy, move);
            double reward = runPlayout(copy);

            totalReward[index] += reward;
            visits[index] += 1;
            totalPlayouts++;
        }

        // Pick the move with the highest mean reward.
        String bestMove = candidates.get(0);
        double bestScore = visits[0] == 0 ? Double.NEGATIVE_INFINITY : totalReward[0] / visits[0];
        for (int i = 1; i < moveCount; i++) {
            if (visits[i] == 0) {
                continue;
            }
            double mean = totalReward[i] / visits[i];
            if (mean > bestScore) {
                bestScore = mean;
                bestMove = candidates.get(i);
            }
        }

        lastMove = MoveSignature.tryParse(bestMove);
        return bestMove;
    }

    private int selectMoveByUcb(double[] totalReward, int[] visits, int totalPlayouts, int moveCount) {
        // Ensure every move is tried at least once.
        for (int i = 0; i < moveCount; i++) {
            if (visits[i] == 0) {
                return i;
            }
        }
        double logTotal = Math.log(totalPlayouts);
        double best = Double.NEGATIVE_INFINITY;
        int bestIndex = 0;
        for (int i = 0; i < moveCount; i++) {
            double mean = totalReward[i] / visits[i];
            double bonus = UCB_C * Math.sqrt(logTotal / visits[i]);
            double score = mean + bonus;
            if (score > best) {
                best = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private double runPlayout(Solitaire solitaire) {
        MoveSignature lastPlayoutMove = null;
        Set<Long> localStates = new HashSet<>();
        localStates.add(solitaire.getStateKey());

        for (int step = 0; step < MAX_PLAYOUT_STEPS; step++) {
            if (isWon(solitaire)) {
                // Very large bonus for a solved game.
                return evaluate(solitaire) + 10_000.0;
            }
            List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
            if (legal.isEmpty()) {
                break;
            }

            String move = pickPlayoutMove(solitaire, legal, lastPlayoutMove, localStates);
            if (move == null || isQuit(move)) {
                break;
            }

            applyMove(solitaire, move);
            lastPlayoutMove = MoveSignature.tryParse(move);

            long key = solitaire.getStateKey();
            if (!localStates.add(key)) {
                // Simple cycle detected within playout: stop this rollout.
                break;
            }
        }

        return evaluate(solitaire);
    }

    private String pickPlayoutMove(Solitaire solitaire, List<String> legal, MoveSignature lastPlayoutMove, Set<Long> localStates) {
        List<String> moves = new ArrayList<>();
        for (String m : legal) {
            if (isQuit(m)) {
                continue;
            }
            if (isFromFoundation(m)) {
                continue;
            }
            if (lastPlayoutMove != null && lastPlayoutMove.isInverseOf(MoveSignature.tryParse(m))) {
                continue;
            }
            if (isUselessKingMove(solitaire, m)) {
                continue;
            }
            moves.add(m);
        }
        if (moves.isEmpty()) {
            return null;
        }

        // With high probability, favour moves to foundation during playout.
        List<String> toFoundation = new ArrayList<>();
        for (String m : moves) {
            if (isToFoundation(m)) {
                toFoundation.add(m);
            }
        }
        if (!toFoundation.isEmpty() && random.nextDouble() < 0.85) {
            return toFoundation.get(random.nextInt(toFoundation.size()));
        }

        return moves.get(random.nextInt(moves.size()));
    }

    private String pickFoundationMove(List<String> moves) {
        for (String m : moves) {
            if (isToFoundation(m)) {
                return m;
            }
        }
        return null;
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
     * Prunes moves that shift a king from a tableau pile to another tableau pile
     * without revealing any new card (no facedown cards underneath).
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
        String dest = parts[parts.length - 1].toUpperCase();
        if (dest.startsWith("F")) {
            // King moves to foundation are always allowed.
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

        List<List<Card>> fullTableau = solitaire.getTableau();
        if (pileIndex < 0 || pileIndex >= fullTableau.size()) {
            return false;
        }

        String cardToken = parts[2];
        List<Card> pile = fullTableau.get(pileIndex);
        if (pile == null || pile.isEmpty()) {
            return false;
        }

        Card moving = null;
        int movingIndex = -1;
        for (int i = 0; i < pile.size(); i++) {
            Card c = pile.get(i);
            if (cardToken.equalsIgnoreCase(c.shortName())) {
                moving = c;
                movingIndex = i;
                break;
            }
        }
        if (moving == null || moving.getRank() != Rank.KING) {
            return false;
        }

        // Facedown cards are those before the visible suffix.
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        if (pileIndex < 0 || pileIndex >= faceDowns.size()) {
            return false;
        }
        int facedownCount = faceDowns.get(pileIndex);
        // If there are no facedown cards underneath, moving this king will not reveal anything.
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
     * State evaluation used as a playout reward:
     * <ul>
     *     <li>strong reward for foundation progress</li>
     *     <li>reward visible tableau cards and empty columns</li>
     *     <li>penalise facedown cards and large stockpile</li>
     * </ul>
     */
    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress.
        int foundationCards = 0;
        for (var pile : solitaire.getFoundation()) {
            foundationCards += pile.size();
        }
        score += foundationCards * 40;

        // Tableau visibility.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        int emptyColumns = 0;
        for (int i = 0; i < faceUps.size(); i++) {
            int up = faceUps.get(i);
            int down = faceDowns.get(i);
            score += up * 4;
            score -= down * 9;
            if (up == 0 && down == 0) {
                emptyColumns++;
            }
        }

        // Reward empty tableau columns for king placement flexibility.
        score += emptyColumns * 20;

        // Stock drag.
        score -= solitaire.getStockpile().size() * 2;

        return score;
    }
}

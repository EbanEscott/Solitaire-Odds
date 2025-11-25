package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * A* search player for Klondike Solitaire.
 *
 * <p>At each decision point, runs a bounded A* search from the current state to choose
 * the first move on the lowest-cost (highest-evaluated) path it can find.
 *
 * <p>Cost model:
 * <ul>
 *     <li>g(n): number of moves taken so far, plus small penalties for "turn" actions.</li>
 *     <li>h(n): negative heuristic score combining foundation progress, tableau visibility,
 *     and stock size (same direction as other AIs).</li>
 *     <li>f(n) = g(n) + h(n).</li>
 * </ul>
 *
 * <p>The search is bounded by a maximum number of node expansions to keep it responsive.
 */
@Component
@Profile("ai-astar")
public class AStarPlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(AStarPlayer.class);

    // Hard cap on A* node expansions per decision to keep moves responsive.
    private static final int MAX_EXPANSIONS = 256;
    // Additional cost applied when taking a "turn" action to discourage excessive stock cycling.
    private static final int TURN_PENALTY = 2;
    // Number of recent engine states remembered to avoid short state-space cycles.
    private static final int RECENT_STATE_HISTORY = 16;
    // Number of recent (from->to) move signatures tracked per card to avoid short move orbits.
    private static final int TABU_MOVES_PER_CARD = 4;

    // Track last real move across decisions to avoid simple ping-pong.
    private static MoveSignature lastMove = null;
    // Short history of recent engine states to avoid small cycles.
    private static final java.util.Deque<Long> recentStates = new java.util.ArrayDeque<>();

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

        long rootKey = solitaire.getStateKey();
        pushRecentState(rootKey);
        int rootHeuristic = evaluate(solitaire);

        Node root = new Node(solitaire.copy(), null, null, 0, rootHeuristic);
        Queue<Node> open = new PriorityQueue<>();
        open.add(root);

        Map<Long, Integer> bestG = new HashMap<>();
        bestG.put(rootKey, 0);

        Node bestNode = null;
        int expansions = 0;

        while (!open.isEmpty() && expansions < MAX_EXPANSIONS) {
            Node current = open.poll();
            expansions++;

            // Track best-scoring node as fallback if we don't find an immediate win.
            if (bestNode == null || current.heuristic > bestNode.heuristic) {
                bestNode = current;
            }

            List<String> moves = LegalMovesHelper.listLegalMoves(current.state);
            for (String move : moves) {
                if (isQuit(move)) {
                    continue;
                }
                // Avoid immediate ping-pong: do not pick the exact inverse of the last move.
                if (isInverseOfLast(move)) {
                    continue;
                }
                // Avoid shuffling kings between tableau columns when nothing is revealed.
                if (isUselessKingMove(current.state, move)) {
                    continue;
                }
                // Avoid obvious backtracking from foundation to tableau.
                if (move.startsWith("move F")) {
                    continue;
                }
                Solitaire copy = current.state.copy();
                applyMove(copy, move);
                long key = copy.getStateKey();
                // Avoid short cycles: skip moves that lead to a recently seen state.
                if (isRecentlySeen(key)) {
                    continue;
                }

                int stepCost = 1 + (isTurn(move) ? TURN_PENALTY : 0);
                int tentativeG = current.g + stepCost;

                int knownG = bestG.getOrDefault(key, Integer.MAX_VALUE);
                if (tentativeG >= knownG) {
                    continue;
                }
                bestG.put(key, tentativeG);

                int h = evaluate(copy);

                Node child = new Node(copy, current, move, tentativeG, h);

                // If we reach a won position, stop early and use this path.
                if (isWon(copy)) {
                    bestNode = child;
                    open.clear();
                    break;
                }
                open.add(child);
            }
        }

        // If search did not find anything better, quit to avoid pointless looping.
        if (bestNode == null || bestNode.parent == null) {
            System.out.println("A* quitting: no improving node found from root state.");
            return "quit";
        }

        // Walk back to root to find the first move.
        Node current = bestNode;
        Node previous = null;
        while (current.parent != null && current.parent.parent != null) {
            previous = current;
            current = current.parent;
        }
        Node firstStep = current.parent == null ? previous : current;
        String chosen = firstStep != null && firstStep.move != null
                ? firstStep.move
                : "quit";
        lastMove = MoveSignature.tryParse(chosen);
        return chosen;
    }

    private boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    private boolean isTurn(String move) {
        return move != null && move.trim().equalsIgnoreCase("turn");
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

    private boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress.
        for (var pile : solitaire.getFoundation()) {
            score += pile.size() * 25;
        }

        // Tableau visibility.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        int emptyColumns = 0;
        for (int i = 0; i < faceUps.size(); i++) {
            int up = faceUps.get(i);
            int down = faceDowns.get(i);
            score += up * 6;
            // Increase the penalty for facedown cards so that moves which
            // reveal hidden cards (and reduce this count) are more strongly
            // preferred by the search.
            score -= down * 8;
            if (up == 0 && down == 0) {
                emptyColumns++;
            }
        }

        // Reward empty tableau columns for king placements.
        score += emptyColumns * 10;

        // Stock drag.
        score -= solitaire.getStockpile().size();

        return score;
    }

    private static void pushRecentState(long key) {
        if (key == 0L) {
            return;
        }
        recentStates.addLast(key);
        while (recentStates.size() > RECENT_STATE_HISTORY) {
            recentStates.removeFirst();
        }
    }

    private static boolean isRecentlySeen(long key) {
        return key != 0L && recentStates.contains(key);
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
        String from = parts[1];
        String dest = parts[parts.length - 1].toUpperCase();
        // Only care about tableau-to-tableau king moves; allow king-to-foundation.
        if (!from.startsWith("T") || !dest.startsWith("T")) {
            return false;
        }
        int pileIndex;
        try {
            pileIndex = Integer.parseInt(from.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return false;
        }
        List<List<Card>> visibleTableau = solitaire.getVisibleTableau();
        if (pileIndex < 0 || pileIndex >= visibleTableau.size()) {
            return false;
        }

        String cardToken = parts[2];
        List<Card> tableauPile = visibleTableau.get(pileIndex);
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

    private static final class Node implements Comparable<Node> {
        final Solitaire state;
        final Node parent;
        final String move;
        final int g;
        final int heuristic;

        Node(Solitaire state, Node parent, String move, int g, int heuristic) {
            this.state = state;
            this.parent = parent;
            this.move = move;
            this.g = g;
            this.heuristic = heuristic;
        }

        int f() {
            // Higher heuristic (good board) should reduce cost, so subtract.
            return g - heuristic;
        }

        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.f(), other.f());
        }
    }
}

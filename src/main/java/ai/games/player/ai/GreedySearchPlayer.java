package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Greedy (1-ply) player for Klondike:
 *
 * - Enumerates ALL legal immediate moves.
 * - Assigns a local score to each move (no search, no simulation, no multi-step planning).
 * - Plays the highest-scoring move. Ties are broken by fixed scan order.
 *
 * Key differences vs the old greedy:
 * 1) Tableau-to-tableau considers ANY face-up run head, not just top cards.
 * 2) Always plays any legal foundation move, with extra score for those that reveal facedown cards.
 * 3) Adds explicit greedy features that matter in solitaire:
 *    - reveal facedown cards
 *    - create / use empty columns
 * 4) Blocks exact inverse moves and known cycle moves to avoid ping-pong loops.
 *
 * Notes:
 * - This stays greedy: it only looks at the current position and never simulates outcomes.
 * - The scoring is just a heuristic for choosing between *already legal* moves.
 */
@Component
@Profile("ai-greedy")
public class GreedySearchPlayer extends AIPlayer implements Player {

    // Tracks whether we made any talon move in the current empty-stock pass.
    // NOTE: made static because the game loop appears to recreate the player each move,
    // which would otherwise reset this and break the "empty stock pass" logic.
    private static boolean talonMovedThisPass = false;

    // Helps detect a full pass through empty stock without progress.
    // NOTE: made static for the same reason as talonMovedThisPass.
    private static boolean sawEmptyStock = false;

    // Last real move, used to block exact inverse ("ping-pong") moves.
    // NOTE: made static because the game loop appears to recreate the player each move.
    // Without persistence, lastMove was always null and inverse blocking never fired.
    private static MoveSignature lastMove = null;

    // Visited engine states and observed transitions for cycle detection.
    private static final Set<Long> visitedStates = new HashSet<>();
    private static final Map<StateCommand, Long> transitions = new HashMap<>();
    private static Long lastStateKey = null;
    private static String lastCommand = null;
    private static long firstTurnStateKey = 0L;
    private static boolean hasFirstTurnState = false;
    private static int consecutiveTurns = 0;
    private static final int MAX_CONSECUTIVE_TURNS = 40;

    /**
     * Resets all static per-game state for this player.
     * Call this when starting a new Solitaire game.
     */
    public static void resetForNewGame() {
        talonMovedThisPass = false;
        sawEmptyStock = false;
        lastMove = null;
        visitedStates.clear();
        transitions.clear();
        lastStateKey = null;
        lastCommand = null;
        firstTurnStateKey = 0L;
        hasFirstTurnState = false;
        consecutiveTurns = 0;
    }

    // -----------------------------
    // Greedy scoring weights
    // -----------------------------

    // Foundation progress is good; safety gate prevents reckless early pushes.
    private static final int SCORE_FOUNDATION = 15;

    // Big reward for revealing facedown cards (opens the game).
    private static final int SCORE_REVEAL_FACEDOWN = 10;

    // Reward for emptying a tableau pile (creates an empty column).
    private static final int SCORE_EMPTY_SOURCE_PILE = 7;

    // Reward for placing onto an empty tableau pile (usually king/run placement).
    private static final int SCORE_TO_EMPTY_PILE = 5;

    // Small baseline for any legal tableau move so we don't stall.
    private static final int SCORE_TABLEAU_MOVE = 1;

    // Talon-to-tableau is often necessary but lower priority than tableau development.
    private static final int SCORE_TALON_TO_TABLEAU = 2;

    @Override
    public String nextCommand(Solitaire solitaire, String recommendedMoves, String feedback) {
        long currentKey = solitaire.getStateKey();

        // Learn the transition from the previous state and command, if any.
        if (lastStateKey != null && lastCommand != null) {
            transitions.put(new StateCommand(lastStateKey, lastCommand), currentKey);
        }
        visitedStates.add(currentKey);

        Move best = pickBestMove(solitaire, currentKey);

        if (best != null) {
            // Remember last move for inverse blocking.
            lastMove = MoveSignature.tryParse(best.command);
            lastStateKey = currentKey;
            lastCommand = best.command;

            // Any non-turn progress breaks a potential stock-cycle loop.
            if (!"turn".equalsIgnoreCase(best.command)) {
                hasFirstTurnState = false;
                consecutiveTurns = 0;
            }

            // Any real move resets empty-stock pass tracking.
            sawEmptyStock = false;

            // If we moved from waste/talon, that's progress in this pass.
            if (best.command.startsWith("move W")) {
                talonMovedThisPass = true;
            }

            return best.command;
        }

        // No legal moves found.
        if (!solitaire.getStockpile().isEmpty()) {
            // Stock still has cards; turning is the only action.
            sawEmptyStock = false;
            talonMovedThisPass = false;
            consecutiveTurns++;
            if (consecutiveTurns >= MAX_CONSECUTIVE_TURNS) {
                return "quit";
            }
            // Detect a pure "turn" cycle: if we are back at the same state after only turning, quit.
            if (hasFirstTurnState && firstTurnStateKey == currentKey) {
                return "quit";
            }
            if (!hasFirstTurnState) {
                firstTurnStateKey = currentKey;
                hasFirstTurnState = true;
            }
            return "turn";
        }

        // Stock empty: allow one recycle/no-op pass before quitting.
        if (sawEmptyStock && !talonMovedThisPass) {
            // Full empty-stock pass with no talon progress -> quit.
            return "quit";
        }

        // First time seeing empty stock this pass: try another turn (recycle/no-op).
        sawEmptyStock = true;
        talonMovedThisPass = false;
        consecutiveTurns++;
        if (consecutiveTurns >= MAX_CONSECUTIVE_TURNS) {
            return "quit";
        }
        return "turn";
    }

    private Move pickBestMove(Solitaire solitaire, long currentKey) {
        List<Move> candidates = new ArrayList<>();

        // Fixed generation order also acts as deterministic tie-breaker.
        addTableauToFoundationMoves(solitaire, candidates);
        addTalonToFoundationMoves(solitaire, candidates);
        addTableauToTableauMoves(solitaire, candidates);
        addTalonToTableauMoves(solitaire, candidates);

        Move best = null;
        for (Move m : candidates) {
            if (isInverseOfLast(m.command)) {
                continue; // hard block ping-pong
            }
            // Avoid moves that lead back to a previously visited state when known.
            if (leadsToVisited(solitaire, currentKey, m.command)) {
                continue;
            }
            if (best == null || m.score > best.score) {
                best = m;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------------
    // Move generation: Foundation moves
    // ---------------------------------------------------------------------

    private void addTableauToFoundationMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
        List<List<Card>> foundations = solitaire.getFoundation();

        for (int from = 0; from < tableau.size(); from++) {
            List<Card> pile = tableau.get(from);
            int faceUp = faceUpCounts.get(from);
            if (pile == null) continue;
            if (faceUp <= 0) continue;

            Card moving = topVisibleTableauCard(pile, faceUp);
            if (moving == null) continue;

            int faceDown = faceDownCounts.get(from);
            boolean revealsFacedown = faceDown > 0 && faceUp == 1;

            boolean safe = isSafeFoundationMove(moving, revealsFacedown, foundations);
            if (!safe) continue;

            for (int f = 0; f < foundations.size(); f++) {
                if (!canMoveToFoundation(moving, foundations.get(f))) continue;

                String cmd = "move T" + (from + 1) + " " + moving.shortName() + " F" + (f + 1);
                if (isInverseOfLast(cmd, moving.shortName())) continue;

                int score = SCORE_FOUNDATION;
                if (revealsFacedown) score += SCORE_REVEAL_FACEDOWN;

                out.add(new Move(cmd, score));
            }
        }
    }

    /**
     * Talon -> Foundation:
     * Talon moves don't reveal facedown directly.
     */
    private void addTalonToFoundationMoves(Solitaire solitaire, List<Move> out) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) return;

        List<List<Card>> foundations = solitaire.getFoundation();
        boolean safe = isSafeFoundationMove(moving, false, foundations);
        if (!safe) return;

        for (int f = 0; f < foundations.size(); f++) {
            if (!canMoveToFoundation(moving, foundations.get(f))) continue;

            String cmd = "move W " + moving.shortName() + " F" + (f + 1);
            if (isInverseOfLast(cmd, moving.shortName())) continue;

            out.add(new Move(cmd, SCORE_FOUNDATION));
        }
    }

    private boolean isSafeFoundationMove(Card moving, boolean revealsFacedown, List<List<Card>> foundations) {
        int rankValue = moving.getRank().getValue();
        if (rankValue <= 2) {
            return true;
        }

        if (revealsFacedown) {
            return true;
        }

        int targetRank = rankValue - 1;
        for (List<Card> pile : foundations) {
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            if (top.getSuit() == moving.getSuit() && top.getRank().getValue() == targetRank) {
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------------
    // Move generation: Tableau -> Tableau
    // ---------------------------------------------------------------------

    /**
     * Tableau -> Tableau:
     * Consider ANY face-up run head in each pile (bottom face-up -> top face-up).
     */
    private void addTableauToTableauMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();

        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            if (fromPile == null) continue;

            int faceUp = faceUpCounts.get(from);
            int faceDown = faceDownCounts.get(from);
            if (fromPile.isEmpty() || faceUp <= 0) continue;

            int lastVisibleIndex = fromPile.size() - 1;

            for (int i = lastVisibleIndex; i >= 0; i--) {
                Card moving = fromPile.get(i);

                if (!isValidRunHead(fromPile, i)) {
                    continue;
                }

                boolean revealsFacedown = (i == lastVisibleIndex && faceDown > 0 && faceUp == fromPile.size());
                boolean emptiesSource = (faceDown + i == 0);

                for (int to = 0; to < tableau.size(); to++) {
                    if (to == from) continue;

                    List<Card> toPile = tableau.get(to);
                    if (!canMoveToTableau(moving, toPile)) continue;

                    String cmd = "move T" + (from + 1) + " " + moving.shortName() + " T" + (to + 1);
                    if (isInverseOfLast(cmd, moving.shortName())) continue;

                    int score = SCORE_TABLEAU_MOVE;
                    if (revealsFacedown) score += SCORE_REVEAL_FACEDOWN;
                    if (emptiesSource) score += SCORE_EMPTY_SOURCE_PILE;
                    if (toPile.isEmpty()) score += SCORE_TO_EMPTY_PILE;

                    out.add(new Move(cmd, score));
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Move generation: Talon -> Tableau
    // ---------------------------------------------------------------------

    /**
     * Talon -> Tableau:
     * Consider placements into tableau, with moderate score.
     */
    private void addTalonToTableauMoves(Solitaire solitaire, List<Move> out) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) return;

        List<List<Card>> tableau = solitaire.getVisibleTableau();
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> toPile = tableau.get(t);

            if (!canMoveToTableau(moving, toPile)) continue;

            String cmd = "move W " + moving.shortName() + " T" + (t + 1);
            if (isInverseOfLast(cmd, moving.shortName())) continue;

            int score = SCORE_TALON_TO_TABLEAU;
            if (toPile.isEmpty()) score += SCORE_TO_EMPTY_PILE;

            out.add(new Move(cmd, score));
        }
    }

    // ---------------------------------------------------------------------
    // Small helpers
    // ---------------------------------------------------------------------

    /**
     * Returns the top visible tableau card given top-first storage.
     */
    private Card topVisibleTableauCard(List<Card> pile, int faceUp) {
        if (pile == null || pile.isEmpty() || faceUp <= 0) return null;
        return pile.get(pile.size() - 1);
    }

    private boolean isValidRunHead(List<Card> pile, int index) {
        for (int i = index; i < pile.size() - 1; i++) {
            Card current = pile.get(i);
            Card next = pile.get(i + 1);
            boolean alternatingColor = current.getSuit().isRed() != next.getSuit().isRed();
            boolean oneLower = current.getRank().getValue() == next.getRank().getValue() + 1;
            if (!alternatingColor || !oneLower) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------
    // Ping-pong prevention (exact inverse only)
    // ---------------------------------------------------------------------

    private boolean leadsToVisited(Solitaire solitaire, long currentKey, String command) {
        StateCommand key = new StateCommand(currentKey, command);
        Long next = transitions.get(key);
        if (next == null) {
            // Simulate the move on a copy to discover the resulting state key.
            Solitaire copy = solitaire.copy();
            String trimmed = command.trim();
            if ("turn".equalsIgnoreCase(trimmed)) {
                copy.turnThree();
            } else if (trimmed.startsWith("move")) {
                String[] parts = trimmed.split("\\s+");
                // Destination pile code is always the last token.
                String dest = parts[parts.length - 1];
                // Never block moves that go to foundation; they always represent progress.
                if (dest.startsWith("F")) {
                    return false;
                }
                if (parts.length == 3) {
                    copy.moveCard(parts[1], null, dest);
                } else if (parts.length >= 4) {
                    copy.moveCard(parts[1], parts[2], dest);
                }
            }
            next = copy.getStateKey();
            transitions.put(key, next);
        }
        return visitedStates.contains(next);
    }

    private boolean isInverseOfLast(String candidateCmd, String candidateCardShort) {
        if (lastMove == null) return false;
        MoveSignature cand = MoveSignature.tryParse(candidateCmd, candidateCardShort);
        if (cand == null) return false;
        return lastMove.isInverseOf(cand);
    }

    private boolean isInverseOfLast(String cmd) {
        if (lastMove == null) return false;
        MoveSignature cand = MoveSignature.tryParse(cmd);
        return cand != null && lastMove.isInverseOf(cand);
    }

    private static class MoveSignature {
        final String from;
        final String to;
        final String cardShort; // may be null for "move Tn Fn"

        private MoveSignature(String from, String to, String cardShort) {
            this.from = from;
            this.to = to;
            this.cardShort = cardShort;
        }

        boolean isInverseOf(MoveSignature other) {
            if (other == null) return false;
            if (cardShort != null && other.cardShort != null && !cardShort.equals(other.cardShort)) {
                return false;
            }
            return from.equals(other.to) && to.equals(other.from);
        }

        static MoveSignature tryParse(String cmd) {
            return tryParse(cmd, null);
        }

        static MoveSignature tryParse(String cmd, String fallbackCardShort) {
            if (cmd == null || !cmd.startsWith("move")) return null;

            // Supported shapes:
            // 1) "move Tn Fn"
            // 2) "move Tn <card> Tm/Fm"
            // 3) "move W <card> Tm/Fm"
            String[] parts = cmd.split("\\s+");
            if (parts.length < 3) return null;

            String from = parts[1];                 // "Tn" or "W"
            String to = parts[parts.length - 1];    // "Tn" or "Fn"
            String cardShort = fallbackCardShort;

            // If explicit card is present, it's the token before the destination.
            if (parts.length >= 4) {
                String maybeCard = parts[parts.length - 2];
                if (maybeCard.matches(".*[A23456789TJQK].*")) {
                    cardShort = maybeCard;
                }
            }

            return new MoveSignature(from, to, cardShort);
        }
    }

    private static class Move {
        final String command;
        final int score;

        Move(String command, int score) {
            this.command = command;
            this.score = score;
        }
    }

    private static final class StateCommand {
        final long stateKey;
        final String command;

        StateCommand(long stateKey, String command) {
            this.stateKey = stateKey;
            this.command = command;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof StateCommand)) {
                return false;
            }
            StateCommand other = (StateCommand) obj;
            if (stateKey != other.stateKey) {
                return false;
            }
            return command == null ? other.command == null : command.equals(other.command);
        }

        @Override
        public int hashCode() {
            int result = Long.hashCode(stateKey);
            result = 31 * result + (command != null ? command.hashCode() : 0);
            return result;
        }
    }
}

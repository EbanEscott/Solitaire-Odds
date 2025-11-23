package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.List;
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
 * 2) Adds explicit greedy features that matter in solitaire:
 *    - reveal facedown cards
 *    - create / use empty columns
 *    - avoid unsafe early foundation moves
 * 3) Blocks exact inverse moves to avoid ping-pong loops.
 *
 * Notes:
 * - This stays greedy: it only looks at the current position and never simulates outcomes.
 * - The scoring is just a heuristic for choosing between *already legal* moves.
 */
@Component
@Profile("ai-greedy")
public class GreedySearchPlayer extends AIPlayer implements Player {

    // Tracks whether we made any talon move in the current empty-stock pass.
    private boolean talonMovedThisPass = false;

    // Helps detect a full pass through empty stock without progress.
    private boolean sawEmptyStock = false;

    // Last real move, used to block exact inverse ("ping-pong") moves.
    private MoveSignature lastMove = null;

    // -----------------------------
    // Greedy scoring weights
    // -----------------------------

    // Foundation progress is good, but not at any cost (see safety gate).
    private static final int SCORE_FOUNDATION = 6;

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
    public String nextCommand(Solitaire solitaire, String feedback) {
        Move best = pickBestMove(solitaire);

        if (best != null) {
            // Remember last move for inverse blocking.
            lastMove = MoveSignature.tryParse(best.command);

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
        return "turn";
    }

    private Move pickBestMove(Solitaire solitaire) {
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
            if (best == null || m.score > best.score) {
                best = m;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------------
    // Move generation: Foundation moves
    // ---------------------------------------------------------------------

    /**
     * Tableau -> Foundation:
     * Only the TOP face-up card can go to foundation.
     * We add a safety gate to avoid locking ourselves early.
     */
    private void addTableauToFoundationMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        List<List<Card>> foundations = solitaire.getFoundation();

        for (int from = 0; from < tableau.size(); from++) {
            List<Card> pile = tableau.get(from);
            int faceUp = faceUpCounts.get(from);

            Card moving = topTableauCard(pile, faceUp);
            if (moving == null) continue;

            boolean revealsFacedown = revealsFacedownIfMovedFromTop(pile, faceUp);
            boolean safe = isSafeFoundationMove(moving, revealsFacedown);
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
     * Greedy-safe gate applied as well. Talon moves don't reveal facedown directly.
     */
    private void addTalonToFoundationMoves(Solitaire solitaire, List<Move> out) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) return;

        List<List<Card>> foundations = solitaire.getFoundation();
        boolean safe = isSafeFoundationMove(moving, false);
        if (!safe) return;

        for (int f = 0; f < foundations.size(); f++) {
            if (!canMoveToFoundation(moving, foundations.get(f))) continue;

            String cmd = "move W " + moving.shortName() + " F" + (f + 1);
            if (isInverseOfLast(cmd, moving.shortName())) continue;

            out.add(new Move(cmd, SCORE_FOUNDATION));
        }
    }

    /**
     * Safety gate for foundation:
     * - Always safe for Ace / Two.
     * - Otherwise only safe if the move *immediately* reveals a facedown card.
     *
     * This avoids dumping low cards to foundation too early and losing mobility.
     */
    private boolean isSafeFoundationMove(Card moving, boolean revealsFacedown) {
        int r = moving.getRank().getValue();
        if (r <= 2) return true;
        return revealsFacedown;
    }

    // ---------------------------------------------------------------------
    // Move generation: Tableau -> Tableau
    // ---------------------------------------------------------------------

    /**
     * Tableau -> Tableau:
     * Consider ANY face-up run head in each pile (bottom face-up -> top face-up).
     */
    private void addTableauToTableauMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();

        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            int faceUp = faceUpCounts.get(from);

            if (fromPile.isEmpty() || faceUp <= 0) continue;

            int firstFaceUpIndex = fromPile.size() - faceUp;

            // Scan all run heads from bottom face-up -> top face-up.
            for (int i = firstFaceUpIndex; i < fromPile.size(); i++) {
                Card moving = fromPile.get(i);

                boolean revealsFacedown = (i == firstFaceUpIndex && firstFaceUpIndex > 0);
                boolean emptiesSource = (i == 0); // moving whole pile

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

        List<List<Card>> tableau = solitaire.getTableau();
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
     * Returns true if moving the TOP face-up card from this pile would reveal a facedown card.
     */
    private boolean revealsFacedownIfMovedFromTop(List<Card> pile, int faceUp) {
        if (pile == null || pile.isEmpty() || faceUp <= 0) return false;
        int firstFaceUpIndex = pile.size() - faceUp;
        return firstFaceUpIndex > 0;
    }

    // ---------------------------------------------------------------------
    // Ping-pong prevention (exact inverse only)
    // ---------------------------------------------------------------------

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
}

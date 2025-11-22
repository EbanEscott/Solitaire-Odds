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
 * Greedy search: evaluates immediate moves and picks the highest-scoring one (short-term gain only).
 */
@Component
@Profile("ai-greedy")
public class GreedySearchPlayer extends AIPlayer implements Player {

    private static final int SCORE_FOUNDATION = 3;
    private static final int SCORE_TABLEAU_FLIP = 2; // prefer moves that might reveal face-down (approximation)
    private static final int SCORE_TABLEAU = 1;

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        Move best = pickBestMove(solitaire);
        if (best != null) {
            return best.command;
        }
        if (!solitaire.getStockpile().isEmpty()) {
            return "turn";
        }
        return "quit";
    }

    private Move pickBestMove(Solitaire solitaire) {
        List<Move> candidates = new ArrayList<>();
        addTalonMoves(solitaire, candidates);
        addTableauMoves(solitaire, candidates);
        addTableauToTableauMoves(solitaire, candidates);

        Move best = null;
        for (Move m : candidates) {
            if (best == null || m.score > best.score) {
                best = m;
            }
        }
        return best;
    }

    private void addTalonMoves(Solitaire solitaire, List<Move> out) {
        Card moving = top(solitaire.getTalon());
        if (moving == null) {
            return;
        }
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int f = 0; f < foundations.size(); f++) {
            if (canMoveToFoundation(moving, foundations.get(f))) {
                out.add(new Move("move W " + moving.shortName() + " F" + (f + 1), SCORE_FOUNDATION));
            }
        }

        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = 0; t < tableau.size(); t++) {
            if (canMoveToTableau(moving, tableau.get(t))) {
                out.add(new Move("move W " + moving.shortName() + " T" + (t + 1), SCORE_TABLEAU));
            }
        }
    }

    private void addTableauMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getTableau();
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int from = 0; from < tableau.size(); from++) {
            Card moving = top(tableau.get(from));
            if (moving == null) {
                continue;
            }
            for (int f = 0; f < foundations.size(); f++) {
                if (canMoveToFoundation(moving, foundations.get(f))) {
                    out.add(new Move("move T" + (from + 1) + " " + moving.shortName() + " F" + (f + 1), SCORE_FOUNDATION));
                }
            }
        }
    }

    private void addTableauToTableauMoves(Solitaire solitaire, List<Move> out) {
        List<List<Card>> tableau = solitaire.getTableau();
        for (int from = 0; from < tableau.size(); from++) {
            Card moving = top(tableau.get(from));
            if (moving == null) {
                continue;
            }
            for (int to = 0; to < tableau.size(); to++) {
                if (from == to) {
                    continue;
                }
                if (canMoveToTableau(moving, tableau.get(to))) {
                    // Slightly favor moves that might expose a face-down card by weighting tableau moves modestly.
                    out.add(new Move("move T" + (from + 1) + " " + moving.shortName() + " T" + (to + 1), SCORE_TABLEAU_FLIP));
                }
            }
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

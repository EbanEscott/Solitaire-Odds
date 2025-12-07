package ai.games.player;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Computes legal move commands for a given {@link Solitaire} state.
 */
public final class LegalMovesHelper {
    private LegalMovesHelper() {
    }

    /**
     * Return all currently legal commands (excluding "quit").
     */
    public static List<String> listLegalMoves(Solitaire solitaire) {
        if (solitaire == null) {
            return Collections.emptyList();
        }
        List<String> moves = new ArrayList<>();
        addTableauToFoundation(solitaire, moves);
        addTableauToTableau(solitaire, moves);
        addTalonToFoundation(solitaire, moves);
        addTalonToTableau(solitaire, moves);
        addFoundationToTableau(solitaire, moves);
        // "turn" is always legal if stock or talon still has cards to cycle.
        if (!solitaire.getStockpile().isEmpty() || !solitaire.getTalon().isEmpty()) {
            moves.add("turn");
        }
        // "quit" is always a legal command from the engine's perspective.
        moves.add("quit");
        return moves;
    }

    private static void addTableauToFoundation(Solitaire solitaire, List<String> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<List<Card>> foundations = solitaire.getFoundation();
        // Only top face-up cards can move to foundation.
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> pile = tableau.get(t);
            int faceUp = faceUps.get(t);
            if (pile.isEmpty() || faceUp <= 0) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            for (int f = 0; f < foundations.size(); f++) {
                if (canPlaceOnFoundation(top, foundations.get(f))) {
                    out.add("move T" + (t + 1) + " " + top.shortName() + " F" + (f + 1));
                }
            }
        }
    }

    private static void addTableauToTableau(Solitaire solitaire, List<String> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        // Any visible card can move as the start of a stack to another tableau pile.
        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            int faceUp = faceUps.get(from);
            if (fromPile.isEmpty() || faceUp <= 0) {
                continue;
            }
            int start = Math.max(0, fromPile.size() - faceUp);
            for (int idx = start; idx < fromPile.size(); idx++) {
                Card moving = fromPile.get(idx);
                for (int to = 0; to < tableau.size(); to++) {
                    if (to == from) {
                        continue;
                    }
                    if (canPlaceOnTableau(moving, tableau.get(to))) {
                        out.add("move T" + (from + 1) + " " + moving.shortName() + " T" + (to + 1));
                    }
                }
            }
        }
    }

    private static void addTalonToFoundation(Solitaire solitaire, List<String> out) {
        List<Card> talon = solitaire.getTalon();
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        List<List<Card>> foundations = solitaire.getFoundation();
        for (int f = 0; f < foundations.size(); f++) {
            if (canPlaceOnFoundation(top, foundations.get(f))) {
                out.add("move W F" + (f + 1));
            }
        }
    }

    private static void addTalonToTableau(Solitaire solitaire, List<String> out) {
        List<Card> talon = solitaire.getTalon();
        if (talon.isEmpty()) {
            return;
        }
        // Only top of talon is playable.
        Card top = talon.get(talon.size() - 1);
        List<List<Card>> tableau = solitaire.getTableau();
        for (int t = 0; t < tableau.size(); t++) {
            if (canPlaceOnTableau(top, tableau.get(t))) {
                out.add("move W T" + (t + 1));
            }
        }
    }

    private static void addFoundationToTableau(Solitaire solitaire, List<String> out) {
        List<List<Card>> foundations = solitaire.getFoundation();
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        // Foundations can only move their top card back to tableau.
        for (int f = 0; f < foundations.size(); f++) {
            List<Card> pile = foundations.get(f);
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            // Do not suggest moving Aces or Twos down from foundations; that is almost
            // always strategically bad and creates noisy moves for the AI.
            if (top.getRank() == Rank.ACE || top.getRank() == Rank.TWO) {
                continue;
            }
            for (int t = 0; t < tableau.size(); t++) {
                if (canPlaceOnTableau(top, tableau.get(t))) {
                    out.add("move F" + (f + 1) + " T" + (t + 1));
                }
            }
        }
    }

    private static boolean canPlaceOnTableau(Card moving, List<Card> toPile) {
        if (moving == null) {
            return false;
        }
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.KING;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean alternatingColor = moving.getSuit().isRed() != target.getSuit().isRed();
        boolean oneLower = moving.getRank().getValue() == target.getRank().getValue() - 1;
        return alternatingColor && oneLower;
    }

    private static boolean canPlaceOnFoundation(Card moving, List<Card> toPile) {
        if (moving == null) {
            return false;
        }
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.ACE;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean sameSuit = moving.getSuit() == target.getSuit();
        boolean oneHigher = moving.getRank().getValue() == target.getRank().getValue() + 1;
        return sameSuit && oneHigher;
    }
}

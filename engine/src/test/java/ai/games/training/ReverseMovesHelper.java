package ai.games.training;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Computes legal reverse move commands for a given {@link Solitaire} state.
 * These are moves that could have led to the current position (predecessors in the game tree).
 * 
 * For example, if the current state has a card on foundation, a reverse move
 * would be moving that card back to tableau or talon.
 */
public final class ReverseMovesHelper {
    private ReverseMovesHelper() {
    }

    /**
     * Return all currently legal reverse move commands.
     * These are moves that could have been made to reach the current state.
     */
    public static List<String> listReverseMoves(Solitaire solitaire) {
        if (solitaire == null) {
            return Collections.emptyList();
        }
        List<String> moves = new ArrayList<>();
        addFoundationToTableau(solitaire, moves);
        addFoundationToTalon(solitaire, moves);
        addTableauToTableau(solitaire, moves);
        addTableauToTalon(solitaire, moves);
        // "turn" reverse: cycling talon backwards (equivalent to turn, but conceptually reverse)
        if (!solitaire.getStockpile().isEmpty() || !solitaire.getTalon().isEmpty()) {
            moves.add("turn");
        }
        return moves;
    }

    /**
     * Add reverse moves from foundation back to tableau.
     * A card that is currently on a foundation could have come from tableau.
     */
    private static void addFoundationToTableau(Solitaire solitaire, List<String> out) {
        List<List<Card>> foundations = solitaire.getFoundation();
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        
        // For each foundation, if it's not empty, the top card could reverse to tableau
        for (int f = 0; f < foundations.size(); f++) {
            List<Card> pile = foundations.get(f);
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            
            // Check if this card could have come from tableau
            for (int t = 0; t < tableau.size(); t++) {
                if (couldHaveComeFromTableau(top, tableau.get(t))) {
                    out.add("move F" + (f + 1) + " " + top.shortName() + " T" + (t + 1));
                }
            }
        }
    }

    /**
     * Add reverse moves from foundation back to talon.
     * A card on a foundation could have originally come from the talon/waste pile.
     */
    private static void addFoundationToTalon(Solitaire solitaire, List<String> out) {
        List<List<Card>> foundations = solitaire.getFoundation();
        
        // For each foundation, if it's not empty, the top card could reverse to talon
        for (int f = 0; f < foundations.size(); f++) {
            List<Card> pile = foundations.get(f);
            if (pile.isEmpty()) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            // Any card can be moved to talon (it could have originally been played from there)
            out.add("move F" + (f + 1) + " " + top.shortName() + " W");
        }
    }
    /**
     * Add reverse moves from tableau to tableau.
     * A card sequence in tableau could have come from another tableau pile.
     */
    private static void addTableauToTableau(Solitaire solitaire, List<String> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        
        // For each tableau pile, cards could have come from another tableau
        for (int from = 0; from < tableau.size(); from++) {
            List<Card> fromPile = tableau.get(from);
            int faceUp = faceUps.get(from);
            if (fromPile.isEmpty() || faceUp <= 0) {
                continue;
            }
            
            // Any visible card in this pile could have come from another tableau
            int start = Math.max(0, fromPile.size() - faceUp);
            for (int idx = start; idx < fromPile.size(); idx++) {
                Card moving = fromPile.get(idx);
                
                for (int to = 0; to < tableau.size(); to++) {
                    if (to == from) {
                        continue;
                    }
                    // Check if this card could have come from the other pile
                    if (couldHaveComeFromTableau(moving, tableau.get(to))) {
                        out.add("move T" + (from + 1) + " " + moving.shortName() + " T" + (to + 1));
                    }
                }
            }
        }
    }

    /**
     * Add reverse moves from tableau back to talon.
     * A card in tableau could have originally come from talon.
     */
    private static void addTableauToTalon(Solitaire solitaire, List<String> out) {
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        
        // For each tableau pile, the top visible card could have come from talon
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> pile = tableau.get(t);
            int faceUp = faceUps.get(t);
            if (pile.isEmpty() || faceUp <= 0) {
                continue;
            }
            Card top = pile.get(pile.size() - 1);
            // Any card can be moved back to talon (it could have been played from there)
            out.add("move T" + (t + 1) + " " + top.shortName() + " W");
        }
    }

    /**
     * Check if a card could have come from a tableau pile (reverse of canPlaceOnTableau).
     * The card is currently positioned somewhere, and we check if it could have
     * originally been the top of the given tableau pile.
     */
    private static boolean couldHaveComeFromTableau(Card moved, List<Card> fromPile) {
        if (moved == null) {
            return false;
        }
        if (fromPile.isEmpty()) {
            // A card could have come from an empty tableau if it was a King
            return moved.getRank() == Rank.KING;
        }
        Card target = fromPile.get(fromPile.size() - 1);
        // The moved card would need to be one rank higher than the current top,
        // with opposite color (standard tableau stacking rules in reverse)
        boolean alternatingColor = moved.getSuit().isRed() != target.getSuit().isRed();
        boolean oneHigher = moved.getRank().getValue() == target.getRank().getValue() + 1;
        return alternatingColor && oneHigher;
    }
}

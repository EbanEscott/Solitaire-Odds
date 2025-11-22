package ai.games.player;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import java.util.List;

/**
 * Base class for AI players with helper methods for rule evaluation.
 */
public abstract class AIPlayer implements Player {

    protected boolean canMoveToFoundation(Card moving, List<Card> foundationPile) {
        if (moving == null) {
            return false;
        }
        if (foundationPile.isEmpty()) {
            return moving.getRank() == Rank.ACE;
        }
        Card top = foundationPile.get(foundationPile.size() - 1);
        boolean sameSuit = moving.getSuit() == top.getSuit();
        boolean oneHigher = moving.getRank().getValue() == top.getRank().getValue() + 1;
        return sameSuit && oneHigher;
    }

    protected boolean canMoveToTableau(Card moving, List<Card> tableauPile) {
        if (moving == null) {
            return false;
        }
        if (tableauPile.isEmpty()) {
            return moving.getRank() == Rank.KING;
        }
        Card top = tableauPile.get(tableauPile.size() - 1);
        boolean alternatingColor = moving.getSuit().isRed() != top.getSuit().isRed();
        boolean oneLower = moving.getRank().getValue() == top.getRank().getValue() - 1;
        return alternatingColor && oneLower;
    }

    protected Card top(List<Card> pile) {
        if (pile == null || pile.isEmpty()) {
            return null;
        }
        return pile.get(pile.size() - 1);
    }
}

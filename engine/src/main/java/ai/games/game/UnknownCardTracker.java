package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tracks which cards have been revealed during play via the stock/talon cycle.
 * <p>
 * In Solitaire PLAN mode (lookahead), face-down cards in the tableau are masked with
 * UNKNOWN placeholders. However, cards that have been revealed during actual play
 * (by cycling them through the talon) are no longer truly hidden and should not be
 * masked. This tracker maintains a set of revealed cards so that lookahead can
 * distinguish between:
 * <ul>
 *   <li><strong>Unknown cards:</strong> Never cycled through talon; truly hidden.</li>
 *   <li><strong>Revealed cards:</strong> Once visible in the talon; now in stockpile but known to the player.</li>
 * </ul>
 * <p>
 * The tracker is created once per game and persists across stock cycles. It is reset
 * only when a new game begins.
 * <p>
 * <strong>Usage:</strong>
 * <ul>
 *   <li>Call {@link #addRevealedCard(Card)} when a card moves from stockpile to talon.</li>
 *   <li>Call {@link #isRevealed(Card)} to check if a specific card has been seen.</li>
 *   <li>Call {@link #getUnknownCards(Solitaire)} to get a list of all cards not yet revealed.</li>
 * </ul>
 */
public class UnknownCardTracker {
    /** Set of cards that have been revealed during play (cycled through talon). */
    private final Set<Card> revealedCards = new HashSet<>();

    /**
     * Constructs a new tracker with an empty set of revealed cards.
     * <p>
     * Called once per game during {@link Solitaire} initialization.
     */
    public UnknownCardTracker() {
        // Initially, no cards have been revealed.
    }

    /**
     * Records that a card has been revealed (visible in the talon).
     * <p>
     * This is called when a card is drawn from the stockpile onto the talon
     * during a turn operation. Once recorded, the card is considered "known" to the player
     * even if it later cycles back to the stockpile.
     *
     * @param card the card that has been revealed; must not be null
     */
    public void addRevealedCard(Card card) {
        if (card != null) {
            revealedCards.add(card);
        }
    }

    /**
     * Checks whether a specific card has been revealed during play.
     *
     * @param card the card to check; must not be null
     * @return {@code true} if this card has been revealed (cycled through talon);
     *         {@code false} if it remains unknown (hidden face-down)
     */
    public boolean isRevealed(Card card) {
        return revealedCards.contains(card);
    }

    /**
     * Returns a list of all cards that are unknown (not yet revealed) in the given game state.
     * <p>
     * This includes:
     * <ul>
     *   <li>Face-down cards in the tableau (cards beyond the visible suffix in each pile).</li>
     *   <li>Cards in the stockpile (which have not cycled through the talon yet).</li>
     * </ul>
     * <p>
     * Cards that have cycled through the talon are <strong>not</strong> included, even if
     * they are currently in the stockpile, because the player has already seen them.
     *
     * @param solitaire the game state to analyze; must not be null
     * @return a list of unknown cards; never null, may be empty
     */
    public List<Card> getUnknownCards(Solitaire solitaire) {
        Set<Card> unknown = new HashSet<>();

        // Add all face-down cards from the tableau.
        List<List<Card>> tableau = solitaire.getTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = i < faceUpCounts.size() ? faceUpCounts.get(i) : 0;
            if (faceUp < pile.size()) {
                // Add all face-down cards (cards before the visible suffix).
                int faceDownStart = Math.max(0, pile.size() - faceUp);
                for (int j = 0; j < faceDownStart; j++) {
                    Card card = pile.get(j);
                    if (!revealedCards.contains(card)) {
                        unknown.add(card);
                    }
                }
            }
        }

        // Add all cards in the stockpile that have not been revealed.
        for (Card card : solitaire.getStockpile()) {
            if (!revealedCards.contains(card)) {
                unknown.add(card);
            }
        }

        return Collections.unmodifiableList(new ArrayList<>(unknown));
    }
}

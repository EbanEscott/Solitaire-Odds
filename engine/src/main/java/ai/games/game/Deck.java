package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Represents a standard 52-card deck used in Solitaire games.
 * <p>
 * A {@code Deck} manages a collection of {@link Card} objects, supporting operations
 * such as shuffling, drawing cards, and querying the deck state. The deck is initialised
 * with all 52 cards (13 ranks × 4 suits) and is automatically shuffled upon construction.
 */
public class Deck {
    /** The list of cards currently in the deck. */
    private final List<Card> cards = new ArrayList<>();

    /**
     * Constructs a new Deck with all 52 cards and shuffles it.
     */
    public Deck() {
        reset();
    }

    /**
     * Shuffles the cards in the deck using a pseudo-random permutation.
     * <p>
     * This method is called automatically during {@link #reset()}, but can be invoked
     * again to reshuffle the deck at any time.
     */
    public void shuffle() {
        Collections.shuffle(cards);
    }

    /**
     * Draws and removes the top card from the deck.
     * <p>
     * The drawn card is removed from the deck's internal list. If the deck is empty,
     * returns {@code null}.
     *
     * @return the top card of the deck, or {@code null} if the deck is empty
     */
    public Card draw() {
        if (cards.isEmpty()) {
            return null;
        }
        return cards.remove(cards.size() - 1);
    }

    /**
     * Returns the number of cards remaining in the deck.
     *
     * @return the number of cards in the deck
     */
    public int size() {
        return cards.size();
    }

    /**
     * Checks whether the deck is empty.
     *
     * @return {@code true} if no cards remain in the deck; {@code false} otherwise
     */
    public boolean isEmpty() {
        return cards.isEmpty();
    }

    /**
     * Returns an unmodifiable view of the cards in the deck.
     * <p>
     * Useful for reading the deck state without allowing external modifications.
     *
     * @return an unmodifiable list of the deck's cards
     */
    public List<Card> asUnmodifiableList() {
        return Collections.unmodifiableList(cards);
    }

    /**
     * Resets the deck to its initial state with all 52 cards.
     * <p>
     * Clears the deck, creates one card for each combination of {@link Suit} and {@link Rank},
     * and then shuffles the deck.
     */
    public final void reset() {
        cards.clear();
        for (Suit suit : Suit.values()) {
            for (Rank rank : Rank.values()) {
                cards.add(new Card(rank, suit));
            }
        }
        shuffle(); // Shuffle by default after creating the full deck.
    }

    /**
     * Returns a string representation of the deck.
     *
     * @return a string showing the deck size (e.g., "Deck(size=52)")
     */
    @Override
    public String toString() {
        return "Deck(size=" + cards.size() + ")";
    }
}

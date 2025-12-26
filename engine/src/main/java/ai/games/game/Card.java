package ai.games.game;

import java.util.Objects;

/**
 * Represents a single playing card with a {@link Rank} and a {@link Suit}.
 * <p>
 * Each card is immutable and uniquely identified by its rank and suit combination.
 * Cards provide methods for string representation (including optional colour formatting
 * for display in terminals) and equality comparison.
 */
public class Card {
    /** The rank (Ace through King) of this card. */
    private final Rank rank;
    /** The suit (Clubs, Diamonds, Hearts, Spades) of this card. */
    private final Suit suit;

    /**
     * Constructs a Card with the given rank and suit.
     *
     * @param rank the rank of the card (must not be null)
     * @param suit the suit of the card (must not be null)
     * @throws NullPointerException if rank or suit is null
     */
    public Card(Rank rank, Suit suit) {
        this.rank = Objects.requireNonNull(rank, "rank");
        this.suit = Objects.requireNonNull(suit, "suit");
    }

    /**
     * Returns the rank of this card.
     *
     * @return the rank (e.g., {@code Rank.ACE}, {@code Rank.KING})
     */
    public Rank getRank() {
        return rank;
    }

    /**
     * Returns the suit of this card.
     *
     * @return the suit (e.g., {@code Suit.SPADES}, {@code Suit.HEARTS})
     */
    public Suit getSuit() {
        return suit;
    }

    /**
     * Returns a short, non-colored string representation of this card.
     * <p>
     * The format is the rank label followed by the suit symbol
     * (e.g., "Q♠", "10♦", "A♣").
     *
     * @return the short name of the card
     */
    public String shortName() {
        return rank.toString() + suit.getSymbol();
    }

    /**
     * Checks whether this card matches the given short name (case-insensitive).
     * <p>
     * The name is compared against {@link #shortName()} after trimming and
     * lowercasing. Useful for parsing user input or log-based card references.
     *
     * @param name the name to match (e.g., "Q♠", "10♦")
     * @return {@code true} if the name matches this card's short name; {@code false} otherwise
     */
    public boolean matchesShortName(String name) {
        if (name == null) {
            return false;
        }
        return shortName().equalsIgnoreCase(name.trim());
    }

    /**
     * Returns a string representation of this card with optional colour formatting.
     * <p>
     * Red suits (Diamonds and Hearts) are coloured for terminal display;
     * black suits (Clubs and Spades) are not.
     *
     * @return the card string (e.g., "Q♠" or coloured "10♦")
     */
    @Override
    public String toString() {
        String value = rank.toString() + suit.getSymbol();
        return Suit.colouriseIfRed(suit, value);
    }

    /**
     * Checks equality based on rank and suit.
     * <p>
     * Two cards are equal if and only if they have the same rank and suit.
     *
     * @param o the object to compare with
     * @return {@code true} if both cards have identical rank and suit; {@code false} otherwise
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Card)) {
            return false;
        }
        Card card = (Card) o;
        return rank == card.rank && suit == card.suit;
    }

    /**
     * Returns the hash code of this card based on rank and suit.
     * <p>
     * Cards with the same rank and suit have the same hash code, ensuring
     * consistency with {@link #equals(Object)}.
     *
     * @return the hash code for this card
     */
    @Override
    public int hashCode() {
        return Objects.hash(rank, suit);
    }
}

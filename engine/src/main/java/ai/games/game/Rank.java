package ai.games.game;

/**
 * Enumeration representing the 13 ranks of a standard playing card deck.
 * <p>
 * Each rank is assigned a numeric value (1–13) for ordering and comparison,
 * and a short label for string representation (e.g., "A", "K", "10").
 * In Solitaire, ranks are used to determine valid moves (e.g., building
 * foundations in ascending order, cascades in descending order).
 */
public enum Rank {
    /** Ace – the lowest rank in most Solitaire variants (value 1). */
    ACE(1, "A"),
    /** Two – rank value 2. */
    TWO(2, "2"),
    /** Three – rank value 3. */
    THREE(3, "3"),
    /** Four – rank value 4. */
    FOUR(4, "4"),
    /** Five – rank value 5. */
    FIVE(5, "5"),
    /** Six – rank value 6. */
    SIX(6, "6"),
    /** Seven – rank value 7. */
    SEVEN(7, "7"),
    /** Eight – rank value 8. */
    EIGHT(8, "8"),
    /** Nine – rank value 9. */
    NINE(9, "9"),
    /** Ten – rank value 10. */
    TEN(10, "10"),
    /** Jack – rank value 11. */
    JACK(11, "J"),
    /** Queen – rank value 12. */
    QUEEN(12, "Q"),
    /** King – the highest rank in most Solitaire variants (value 13). */
    KING(13, "K");

    /** Numeric value of the rank, used for ordering and comparisons (1–13). */
    private final int value;
    /** Short string label for display (e.g., "A", "K", "10"). */
    private final String label;

    /**
     * Constructs a Rank with a numeric value and display label.
     *
     * @param value the numeric rank value (1–13)
     * @param label the short string representation of the rank
     */
    Rank(int value, String label) {
        this.value = value;
        this.label = label;
    }

    /**
     * Returns the numeric value of this rank.
     * <p>
     * Used for comparing ranks and determining valid moves in Solitaire
     * (e.g., foundations require consecutive ascending ranks).
     *
     * @return the numeric rank value (1 for Ace, 13 for King)
     */
    public int getValue() {
        return value;
    }

    /**
     * Returns the short string label of this rank.
     *
     * @return the label (e.g., "A", "K", "10")
     */
    public String getLabel() {
        return label;
    }

    /**
     * Returns the string representation of this rank.
     * <p>
     * Equivalent to {@link #getLabel()}.
     *
     * @return the rank label
     */
    @Override
    public String toString() {
        return label;
    }
}

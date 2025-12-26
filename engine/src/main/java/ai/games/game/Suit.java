package ai.games.game;

/**
 * Enumeration representing the four suits of a standard playing card deck.
 * <p>
 * Each suit is represented by a Unicode symbol and is classified as either red
 * (Diamonds, Hearts) or black (Clubs, Spades). In Solitaire, suit alternation
 * (red vs. black) is often used to validate moves on cascade piles.
 * <p>
 * This enum also provides ANSI colour formatting utilities for terminal display,
 * allowing red suits to be rendered in red text and black suits in default text.
 */
public enum Suit {
    /** Clubs – a black suit represented by the ♣ symbol. */
    CLUBS("♣", false),
    /** Diamonds – a red suit represented by the ♦ symbol. */
    DIAMONDS("♦", true),
    /** Hearts – a red suit represented by the ♥ symbol. */
    HEARTS("♥", true),
    /** Spades – a black suit represented by the ♠ symbol. */
    SPADES("♠", false);

    /** ANSI escape code for red text output in terminals. */
    private static final String ANSI_RED = "\u001B[31m";
    /** ANSI escape code to reset text formatting in terminals. */
    private static final String ANSI_RESET = "\u001B[0m";

    /** The Unicode symbol representing this suit (e.g., "♣", "♦"). */
    private final String symbol;
    /** {@code true} if this suit is red (Diamonds or Hearts); {@code false} if black (Clubs or Spades). */
    private final boolean red;

    /**
     * Constructs a Suit with the given symbol and color classification.
     *
     * @param symbol the Unicode symbol for the suit
     * @param red {@code true} if the suit is red; {@code false} if black
     */
    Suit(String symbol, boolean red) {
        this.symbol = symbol;
        this.red = red;
    }

    /**
     * Returns the Unicode symbol of this suit.
     *
     * @return the suit symbol (e.g., "♣", "♦", "♥", "♠")
     */
    public String getSymbol() {
        return symbol;
    }

    /**
     * Checks whether this suit is red.
     * <p>
     * Diamonds and Hearts are red; Clubs and Spades are black.
     *
     * @return {@code true} if this suit is red; {@code false} if black
     */
    public boolean isRed() {
        return red;
    }

    /**
     * Returns the string representation of this suit's symbol.
     *
     * @return the suit symbol
     */
    @Override
    public String toString() {
        return symbol;
    }

    /**
     * Returns the symbol with ANSI colour formatting if this suit is red.
     * <p>
     * Red suits (Diamonds, Hearts) are wrapped in ANSI red colour codes;
     * black suits are returned unformatted.
     *
     * @return the coloured symbol (or uncoloured if black)
     */
    public String toColoredString() {
        if (red) {
            return ANSI_RED + symbol + ANSI_RESET;
        }
        return symbol;
    }

    /**
     * Colourises the given value string using ANSI red codes if the suit is red.
     * <p>
     * This utility method is useful for colouring entire card representations
     * (e.g., "10♦") based on their suit colour.
     *
     * @param suit the suit to check for colour (may be null)
     * @param value the string value to colourise
     * @return the value wrapped in ANSI red codes if suit is red; otherwise the value unchanged
     */
    public static String colouriseIfRed(Suit suit, String value) {
        if (suit != null && suit.isRed()) {
            return ANSI_RED + value + ANSI_RESET;
        }
        return value;
    }
}

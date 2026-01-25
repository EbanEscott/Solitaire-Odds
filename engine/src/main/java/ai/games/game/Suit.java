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
    /** Unknown – placeholder for hidden cards in PLAN mode. */
    UNKNOWN("?", "?", false),
    /** Clubs – a black suit represented by the ♣ symbol. */
    CLUBS("♣", "C", false),
    /** Diamonds – a red suit represented by the ♦ symbol. */
    DIAMONDS("♦", "D", true),
    /** Hearts – a red suit represented by the ♥ symbol. */
    HEARTS("♥", "H", true),
    /** Spades – a black suit represented by the ♠ symbol. */
    SPADES("♠", "S", false);

    /** ANSI escape code for red text output in terminals. */
    private static final String ANSI_RED = "\u001B[31m";
    /** ANSI escape code to reset text formatting in terminals. */
    private static final String ANSI_RESET = "\u001B[0m";
    
    /** Flag to use ASCII fallback for systems that don't support Unicode. */
    private static final boolean USE_ASCII = Boolean.getBoolean("suit.ascii");

    /** The Unicode symbol representing this suit (e.g., "♣", "♦"). */
    private final String symbol;
    /** The ASCII fallback character for this suit (e.g., "C", "D"). */
    private final String asciiSymbol;
    /** {@code true} if this suit is red (Diamonds or Hearts); {@code false} if black (Clubs or Spades). */
    private final boolean red;

    /**
     * Constructs a Suit with the given symbol and color classification.
     *
     * @param symbol the Unicode symbol for the suit
     * @param asciiSymbol the ASCII fallback character for the suit
     * @param red {@code true} if the suit is red; {@code false} if black
     */
    Suit(String symbol, String asciiSymbol, boolean red) {
        this.symbol = symbol;
        this.asciiSymbol = asciiSymbol;
        this.red = red;
    }

    /**
     * Returns the Unicode symbol of this suit.
     *
     * @return the suit symbol (e.g., "♣", "♦", "♥", "♠") or ASCII fallback if enabled
     */
    public String getSymbol() {
        return USE_ASCII ? asciiSymbol : symbol;
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
     * @return the suit symbol (Unicode or ASCII fallback)
     */
    @Override
    public String toString() {
        return USE_ASCII ? asciiSymbol : symbol;
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
        String displaySymbol = USE_ASCII ? asciiSymbol : symbol;
        if (red) {
            return ANSI_RED + displaySymbol + ANSI_RESET;
        }
        return displaySymbol;
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

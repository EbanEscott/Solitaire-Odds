public enum Suit {
    CLUBS("♣", false),
    DIAMONDS("♦", true),
    HEARTS("♥", true),
    SPADES("♠", false);

    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_RESET = "\u001B[0m";

    private final String symbol;
    private final boolean red;

    Suit(String symbol, boolean red) {
        this.symbol = symbol;
        this.red = red;
    }

    public String getSymbol() {
        return symbol;
    }

    public boolean isRed() {
        return red;
    }

    @Override
    public String toString() {
        return symbol;
    }

    public String toColoredString() {
        if (red) {
            return ANSI_RED + symbol + ANSI_RESET;
        }
        return symbol;
    }

    public static String colorizeIfRed(Suit suit, String value) {
        if (suit != null && suit.isRed()) {
            return ANSI_RED + value + ANSI_RESET;
        }
        return value;
    }
}

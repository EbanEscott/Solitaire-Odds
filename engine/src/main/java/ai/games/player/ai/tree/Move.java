package ai.games.player.ai.tree;

import ai.games.game.Rank;
import ai.games.game.Suit;
import java.util.Locale;
import java.util.Objects;

/**
 * Structured representation of a Solitaire command.
 *
 * <p><b>Why this exists:</b> Many search components treat moves as strings (e.g. "move T6 A♠ F2").
 * Parsing strings repeatedly is error-prone (multiple formats exist) and slow.
 * This class provides:
 * <ul>
 *   <li>Fast equality/hash via a precomputed packed {@code long} key</li>
 *   <li>Robust parsing for the command formats emitted by {@code LegalMovesHelper}</li>
 *   <li>Typed accessors for source/destination piles and (optional) card</li>
 * </ul>
 *
 * <p><b>Supported command shapes:</b>
 * <ul>
 *   <li>{@code quit}</li>
 *   <li>{@code turn}</li>
 *   <li>{@code move <from> <card> <to>} (e.g. {@code move T6 A♠ F2})</li>
 *   <li>{@code move <from> <to>} (e.g. {@code move W F1} or {@code move W T3})</li>
 * </ul>
 *
 * <p><b>Notes:</b>
 * <ul>
 *   <li>We only ever turn-three in this codebase, so {@code turn} has no parameter here.</li>
 *   <li>For 3-token move forms (no explicit card token), {@link #card()} is {@code null}.</li>
 * </ul>
 */
public final class Move {

    /**
     * The command type: move, turn, or quit.
     */
    private final Type type;

    /**
     * Source pile for {@link Type#MOVE} commands; null for TURN/QUIT.
     */
    private final PileRef from;

    /**
     * Destination pile for {@link Type#MOVE} commands; null for TURN/QUIT.
     */
    private final PileRef to;

    /**
     * Explicit card token for {@link Type#MOVE} commands; may be null for 3-token move forms.
     */
    private final CardRef card;

    /**
     * Packed key for fast equality/hash lookups.
     */
    private final long key;

    public enum Type {
        MOVE,
        TURN,
        QUIT
    }

    public enum PileType {
        TABLEAU,
        FOUNDATION,
        WASTE,
        STOCK
    }

    /**
     * Typed pile reference.
     *
     * <p>Index is 0-based for TABLEAU (0..6) and FOUNDATION (0..3). For WASTE/STOCK it is -1.
     */
    public record PileRef(PileType type, int index) {
        public PileRef {
            Objects.requireNonNull(type, "type");
            if ((type == PileType.TABLEAU && (index < 0 || index > 6))
                    || (type == PileType.FOUNDATION && (index < 0 || index > 3))
                    || ((type == PileType.WASTE || type == PileType.STOCK) && index != -1)) {
                throw new IllegalArgumentException("Invalid pile index " + index + " for " + type);
            }
        }

        /**
         * Returns the engine command pile code for this pile reference.
         *
         * <p>Examples: {@code T1}, {@code F4}, {@code W}, {@code S}.
         */
        public String toCode() {
            return switch (type) {
                case TABLEAU -> "T" + (index + 1);
                case FOUNDATION -> "F" + (index + 1);
                case WASTE -> "W";
                case STOCK -> "S";
            };
        }
    }

    /**
     * Lightweight card reference for fast comparison.
     */
    public record CardRef(Rank rank, Suit suit) {
        public CardRef {
            Objects.requireNonNull(rank, "rank");
            Objects.requireNonNull(suit, "suit");
        }

        /**
         * Returns a short, normalised card token.
         *
         * <p>Examples: {@code A♠}, {@code 10♦}. Unknown returns {@code ?}.
         */
        public String shortName() {
            if (rank == Rank.UNKNOWN || suit == Suit.UNKNOWN) {
                return "?";
            }
            return rank.toString() + suit.getSymbol();
        }
    }

    /**
     * Constructs a fully-specified Move and precomputes its {@link #key()}.
     */
    private Move(Type type, PileRef from, PileRef to, CardRef card) {
        this.type = Objects.requireNonNull(type, "type");
        this.from = from;
        this.to = to;
        this.card = card;
        this.key = packKey(type, from, to, card);
    }

    /**
     * Returns the parsed command type.
     */
    public Type type() {
        return type;
    }

    /**
     * Returns true if this command is {@code move ...}.
     */
    public boolean isMove() {
        return type == Type.MOVE;
    }

    /**
     * Returns true if this command is {@code turn}.
     */
    public boolean isTurn() {
        return type == Type.TURN;
    }

    /**
     * Returns true if this command is {@code quit}.
     */
    public boolean isQuit() {
        return type == Type.QUIT;
    }

    /**
     * Returns the source pile for {@code move} commands; otherwise null.
     */
    public PileRef from() {
        return from;
    }

    /**
     * Returns the destination pile for {@code move} commands; otherwise null.
     */
    public PileRef to() {
        return to;
    }

    /**
     * The explicitly-specified card token, if present.
     *
     * <p>Some move strings omit a card token (e.g. {@code move W F1}). In those cases this is null.
     */
    public CardRef card() {
        return card;
    }

    /**
     * Precomputed packed key representing this move.
     */
    public long key() {
        return key;
    }

    /**
     * Returns a normalised command string.
     *
     * <p>MOVE commands are normalised to either:
     * <ul>
     *   <li>{@code move <from> <card> <to>} when a card token exists</li>
     *   <li>{@code move <from> <to>} when a card token is absent</li>
     * </ul>
     */
    public String toCommandString() {
        return switch (type) {
            case QUIT -> "quit";
            case TURN -> "turn";
            case MOVE -> {
                String fromCode = from != null ? from.toCode() : "";
                String toCode = to != null ? to.toCode() : "";
                if (card != null) {
                    yield "move " + fromCode + " " + card.shortName() + " " + toCode;
                }
                yield "move " + fromCode + " " + toCode;
            }
        };
    }

    /**
     * Returns the normalised command string.
     */
    @Override
    public String toString() {
        return toCommandString();
    }

    /**
     * Equality is based on the packed {@link #key()} for fast comparisons.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Move other)) {
            return false;
        }
        return this.key == other.key;
    }

    /**
     * Hash code derived from the packed {@link #key()} value.
     */
    @Override
    public int hashCode() {
        return (int) (key ^ (key >>> 32));
    }

    /**
     * Constructs a {@code quit} command.
     */
    public static Move quit() {
        return new Move(Type.QUIT, null, null, null);
    }

    /**
     * Constructs a {@code turn} command.
     */
    public static Move turn() {
        return new Move(Type.TURN, null, null, null);
    }

    /**
     * Constructs a {@code move} command.
     *
     * <p>{@code card} may be null for move strings that omit a card token.
     */
    public static Move move(PileRef from, CardRef card, PileRef to) {
        return new Move(Type.MOVE, Objects.requireNonNull(from, "from"), Objects.requireNonNull(to, "to"), card);
    }

    /**
     * Parses a command string into a {@link Move}.
     *
     * @throws IllegalArgumentException if parsing fails
     */
    public static Move parse(String command) {
        return tryParse(command);
    }

    /**
     * Attempts to parse a command string into a {@link Move}.
     *
     * @return a Move if parsing succeeds
     * @throws IllegalArgumentException if parsing fails
     */
    public static Move tryParse(String command) {
        if (command == null) {
            throw new IllegalArgumentException("Command cannot be null");
        }
        String trimmed = command.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("Command cannot be blank");
        }

        String lower = trimmed.toLowerCase(Locale.ROOT);
        if (lower.equals("quit")) {
            return quit();
        }
        if (lower.startsWith("turn")) {
            // We only support turn-three in this codebase; treat all turn variants as TURN.
            return turn();
        }

        String[] parts = trimmed.split("\\s+");
        if (parts.length < 3) {
            throw new IllegalArgumentException("Unrecognised move command: " + command);
        }
        if (!parts[0].equalsIgnoreCase("move")) {
            throw new IllegalArgumentException("Unrecognised move command: " + command);
        }

        if (parts.length == 3) {
            // "move <from> <to>" e.g. "move W F1"
            PileRef from = tryParsePile(parts[1]);
            PileRef to = tryParsePile(parts[2]);
            if (from == null || to == null) {
                throw new IllegalArgumentException("Unrecognised move command: " + command);
            }
            return new Move(Type.MOVE, from, to, null);
        }
        if (parts.length == 4) {
            // "move <from> <card> <to>" e.g. "move T6 A♠ F2"
            PileRef from = tryParsePile(parts[1]);
            CardRef card = tryParseCard(parts[2]);
            PileRef to = tryParsePile(parts[3]);
            if (from == null || to == null || card == null) {
                throw new IllegalArgumentException("Unrecognised move command: " + command);
            }
            return new Move(Type.MOVE, from, to, card);
        }

        throw new IllegalArgumentException("Unrecognised move command: " + command);
    }

    private static PileRef tryParsePile(String token) {
        if (token == null) {
            return null;
        }
        String t = token.trim().toUpperCase(Locale.ROOT);
        if (t.isEmpty()) {
            return null;
        }

        if (t.equals("W")) {
            return new PileRef(PileType.WASTE, -1);
        }
        if (t.equals("S")) {
            return new PileRef(PileType.STOCK, -1);
        }

        if (t.length() < 2) {
            return null;
        }

        char prefix = t.charAt(0);
        int idx;
        try {
            idx = Integer.parseInt(t.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return null;
        }

        return switch (prefix) {
            case 'T' -> (idx >= 0 && idx <= 6) ? new PileRef(PileType.TABLEAU, idx) : null;
            case 'F' -> (idx >= 0 && idx <= 3) ? new PileRef(PileType.FOUNDATION, idx) : null;
            default -> null;
        };
    }

    /**
     * Attempts to parse a card token such as {@code A♠} or {@code 10D}.
     */
    private static CardRef tryParseCard(String token) {
        if (token == null) {
            return null;
        }
        String t = token.trim();
        if (t.isEmpty()) {
            return null;
        }

        // Allow unknown placeholders.
        if (t.equals("?") || t.equalsIgnoreCase("UNKNOWN")) {
            return new CardRef(Rank.UNKNOWN, Suit.UNKNOWN);
        }

        // Unicode suit symbol format: "A♠", "10♦"
        char last = t.charAt(t.length() - 1);
        Suit suit = switch (last) {
            case '♣' -> Suit.CLUBS;
            case '♦' -> Suit.DIAMONDS;
            case '♥' -> Suit.HEARTS;
            case '♠' -> Suit.SPADES;
            default -> null;
        };

        String rankPart = t;
        if (suit != null) {
            rankPart = t.substring(0, t.length() - 1);
        } else {
            // Also accept two-character suit codes like AS/KH/10D.
            // If last char looks like a suit letter, treat it as such.
            char suitChar = Character.toUpperCase(last);
            suit = switch (suitChar) {
                case 'C' -> Suit.CLUBS;
                case 'D' -> Suit.DIAMONDS;
                case 'H' -> Suit.HEARTS;
                case 'S' -> Suit.SPADES;
                default -> null;
            };
            if (suit != null && t.length() >= 2) {
                rankPart = t.substring(0, t.length() - 1);
            }
        }

        if (suit == null) {
            return null;
        }

        Rank rank = parseRank(rankPart);
        if (rank == null) {
            return null;
        }

        return new CardRef(rank, suit);
    }

    /**
     * Parses a rank token ("A", "10", "K", etc.) into a {@link Rank}.
     */
    private static Rank parseRank(String token) {
        if (token == null) {
            return null;
        }
        String r = token.trim().toUpperCase(Locale.ROOT);
        if (r.isEmpty()) {
            return null;
        }
        return switch (r) {
            case "A" -> Rank.ACE;
            case "2" -> Rank.TWO;
            case "3" -> Rank.THREE;
            case "4" -> Rank.FOUR;
            case "5" -> Rank.FIVE;
            case "6" -> Rank.SIX;
            case "7" -> Rank.SEVEN;
            case "8" -> Rank.EIGHT;
            case "9" -> Rank.NINE;
            case "10" -> Rank.TEN;
            case "J" -> Rank.JACK;
            case "Q" -> Rank.QUEEN;
            case "K" -> Rank.KING;
            case "?" -> Rank.UNKNOWN;
            default -> null;
        };
    }

    /**
     * Packs all relevant move fields into a single {@code long} for fast comparisons.
     */
    private static long packKey(Type type, PileRef from, PileRef to, CardRef card) {
        // Compact key for fast comparisons.
        // Layout (LSB -> MSB):
        //   type(2)
        //   fromType(3) fromIndex+1(4)
        //   toType(3)   toIndex+1(4)
        //   rank(4) suit(3)
        int typeBits = switch (type) {
            case MOVE -> 1;
            case TURN -> 2;
            case QUIT -> 3;
        };

        int fromType = pileTypeBits(from);
        int fromIndex = pileIndexBits(from);
        int toType = pileTypeBits(to);
        int toIndex = pileIndexBits(to);

        int rank = (card == null) ? 0 : card.rank().getValue();
        int suit = (card == null) ? 0 : suitBits(card.suit());

        long key = 0L;
        key |= (long) (typeBits & 0b11);
        key |= (long) (fromType & 0b111) << 2;
        key |= (long) (fromIndex & 0b1111) << 5;
        key |= (long) (toType & 0b111) << 9;
        key |= (long) (toIndex & 0b1111) << 12;
        key |= (long) (rank & 0b1111) << 16;
        key |= (long) (suit & 0b111) << 20;
        return key;
    }

    /**
     * Encodes the pile type into a small integer for packing into {@link #key()}.
     */
    private static int pileTypeBits(PileRef ref) {
        if (ref == null) {
            return 0;
        }
        return switch (ref.type()) {
            case TABLEAU -> 1;
            case FOUNDATION -> 2;
            case WASTE -> 3;
            case STOCK -> 4;
        };
    }

    /**
     * Encodes the pile index into a small integer for packing into {@link #key()}.
     */
    private static int pileIndexBits(PileRef ref) {
        if (ref == null) {
            return 0;
        }
        // store index+1 so "no index" (W/S) becomes 0.
        return Math.max(0, ref.index() + 1);
    }

    /**
     * Encodes the suit into a small integer for packing into {@link #key()}.
     */
    private static int suitBits(Suit suit) {
        return switch (suit) {
            case UNKNOWN -> 0;
            case CLUBS -> 1;
            case DIAMONDS -> 2;
            case HEARTS -> 3;
            case SPADES -> 4;
        };
    }
}

package ai.games.unit.helpers;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Fluent builder for constructing Solitaire states for tests.
 *
 * <p><strong>Why this exists</strong>
 * <ul>
 *   <li><strong>Test readability:</strong> tests should describe a board position in a few lines.</li>
 *   <li><strong>Fail-fast correctness:</strong> seeded states that accidentally omit/duplicate cards
 *       produce misleading failures. {@link #build()} always validates aggressively.</li>
 *   <li><strong>Realism by default:</strong> the engine assumes a complete 52-card deal. This builder
 *       enforces that invariant even when a test only cares about a few piles.</li>
 *   <li><strong>Encapsulation-friendly:</strong> production code returns unmodifiable pile views;
 *       tests occasionally need to seed internals. This builder centralises the minimal reflection
 *       required so it can be audited and eventually replaced.</li>
 * </ul>
 *
 * <p><strong>How to use</strong>
 * <pre>{@code
 * Solitaire game = SolitaireBuilder
 *     .newGame()
 *     .tableau("T1", "K♠")
 *     .tableau("T2", 2, "Q♥", "J♣")
 *     .foundation("F1", "A♥")
 *     .waste("3♠", "7♦")
 *     .stock("9♣")
 *     .build();
 * }
 * </pre>
 *
 * <p><strong>Ordering conventions</strong> (all lists are <em>bottom-to-top</em>):
 * <ul>
 *   <li>{@link #tableau(String, String...)} / {@link #tableau(String, int, String...)}:
 *       first card is bottom of the pile, last is the top/visible card.</li>
 *   <li>{@link #waste(String...)}: first card is bottom of talon, last card is the top (playable) card.</li>
 *   <li>{@link #stock(String...)}: first card is bottom of stock, last card is the top (drawn first).</li>
 * </ul>
 */
public final class SolitaireBuilder {

    private final List<List<Card>> tableau = new ArrayList<>();
    private final List<Integer> faceUp = new ArrayList<>();
    private final List<List<Card>> foundation = new ArrayList<>();
    private final List<Card> stock = new ArrayList<>();
    private final List<Card> waste = new ArrayList<>();

    /**
     * Private ctor to force use of {@link #newGame()}.
     *
     * <p>We pre-initialize pile lists so callers can set piles in any order without worrying
     * about nulls or sizes.
     */
    private SolitaireBuilder() {
        for (int i = 0; i < 7; i++) {
            tableau.add(new ArrayList<>());
            faceUp.add(0);
        }
        for (int i = 0; i < 4; i++) {
            foundation.add(new ArrayList<>());
        }
    }

    /**
     * Creates a new builder.
     *
     * <p><strong>Why:</strong> this is intentionally not called "empty"; the builder produces
     * a complete game state on {@link #build()} by auto-filling unspecified cards.
     */
    public static SolitaireBuilder newGame() {
        return new SolitaireBuilder();
    }

    /**
     * Sets a tableau pile with all cards face-up.
     *
     * <p><strong>Why:</strong> most tests reason about visible suffixes only. This overload makes
     * common cases concise.
     */
    public SolitaireBuilder tableau(String pile, String... cards) {
        return tableau(pile, cards.length, cards);
    }

    /**
     * Sets a tableau pile with an explicit face-up count.
     *
     * <p><strong>Why:</strong> tests for flip behavior and hidden-card boundaries need precise
     * face-down/face-up splits. We validate early so mistakes show up at the call site.
     *
     * @param pile e.g. "T1".."T7"
     * @param faceUpCount number of cards at the end of the pile that are face-up
     * @param cards bottom-to-top card list
     */
    public SolitaireBuilder tableau(String pile, int faceUpCount, String... cards) {
        int idx = parseIndex(pile, 'T', 7);
        List<Card> p = tableau.get(idx);
        p.clear();
        for (String c : cards) {
            p.add(parseCard(c));
        }
        if (faceUpCount < 0 || faceUpCount > p.size()) {
            throw new IllegalArgumentException("Invalid faceUpCount=" + faceUpCount + " for " + pile);
        }
        if (!p.isEmpty() && faceUpCount == 0) {
            throw new IllegalArgumentException("Non-empty " + pile + " must have at least 1 face-up card");
        }
        faceUp.set(idx, faceUpCount);
        return this;
    }

    /**
     * Sets a foundation pile.
     *
     * <p><strong>Why:</strong> many tests want to start from a partially-built foundation.
     * Full legality is enforced at {@link #build()} (A..N, same suit).
     */
    public SolitaireBuilder foundation(String pile, String... cards) {
        int idx = parseIndex(pile, 'F', 4);
        List<Card> p = foundation.get(idx);
        p.clear();
        for (String c : cards) {
            p.add(parseCard(c));
        }
        return this;
    }

    /**
     * Sets the talon (waste pile).
     *
     * <p><strong>Why:</strong> talon interactions matter for turn-three and cycling logic.
     * The top card is the last element.
     */
    public SolitaireBuilder waste(String... cards) {
        waste.clear();
        for (String c : cards) {
            waste.add(parseCard(c));
        }
        return this;
    }

    /**
     * Sets the stockpile.
     *
     * <p><strong>Why:</strong> some tests require a specific next draw. The top (drawn first)
     * is the last element.
     */
    public SolitaireBuilder stock(String... cards) {
        stock.clear();
        for (String c : cards) {
            stock.add(parseCard(c));
        }
        return this;
    }

    /**
     * Builds a {@link Solitaire} instance in GAME mode and validates it.
     *
     * <p><strong>Why auto-fill?</strong> Most tests care about a small portion of the board.
     * Leaving the rest unspecified leads to illegal (non-52-card) states. Instead, any
     * unspecified cards are deterministically placed into the stockpile, keeping the state
     * complete and making tests harder to accidentally invalidate.
     *
     * <p><strong>Why reflection?</strong> The production API intentionally prevents direct
     * mutation of internal piles. Tests still need to seed positions; reflection is scoped
     * to this builder so it can be removed later without touching every test.
     */
    public Solitaire build() {
        // Start from a real instance so all internal lists exist and have correct shapes.
        Solitaire s = new Solitaire(new Deck());
        s.setMode(Solitaire.GameMode.GAME);

        // Auto-fill remaining cards into stock (bottom-to-top), preserving any explicit stock order.
        List<Card> fullDeck = fullDeckInStableOrder();
        Set<Card> specified = new HashSet<>();

        addAllSpecifiedPiles(specified, tableau);
        addAllSpecifiedPiles(specified, foundation);
        addAllSpecifiedPile(specified, stock);
        addAllSpecifiedPile(specified, waste);

        for (Card c : fullDeck) {
            if (!specified.contains(c)) {
                stock.add(c);
            }
        }

        // Reflectively seed internal state (tests only).
        setSolitaireStateViaReflection(s);

        assertValidGameState(s);
        return s;
    }

    /**
     * Applies the builder's piles into the Solitaire instance.
     *
     * <p>We set fields directly rather than going through public move APIs because:
     * (a) some tests need "impossible to reach quickly" states, and
     * (b) the builder is about describing a snapshot, not replaying a whole game.
     */
    private void setSolitaireStateViaReflection(Solitaire s) {
        setNestedListField(s, "tableau", tableau);
        setListField(s, "tableauFaceUp", new ArrayList<>(faceUp));
        setNestedListField(s, "foundation", foundation);
        setListField(s, "stockpile", new ArrayList<>(stock));
        setListField(s, "talon", new ArrayList<>(waste));
    }

    /**
     * Strict validation to keep seeded tests honest.
     *
     * <p><strong>Why strict?</strong> If a test fails because the setup was illegal, you want the
     * failure to be immediate and obvious, not a downstream "move failed" that wastes time.
     */
    /**
     * Validates a Solitaire state with the same strict invariants enforced by {@link #build()}.
     *
     * <p><strong>Why public?</strong> Many tests start from a real dealt game (or apply additional
     * mutations like reverse-moves) and still want the same "complete, coherent, 52-unique-cards"
     * validation without going through the builder.
     */
    public static void assertValidGameState(Solitaire s) {
        if (s.getMode() != Solitaire.GameMode.GAME) {
            throw new IllegalStateException("Builder only produces GAME mode states.");
        }

        // Validate face-up counts are coherent (a core invariant used by move legality).
        List<Integer> faceUpCounts = s.getTableauFaceUpCounts();
        if (faceUpCounts.size() != 7) {
            throw new IllegalStateException("Expected 7 tableau face-up counts");
        }

        List<List<Card>> tableauInternal = getNestedListField(s, "tableau");
        if (tableauInternal.size() != 7) {
            throw new IllegalStateException("Expected 7 tableau piles");
        }

        for (int i = 0; i < 7; i++) {
            List<Card> pile = tableauInternal.get(i);
            int faceUp = faceUpCounts.get(i);
            if (faceUp < 0 || faceUp > pile.size()) {
                throw new IllegalStateException("Invalid faceUp for T" + (i + 1));
            }
            if (!pile.isEmpty() && faceUp == 0) {
                throw new IllegalStateException("Non-empty T" + (i + 1) + " must have >=1 face-up card");
            }
        }

        // Validate foundation piles are strictly legal sequences (Ace..N), same suit.
        for (List<Card> pile : s.getFoundation()) {
            validateFoundationPile(pile);
        }

        // Validate full 52-card uniqueness across all piles.
        // This is the single most important guard against "partial" or duplicated-card states.
        Set<Card> unique = new HashSet<>();
        int total = 0;

        for (List<Card> pile : tableauInternal) {
            for (Card c : pile) {
                total = addCardOrThrow(unique, total, c);
            }
        }
        for (List<Card> pile : s.getFoundation()) {
            for (Card c : pile) {
                total = addCardOrThrow(unique, total, c);
            }
        }
        for (Card c : s.getStockpile()) {
            total = addCardOrThrow(unique, total, c);
        }
        for (Card c : s.getTalon()) {
            total = addCardOrThrow(unique, total, c);
        }

        if (total != 52 || unique.size() != 52) {
            throw new IllegalStateException("Expected 52 unique cards but got total=" + total + ", unique=" + unique.size());
        }

        // Disallow UNKNOWN in GAME mode: UNKNOWN is a PLAN-mode concept and would change move logic.
        for (Card c : unique) {
            if (c.getRank() == Rank.UNKNOWN || c.getSuit() == Suit.UNKNOWN) {
                throw new IllegalStateException("UNKNOWN card present in GAME mode");
            }
        }
    }

    /**
     * Adds a card into the uniqueness set with a helpful error on duplicates.
     *
     * <p><strong>Why:</strong> duplicates are the most common silent bug in hand-seeded tests.
     */
    private static int addCardOrThrow(Set<Card> unique, int total, Card c) {
        if (c == null) {
            throw new IllegalStateException("Null card in state");
        }
        total++;
        if (!unique.add(c)) {
            throw new IllegalStateException("Duplicate card in state: " + c.shortName());
        }
        return total;
    }

    /**
     * Enforces strict foundation legality: a pile is either empty or exactly A..N of one suit.
     *
     * <p><strong>Why:</strong> foundation state affects win conditions and move legality.
     * Allowing "gappy" or wrong-suit foundations makes AI/player logic hard to trust.
     */
    private static void validateFoundationPile(List<Card> pile) {
        if (pile == null || pile.isEmpty()) {
            return;
        }
        Suit suit = pile.get(0).getSuit();
        if (suit == Suit.UNKNOWN) {
            throw new IllegalStateException("Foundation suit cannot be UNKNOWN");
        }
        int expectedRank = Rank.ACE.getValue();
        for (Card c : pile) {
            if (c.getSuit() != suit) {
                throw new IllegalStateException("Foundation pile must be single-suit");
            }
            if (c.getRank().getValue() != expectedRank) {
                throw new IllegalStateException("Foundation ranks must be ascending from Ace");
            }
            expectedRank++;
        }
    }

    /**
     * Collects all explicitly specified cards from multiple piles.
     *
     * <p><strong>Why:</strong> used to compute which cards remain to auto-fill into stock.
     */
    private static void addAllSpecifiedPiles(Set<Card> specified, List<List<Card>> piles) {
        for (List<Card> pile : piles) {
            for (Card c : pile) {
                if (!specified.add(c)) {
                    throw new IllegalArgumentException("Duplicate specified card: " + c.shortName());
                }
            }
        }
    }

    /**
     * Collects all explicitly specified cards from a single pile.
     */
    private static void addAllSpecifiedPile(Set<Card> specified, List<Card> pile) {
        for (Card c : pile) {
            if (!specified.add(c)) {
                throw new IllegalArgumentException("Duplicate specified card: " + c.shortName());
            }
        }
    }

    /**
     * Returns a stable, deterministic 52-card deck ordering.
     *
     * <p><strong>Why:</strong> auto-filled cards should be predictable so tests are reproducible.
     * This is intentionally not shuffled.
     */
    static List<Card> fullDeckInStableOrder() {
        List<Card> deck = new ArrayList<>(52);
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) {
                continue;
            }
            for (Rank rank : Rank.values()) {
                if (rank == Rank.UNKNOWN) {
                    continue;
                }
                deck.add(new Card(rank, suit));
            }
        }
        return deck;
    }

    /**
     * Parses pile identifiers like "T1".."T7" or "F1".."F4".
     *
     * <p><strong>Why:</strong> using domain strings in tests is clearer than 0-based indexes.
     */
    private static int parseIndex(String code, char expectedPrefix, int max) {
        if (code == null || code.trim().isEmpty()) {
            throw new IllegalArgumentException("Missing pile code");
        }
        String trimmed = code.trim().toUpperCase();
        if (trimmed.charAt(0) != expectedPrefix) {
            throw new IllegalArgumentException("Expected pile starting with " + expectedPrefix + ": " + code);
        }
        int idx;
        try {
            idx = Integer.parseInt(trimmed.substring(1)) - 1;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid pile: " + code);
        }
        if (idx < 0 || idx >= max) {
            throw new IllegalArgumentException("Pile out of range: " + code);
        }
        return idx;
    }

    /**
     * Parses a compact card representation.
     *
    * <p>Supported format:
    * <ul>
    *   <li>Unicode suits: "Q♥", "10♣", "7♦", "K♠"</li>
    * </ul>
     *
    * <p><strong>Why:</strong> this intentionally matches the main CLI parsing in
    * [engine/src/main/java/ai/games/Game.java](engine/src/main/java/ai/games/Game.java), which passes card tokens
    * directly into {@code Solitaire.attemptMove(...)} and relies on {@code Card.shortName()} matching.
    * Keeping test parsing consistent avoids having tests accept inputs the real game would reject.
     */
    static Card parseCard(String shortName) {
        if (shortName == null || shortName.trim().isEmpty()) {
            throw new IllegalArgumentException("Empty card");
        }
        String s = shortName.trim();
        String rankPart;
        String suitPart;

        if (s.length() < 2) {
            throw new IllegalArgumentException("Invalid card: " + shortName);
        }

        // Suit is the last character (works for both unicode suits and single-letter suits).
        rankPart = s.substring(0, s.length() - 1);
        suitPart = s.substring(s.length() - 1);

        Rank rank = parseRank(rankPart);
        Suit suit = parseSuit(suitPart);
        return new Card(rank, suit);
    }

    /**
     * Parses a rank token like A, 2..10, J, Q, K.
     */
    private static Rank parseRank(String rankPart) {
        String r = rankPart.toUpperCase();
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
            default -> throw new IllegalArgumentException("Invalid rank: " + rankPart);
        };
    }

    /**
     * Parses a unicode suit symbol: ♣, ♦, ♥, ♠.
     */
    private static Suit parseSuit(String suitPart) {
        return switch (suitPart) {
            case "♣" -> Suit.CLUBS;
            case "♦" -> Suit.DIAMONDS;
            case "♥" -> Suit.HEARTS;
            case "♠" -> Suit.SPADES;
            default -> throw new IllegalArgumentException("Invalid suit: " + suitPart);
        };
    }

    /**
     * Reads a private nested-list field from {@link Solitaire}.
     *
     * <p><strong>Why:</strong> {@link Solitaire} exposes unmodifiable views for encapsulation.
     * Tests need read access here for validation.
     */
    @SuppressWarnings("unchecked")
    private static List<List<Card>> getNestedListField(Solitaire solitaire, String fieldName) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (List<List<Card>>) field.get(solitaire);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to read field '" + fieldName + "'", e);
        }
    }

    /**
     * Sets a private nested-list field on {@link Solitaire} by deep-copying each pile.
     *
     * <p><strong>Why deep-copy?</strong> prevents accidental mutation of the builder's internal
     * lists after build.
     */
    @SuppressWarnings("unchecked")
    private static void setNestedListField(Solitaire solitaire, String fieldName, List<List<Card>> newValue) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            List<List<Card>> target = (List<List<Card>>) field.get(solitaire);
            target.clear();
            for (List<Card> pile : newValue) {
                target.add(new ArrayList<>(pile));
            }
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to set field '" + fieldName + "'", e);
        }
    }

    /**
     * Sets a private list field on {@link Solitaire}.
     */
    @SuppressWarnings("unchecked")
    private static <T> void setListField(Solitaire solitaire, String fieldName, List<T> newValue) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            List<T> target = (List<T>) field.get(solitaire);
            target.clear();
            target.addAll(newValue);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to set field '" + fieldName + "'", e);
        }
    }
}

package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Complete model of a Solitaire/Klondike game state and board layout.
 * <p>
 * Represents the complete game board including seven tableau piles, four foundation piles,
 * a stockpile of undealt cards, and a talon (waste pile) for revealed cards. Provides
 * methods for game play (moving cards, turning stock), state inspection (visibility-aware
 * views), and cycle detection (Zobrist hashing).
 * <p>
 * <strong>Game Layout:</strong>
 * <ul>
 *   <li><strong>Tableau (T1–T7):</strong> Seven piles with mixed face-up and face-down cards.
 *       Each pile tracks how many of its cards are visible (the rest are hidden).</li>
 *   <li><strong>Foundation (F1–F4):</strong> Four piles, one per suit, built from Ace to King.</li>
 *   <li><strong>Stockpile (S):</strong> Remaining undealt cards (face-down). When empty,
 *       the talon can be recycled back.</li>
 *   <li><strong>Talon (W):</strong> Cards revealed from the stockpile during \"turn three\" operations.
 *       Per Klondike Turn-3 rules: three cards are drawn and placed face-up into the talon.
 *       All three cards are strategically visible to the player, but only the top card is legally
 *       playable to the foundations or tableau at any given time. Use {@link #getTalon()} to inspect
 *       all cards in the talon (important for AI planning), and {@link LegalMovesHelper} enforces
 *       that only the top card can actually be moved.</li>
 * </ul>
 * <p>
 * <strong>Card Visibility:</strong>
 * Hidden cards in the tableau are not exposed to AI players or move validation checks.
 * Use {@link #getVisibleTableau()} for AI decision-making and {@link #getTableauFaceDownCounts()}
 * to know how many cards are hidden in each pile. The talon, by contrast, is fully visible (all
 * cards dealt face-up per Klondike rules), and AI players have full access to all three cards via
 * {@link #getTalon()} for strategic planning, even though only the top card is playable.
 * <p>
 * <strong>Move Validation:</strong>
 * All moves are validated against Solitaire rules:
 * <ul>
 *   <li>Tableau moves require alternating colours and descending ranks.</li>
 *   <li>Foundation moves require matching suits and ascending ranks from Ace.</li>
 *   <li>Stacks can be moved from tableau but only single cards from stockpile/talon/foundation.</li>
 * </ul>
 */
public class Solitaire {
    private static final long[] STATE_ZOBRIST = initStateZobrist();

    /** Tableau: seven piles where most play occurs; may contain both face-up and face-down cards. */
    private final List<List<Card>> tableau = new ArrayList<>();
    
    /** Face-up card counts per tableau pile; visible cards are the last N cards in each pile. */
    private final List<Integer> tableauFaceUp = new ArrayList<>();
    
    /** Foundation: four suit piles built up from Ace to King. */
    private final List<List<Card>> foundation = new ArrayList<>();
    
    /** Stockpile: undealt cards remaining after the tableau is dealt (face-down). */
    private final List<Card> stockpile = new ArrayList<>();
    
    /** Talon: revealed cards from the stockpile during play. */
    private final List<Card> talon = new ArrayList<>();
    
    /** Action history for replay and undo operations (training mode). */
    private final List<Action> moveHistory = new ArrayList<>();
    
    /** Original deck order captured at game start for deterministic replay during undo. */
    private final List<Card> originalDeckOrder = new ArrayList<>();


    /**
     * Constructs a new Solitaire game with a shuffled deck.
     * <p>
     * Initialises the game by:
     * <ol>
     *   <li>Setting up four empty foundation piles (one per suit).</li>
     *   <li>Dealing cards into the seven tableau piles (pile i gets i+1 cards, with only the top visible).</li>
     *   <li>Placing all remaining cards into the stockpile (face-down).</li>
     * </ol>
     *
     * @param deck the shuffled deck to initialise from; must not be null
     * @throws NullPointerException if deck is null
     */
    public Solitaire(Deck deck) {
        Objects.requireNonNull(deck, "deck");
        // Capture the original deck order for deterministic replay during undo.
        originalDeckOrder.addAll(deck.asUnmodifiableList());
        initializeFoundation();
        dealTableau(deck);
        moveRemainingToStockpile(deck);
    }

    /**
     * Creates a deep copy of this Solitaire state for simulation purposes.
     * <p>
     * Card instances are immutable and reused; all pile and count lists are deep-copied
     * so that modifications to the clone do not affect the original state. This is essential
     * for AI search algorithms that simulate moves without altering the current game state.
     *
     * @return a new Solitaire instance with identical state but independent pile lists
     */
    public Solitaire copy() {
        Deck dummy = new Deck();
        Solitaire clone = new Solitaire(dummy);
        // Clear dealt state.
        clone.tableau.clear();
        clone.tableauFaceUp.clear();
        clone.foundation.clear();
        clone.stockpile.clear();
        clone.talon.clear();

        // Copy tableau and face-up counts.
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            clone.tableau.add(new ArrayList<>(pile));
        }
        clone.tableauFaceUp.addAll(tableauFaceUp);

        // Copy foundation.
        for (List<Card> pile : foundation) {
            clone.foundation.add(new ArrayList<>(pile));
        }

        // Copy stock and talon.
        clone.stockpile.addAll(stockpile);
        clone.talon.addAll(talon);

        return clone;
    }

    /**
     * Returns full tableau piles including facedown cards.
     * Engine/UI only; AIs should prefer {@link #getVisibleTableau()}.
     */
    @Deprecated
    public List<List<Card>> getTableau() {
        return unmodifiablePiles(tableau);
    }

    /**
     * Returns a visibility-safe view of the tableau where each pile contains only
     * the face-up (visible) suffix. Hidden cards are excluded and the returned
     * lists are immutable snapshots.
     */
    public List<List<Card>> getVisibleTableau() {
        List<List<Card>> visible = new ArrayList<>(tableau.size());
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = i < tableauFaceUp.size() ? tableauFaceUp.get(i) : 0;
            if (pile.isEmpty() || faceUp <= 0) {
                visible.add(Collections.emptyList());
                continue;
            }
            int start = Math.max(0, pile.size() - faceUp);
            List<Card> suffix = new ArrayList<>(pile.subList(start, pile.size()));
            visible.add(Collections.unmodifiableList(suffix));
        }
        return Collections.unmodifiableList(visible);
    }

    /**
     * Returns an immutable view of all foundation piles (F1–F4).
     * <p>
     * Each foundation pile is in ascending order from Ace to King for a single suit.
     *
     * @return an unmodifiable list of four foundation piles
     */
    public List<List<Card>> getFoundation() {
        return unmodifiablePiles(foundation);
    }

    /**
     * Returns an immutable view of the stockpile (undealt cards, face-down).
     * <p>
     * The stockpile is a LIFO stack; the last card added is the first to be drawn.
     * When empty and the talon is not empty, the talon can be recycled back into the stockpile.
     *
     * @return an unmodifiable list representing the stockpile
     */
    public List<Card> getStockpile() {
        return Collections.unmodifiableList(stockpile);
    }

    /**
     * Returns an immutable view of the talon (revealed cards from the stockpile).
     * <p>
     * The talon grows as cards are turned from the stockpile and shrinks as cards are moved
     * to the tableau or foundation. The top card (last in the list) is the most recently revealed.
     *
     * @return an unmodifiable list representing the talon
     */
    public List<Card> getTalon() {
        return Collections.unmodifiableList(talon);
    }

    /**
     * Returns the count of face-up (visible) cards for each tableau pile.
     * Visible cards are the last N cards in the corresponding internal pile.
     */
    public List<Integer> getTableauFaceUpCounts() {
        return Collections.unmodifiableList(new ArrayList<>(tableauFaceUp));
    }

    /**
     * Returns the count of facedown (hidden) cards for each tableau pile,
     * derived as total size minus the face-up count.
     */
    public List<Integer> getTableauFaceDownCounts() {
        List<Integer> faceDown = new ArrayList<>(tableau.size());
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            int faceUp = i < tableauFaceUp.size() ? tableauFaceUp.get(i) : 0;
            int count = Math.max(0, pile.size() - faceUp);
            faceDown.add(count);
        }
        return Collections.unmodifiableList(faceDown);
    }

    /**
     * Turns three cards from the stockpile onto the talon, preserving order of draw.
     * <p>
     * If the stockpile is empty but the talon is not, recycles the talon back into the stockpile
     * (face-down, in reverse order so the top of the talon becomes the bottom of the new stock)
     * before turning. Then turns up to three cards from the stockpile onto the talon.
     * <p>
     * This operation is central to Solitaire play and can be called repeatedly to cycle through
     * the deck.
     */
    public void turnThree() {
        if (stockpile.isEmpty() && !talon.isEmpty()) {
            // Recycle talon back to stockpile: top of talon becomes bottom of new stock.
            for (int i = talon.size() - 1; i >= 0; i--) {
                stockpile.add(talon.get(i));
            }
            talon.clear();
        }

        for (int i = 0; i < 3 && !stockpile.isEmpty(); i++) {
            talon.add(stockpile.remove(stockpile.size() - 1));
        }
    }

    /**
     * Attempts to move a card (and any cards above it) from one pile to another.
     * <p>
     * Pile codes are single-letter codes followed by an optional index:
     * <ul>
     *   <li>T1–T7: Tableau piles (can move stacks if rules allow)</li>
     *   <li>F1–F4: Foundation piles (single card only)</li>
     *   <li>S: Stockpile (single card only)</li>
     *   <li>W: Waste/Talon (single card only)</li>
     * </ul>
     * <p>
     * If cardCode is null, uses the top visible card for tableau sources or the top card for others.
     * Card codes are case-insensitive (e.g., \"Q♣\" or \"qc\").
     *
     * @param from the source pile code (e.g., \"T1\", \"W\"); must not be null
     * @param cardCode the card to move (optional for moving the top card); may be null
     * @param to the destination pile code (e.g., \"F1\", \"T3\"); must not be null
     * @return true if the move succeeded, false otherwise
     */
    public boolean moveCard(String from, String cardCode, String to) {
        MoveResult result = attemptMove(from, cardCode, to);
        return result.success;
    }

    /**
     * Attempts to move a card and returns a descriptive result for error reporting.
     * <p>
     * This method performs full move validation, including:
     * <ul>
     *   <li>Pile code parsing and validation</li>
     *   <li>Card visibility checks (face-up cards in tableau, top cards elsewhere)</li>
     *   <li>Solitaire rule validation (alternating colours, matching suits, ascending ranks)</li>
     *   <li>State updates if the move is legal (pile contents, face-up counts, talon recycling)</li>
     * </ul>
     * <p>
     * The returned {@link MoveResult} contains both a success flag and a descriptive message
     * (e.g., \"Moved Q♣ from T1 to F1\" or \"Card Q♠ is not visible in T1\").
     *
     * @param from the source pile code (e.g., \"T1\", \"W\"); must not be null
     * @param cardCode the card to move (optional for moving the top card); may be null
     * @param to the destination pile code (e.g., \"F1\", \"T3\"); must not be null
     * @return a MoveResult containing success flag and descriptive message; never null
     */
    public MoveResult attemptMove(String from, String cardCode, String to) {
        String fromNormalized = normalizeCode(from);
        String toNormalized = normalizeCode(to);
        if (fromNormalized == null) {
            return MoveResult.failure("Invalid source pile code: " + from);
        }
        if (toNormalized == null) {
            return MoveResult.failure("Invalid destination pile code: " + to);
        }

        char fromType = fromNormalized.charAt(0);
        char toType = toNormalized.charAt(0);
        int fromIndex = parseIndex(fromNormalized);
        int toIndex = parseIndex(toNormalized);

        List<Card> fromPile = resolvePile(fromNormalized);
        List<Card> toPile = resolvePile(toNormalized);
        if (fromPile == null) {
            return MoveResult.failure("Unknown source pile: " + fromNormalized);
        }
        if (toPile == null) {
            return MoveResult.failure("Unknown destination pile: " + toNormalized);
        }
        if (fromPile.isEmpty()) {
            return MoveResult.failure("Source pile " + fromNormalized + " is empty.");
        }

        int faceUp = fromType == 'T' ? tableauFaceUp.get(fromIndex) : 1;
        if (fromType == 'T' && faceUp <= 0) {
            return MoveResult.failure("No face-up cards available in " + fromNormalized + ".");
        }
        int startFaceUpIdx = fromType == 'T' ? Math.max(0, fromPile.size() - faceUp) : fromPile.size() - 1;
        int movingIdx = fromPile.size() - 1; // default top card
        if (cardCode != null) {
            movingIdx = -1;
            for (int i = startFaceUpIdx; i < fromPile.size(); i++) {
                if (fromPile.get(i).matchesShortName(cardCode)) {
                    movingIdx = i;
                    break;
                }
            }
            if (movingIdx == -1) {
                return MoveResult.failure("Card " + cardCode + " is not visible in " + fromNormalized + ".");
            }
        }
        // Non-tableau sources must move the top card only.
        if (fromType != 'T' && movingIdx != fromPile.size() - 1) {
            return MoveResult.failure("Can only move the top card from " + fromNormalized + ".");
        }

        Card moving = fromPile.get(movingIdx);
        if (!isLegalMove(moving, toNormalized, toPile)) {
            String reason = toType == 'T'
                    ? "Tableau requires alternating color and one rank lower."
                    : "Foundation requires same suit and one rank higher starting with Ace.";
            return MoveResult.failure("Cannot place " + moving.shortName() + " on " + toNormalized + ". " + reason);
        }

        // Foundation accepts only a single top card.
        if (toType == 'F' && movingIdx != fromPile.size() - 1) {
            return MoveResult.failure("Foundation only accepts the top card, not a stack.");
        }

        int movedCount = fromPile.size() - movingIdx;
        List<Card> segment = new ArrayList<>(fromPile.subList(movingIdx, fromPile.size()));
        for (int i = fromPile.size() - 1; i >= movingIdx; i--) {
            fromPile.remove(i);
        }
        toPile.addAll(segment);

        if (fromType == 'T') {
            int remainingFaceUp = Math.max(0, tableauFaceUp.get(fromIndex) - movedCount);
            if (remainingFaceUp == 0 && !fromPile.isEmpty()) {
                remainingFaceUp = 1; // flip next card
            }
            tableauFaceUp.set(fromIndex, remainingFaceUp);
        }

        if (toType == 'T' && toIndex >= 0 && toIndex < tableauFaceUp.size()) {
            tableauFaceUp.set(toIndex, tableauFaceUp.get(toIndex) + movedCount);
        }
        
        // Record the move for undo functionality (training mode).
        recordAction(Action.move(from, moving.shortName(), to));
        
        return MoveResult.success("Moved " + moving.shortName() + " from " + fromNormalized + " to " + toNormalized + ".");
    }

    /**
     * Backward-compatible convenience method to move the top/visible card from one pile to another.
     * <p>
     * Equivalent to {@link #moveCard(String, String, String)} with cardCode = null.
     *
     * @param from the source pile code (e.g., \"T1\", \"W\"); must not be null
     * @param to the destination pile code (e.g., \"F1\", \"T3\"); must not be null
     * @return true if the move succeeded, false otherwise
     */
    public boolean moveCard(String from, String to) {
        return moveCard(from, null, to);
    }

    /**
     * Initialises all four foundation piles as empty lists.
     * <p>
     * Called during construction to set up the foundation structure before dealing cards.
     */
    private void initializeFoundation() {
        for (int i = 0; i < 4; i++) {
            foundation.add(new ArrayList<>());
        }
    }

    /**
     * Deals cards from the deck into the seven tableau piles following Solitaire rules.
     * <p>
     * Pile i (0-indexed) receives i+1 cards, with only the topmost card initially visible.
     * The remaining cards in each pile are face-down. This creates the classic Solitaire layout.
     *
     * @param deck the deck to deal from
     */
    private void dealTableau(Deck deck) {
        for (int pileIndex = 0; pileIndex < 7; pileIndex++) {
            List<Card> pile = new ArrayList<>();
            for (int cardCount = 0; cardCount <= pileIndex && !deck.isEmpty(); cardCount++) {
                pile.add(deck.draw());
            }
            tableau.add(pile);
            tableauFaceUp.add(pile.isEmpty() ? 0 : 1); // only the top card is face up initially
        }
    }

    /**
     * Moves all remaining cards from the deck into the stockpile.
     * <p>
     * Called after the tableau is dealt to initialise the stockpile with face-down cards.
     *
     * @param deck the deck to drain
     */
    private void moveRemainingToStockpile(Deck deck) {
        while (!deck.isEmpty()) {
            stockpile.add(deck.draw());
        }
    }

    /**
     * Returns an opaque 64-bit key representing the full internal game state.
     * <p>
     * This key uses Zobrist hashing to create a pseudo-unique fingerprint of the entire game state
     * (tableau, foundation, stockpile, talon) including both visible and hidden cards.
     * The state key is intended for AI cycle detection and transposition tables, allowing
     * search algorithms to detect when they have returned to a previously-visited state.
     * <p>
     * <strong>Important:</strong> The state key does not expose hidden card identities directly;
     * it only contributes to a hash. Different hidden card arrangements may produce different keys.
     *
     * @return a 64-bit Zobrist hash of the entire game state
     */
    public long getStateKey() {
        long h = 0L;
        int idx = 0;

        // Tableau piles (include hidden + visible).
        for (List<Card> pile : tableau) {
            for (Card c : pile) {
                h ^= cardSlotKey(idx++, c);
            }
        }

        // Foundation piles.
        for (List<Card> pile : foundation) {
            for (Card c : pile) {
                h ^= cardSlotKey(idx++, c);
            }
        }

        // Stockpile.
        for (Card c : stockpile) {
            h ^= cardSlotKey(idx++, c);
        }

        // Talon.
        for (Card c : talon) {
            h ^= cardSlotKey(idx++, c);
        }

        // Encode stock vs talon presence as a simple extra bit of state.
        if (stockpile.isEmpty() && !talon.isEmpty()) {
            h ^= STATE_ZOBRIST[STATE_ZOBRIST.length - 1];
        }

        return h;
    }

    /**
     * Wraps piles in unmodifiable lists to prevent external mutation.
     * <p>
     * Creates snapshots of pile lists, protecting internal game state from accidental
     * or malicious modification via returned references.
     *
     * @param piles the piles to wrap; must not be null
     * @return an unmodifiable list of unmodifiable piles
     */
    private List<List<Card>> unmodifiablePiles(List<List<Card>> piles) {
        List<List<Card>> snapshot = new ArrayList<>(piles.size());
        for (List<Card> pile : piles) {
            snapshot.add(Collections.unmodifiableList(pile));
        }
        return Collections.unmodifiableList(snapshot);
    }

    /**
     * Renders the complete game board as a formatted multi-line string.
     * <p>
     * Delegates to {@link BoardFormatter} to generate a human-readable display showing
     * foundation, tableau, and stockpile/talon with proper borders, alignment, and colour codes.
     *
     * @return a formatted board representation
     */
    @Override
    public String toString() {
        return new BoardFormatter(this).format();
    }

    /**
     * Resolves a pile code string to the actual internal pile list.
     * <p>
     * Supports pile codes: T1–T7 (tableau), F1–F4 (foundation), S (stockpile), W (waste/talon).
     * Case-insensitive. Codes like \"T3\" are parsed as type='T', index=2.
     *
     * @param code the pile code (e.g., \"T1\", \"F2\", \"W\", \"S\"); may be null
     * @return the pile list, or null if the code is invalid or resolves to a non-existent pile
     */
    private List<Card> resolvePile(String code) {
        if (code == null || code.isEmpty()) {
            return null;
        }
        String normalized = code;
        char type = normalized.charAt(0);
        int index = 0;
        if (normalized.length() > 1) {
            try {
                index = Integer.parseInt(normalized.substring(1)) - 1;
            } catch (NumberFormatException e) {
                return null;
            }
        }

        switch (type) {
            case 'T':
                return (index >= 0 && index < tableau.size()) ? tableau.get(index) : null;
            case 'F':
                return (index >= 0 && index < foundation.size()) ? foundation.get(index) : null;
            case 'W': // Waste / talon
                return talon;
            case 'S': // Stockpile
                return stockpile;
            default:
                return null;
        }
    }

    /**
     * Normalises a pile code string by trimming whitespace and converting to uppercase.
     * <p>
     * Makes pile codes case-insensitive (e.g., \"t1\", \"T1\", \"T 1\" all become \"T1\").
     *
     * @param code the pile code to normalise; may be null
     * @return the normalised code, or null if the input is null or empty/whitespace-only
     */
    private String normalizeCode(String code) {
        if (code == null || code.trim().isEmpty()) {
            return null;
        }
        return code.trim().toUpperCase();
    }

    /**
     * Parses the numeric index from a normalised pile code (1-indexed input, 0-indexed output).
     * <p>
     * Codes like \"T3\" extract index 3, which is converted to array index 2.
     * Codes with no numeric suffix default to 0 (first index).
     *
     * @param normalized the normalised pile code (e.g., \"T1\", \"F3\", \"W\"); may be null
     * @return the zero-based index, or -1 if parsing fails
     */
    private int parseIndex(String normalized) {
        if (normalized == null || normalized.length() <= 1) {
            return 0;
        }
        try {
            return Integer.parseInt(normalized.substring(1)) - 1;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /**
     * Validates whether a card can legally be placed on a destination pile.
     * <p>
     * Rules differ by destination type:
     * <ul>
     *   <li><strong>Tableau:</strong> Alternating colours (red/black) and one rank lower.</li>
     *   <li><strong>Foundation:</strong> Matching suit and one rank higher (Ace is lowest).</li>
     * </ul>
     *
     * @param moving the card being moved; must not be null
     * @param toCode the destination pile code (e.g., \"T1\", \"F1\"); must not be null
     * @param toPile the destination pile; must not be null
     * @return true if the move is legal according to Solitaire rules
     */
    private boolean isLegalMove(Card moving, String toCode, List<Card> toPile) {
        if (moving == null || toCode == null) {
            return false;
        }
        String normalized = toCode.trim().toUpperCase();
        if (normalized.isEmpty()) {
            return false;
        }

        char type = normalized.charAt(0);
        switch (type) {
            case 'T':
                return canPlaceOnTableau(moving, toPile);
            case 'F':
                return canPlaceOnFoundation(moving, toPile);
            default:
                return false;
        }
    }

    /**
     * Checks if a card can be placed on a tableau pile.
     * <p>
     * Tableau rules: empty piles only accept Kings; non-empty piles require alternating colours
     * and the moving card to be exactly one rank lower than the top card.
     *
     * @param moving the card being moved; must not be null
     * @param toPile the tableau pile; must not be null
     * @return true if the move is legal for the tableau
     */
    private boolean canPlaceOnTableau(Card moving, List<Card> toPile) {
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.KING;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean alternatingColor = moving.getSuit().isRed() != target.getSuit().isRed();
        boolean oneLower = moving.getRank().getValue() == target.getRank().getValue() - 1;
        return alternatingColor && oneLower;
    }

    /**
     * Checks if a card can be placed on a foundation pile.
     * <p>
     * Foundation rules: empty piles only accept Aces; non-empty piles require matching suit
     * and the moving card to be exactly one rank higher than the top card.
     *
     * @param moving the card being moved; must not be null
     * @param toPile the foundation pile; must not be null
     * @return true if the move is legal for the foundation
     */
    private boolean canPlaceOnFoundation(Card moving, List<Card> toPile) {
        if (toPile.isEmpty()) {
            return moving.getRank() == Rank.ACE;
        }
        Card target = toPile.get(toPile.size() - 1);
        boolean sameSuit = moving.getSuit() == target.getSuit();
        boolean oneHigher = moving.getRank().getValue() == target.getRank().getValue() + 1;
        return sameSuit && oneHigher;
    }

    /**
     * Encapsulates the result of a move attempt.
     * <p>
     * Provides both a success indicator and a descriptive message for user feedback
     * (e.g., \"Moved Q♣ from T1 to F1\" or \"Tableau requires alternating colour...\").
     */
    public static class MoveResult {
        /** Whether the move attempt succeeded. */
        public final boolean success;
        
        /** Descriptive message (success or error details). */
        public final String message;

        /**
         * Constructs a MoveResult with the given outcome and message.
         *
         * @param success whether the move succeeded
         * @param message the descriptive message; must not be null
         */
        private MoveResult(boolean success, String message) {
            this.success = success;
            this.message = message;
        }

        /**
         * Creates a successful MoveResult with the given message.
         *
         * @param message a descriptive success message; must not be null
         * @return a successful MoveResult
         */
        public static MoveResult success(String message) {
            return new MoveResult(true, message);
        }

        /**
         * Creates a failed MoveResult with the given error message.
         *
         * @param message a descriptive error message; must not be null
         * @return a failed MoveResult
         */
        public static MoveResult failure(String message) {
            return new MoveResult(false, message);
        }
    }

    /**
     * Initialises the Zobrist hashing table with pseudo-random 64-bit values.
     * <p>
     * Zobrist hashing is a technique for creating hash values from complex game states.
     * This table is indexed by (slot index, card ordinal) pairs to produce unique hash contributions
     * for each card in each position. The same table is reused across all Solitaire instances.
     *
     * @return an array of 64-bit Zobrist hash values (4096 + 1 entries)
     */
    private static long[] initStateZobrist() {
        // Enough entries for all possible card slots plus one extra flag.
        int size = 64 * 64 + 1;
        long[] table = new long[size];
        long seed = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < size; i++) {
            seed ^= (seed << 7);
            seed ^= (seed >>> 9);
            seed ^= (seed << 8);
            table[i] = seed;
        }
        return table;
    }

    /**
     * Computes the Zobrist hash contribution for a card at a specific slot.
     * <p>
     * Maps (slot index, card) pairs to entries in the Zobrist table via modulo arithmetic,
     * allowing efficient XOR-based incremental hashing during game state transitions.
     *
     * @param slotIndex the slot position (incremented for each card in game state order)
     * @param card the card occupying the slot; must not be null
     * @return a 64-bit hash value to XOR into the running state hash
     */
    private long cardSlotKey(int slotIndex, Card card) {
        int ordinal = card.getSuit().ordinal() * Rank.values().length + card.getRank().ordinal();
        int index = (slotIndex * 64 + ordinal) % (STATE_ZOBRIST.length - 1);
        return STATE_ZOBRIST[index];
    }

    /**
     * Records a single action (move or turn) for replay and undo operations (training mode).
     * <p>
     * Stores either a card move (with source, card code, and destination) or a turn action
     * to enable move history replay. When undoing, the game is reconstructed from the start
     * by replaying all actions except the last.
     *
     * @param type the action type: "move" for card moves, "turn" for stock turns
     * @param from the source pile code for moves ('T1'–'T7', 'F1'–'F4', 'W', 'S'); null for turns
     * @param cardCode the compact card code for moves ('R♠', 'A♥', etc.); null for turns
     * @param to the destination pile code for moves; null for turns
     * @since 1.1
     */
    public record Action(String type, String from, String cardCode, String to) {
        /**
         * Constructs an Action record.
         * @param type action type ("move" or "turn")
         * @param from source pile code (for moves) or null
         * @param cardCode moving card code (for moves) or null
         * @param to destination pile code (for moves) or null
         */
        public Action {
            Objects.requireNonNull(type, "type");
            if (!"move".equals(type) && !"turn".equals(type)) {
                throw new IllegalArgumentException("Invalid action type: " + type);
            }
        }

        /**
         * Creates a move action.
         * @param from source pile code
         * @param cardCode moving card code
         * @param to destination pile code
         * @return a new move action
         */
        public static Action move(String from, String cardCode, String to) {
            return new Action("move", from, cardCode, to);
        }

        /**
         * Creates a turn action.
         * @return a new turn action
         */
        public static Action turn() {
            return new Action("turn", null, null, null);
        }
    }

    /**
     * Records an action for later replay (training mode).
     * <p>
     * Called after each successful move or turn. If undo is needed, the game replays from the start
     * excluding the last action in the history.
     *
     * @param action the action to record
     */
    public void recordAction(Action action) {
        moveHistory.add(action);
    }


    /**
     * Undoes the last move or turn by replaying the entire game history minus the final action.
     * <p>
     * Implementation (Option B – Move History Replay):
     * <ol>
     *   <li>Clear all piles and reset to initial deal state.</li>
     *   <li>Replay all actions in moveHistory except the last one.</li>
     *   <li>Remove the last action from the history.</li>
     * </ol>
     * This approach trades memory efficiency for simplicity and correctness; a future
     * hybrid approach (Option C) could add a limited state stack for recent moves.
     *
     * @return true if undo succeeded, false if no moves to undo
     * @throws IllegalStateException if replay encounters an invalid move (should not occur
     *         if moveHistory is correctly maintained)
     */
    public boolean undoLastMove() {
        if (moveHistory.isEmpty()) {
            return false;
        }

        // Create a deck with the exact same card order captured at game start.
        Deck replayDeck = new Deck(originalDeckOrder);
        
        // Record all but the last action.
        List<Action> replayActions = new ArrayList<>(moveHistory);
        replayActions.remove(replayActions.size() - 1);

        // Reset to initial state.
        tableau.clear();
        tableauFaceUp.clear();
        foundation.clear();
        stockpile.clear();
        talon.clear();

        // Re-deal from scratch using the original deck order.
        initializeFoundation();
        dealTableau(replayDeck);
        moveRemainingToStockpile(replayDeck);

        // Replay all non-undone actions.
        for (Action action : replayActions) {
            if ("move".equals(action.type())) {
                MoveResult result = attemptMove(action.from(), action.cardCode(), action.to());
                if (!result.success) {
                    throw new IllegalStateException("Action replay failed: " + result.message);
                }
            } else if ("turn".equals(action.type())) {
                turnThree();
            }
        }

        // Remove the last action from history.
        moveHistory.remove(moveHistory.size() - 1);
        return true;
    }

    /**
     * Checks if undo is currently possible.
     *
     * @return true if there is at least one move to undo, false otherwise
     */
    public boolean canUndo() {
        return !moveHistory.isEmpty();
    }

    /**
     * Returns the number of moves in the current history.
     *
     * @return the size of moveHistory
     */
    public int getMoveHistorySize() {
        return moveHistory.size();
    }
}


package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Complete model of a Solitaire/Klondike game state and board layout.
 * <p>
 * Represents the complete game board including seven tableau piles, four foundation piles,
 * a stockpile of undealt cards, and a talon (waste pile) for revealed cards. Provides
 * methods for game play (moving cards, turning stock), state inspection (visibility-aware
 * views), and cycle detection (Zobrist hashing).
 * <p>
 * <strong>Game Mode:</strong>
 * Each Solitaire instance operates in one of two modes:
 * <ul>
 *   <li><strong>GAME:</strong> The actual game being played. All card information is real.</li>
 *   <li><strong>PLAN:</strong> A lookahead copy for AI planning. Face-down cards are masked
 *       with UNKNOWN placeholders to prevent information leaks during search.</li>
 * </ul>
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
 *       all cards in the talon (important for AI planning), and {@link ai.games.player.LegalMovesHelper} enforces
 *       that only the top card can actually be moved.</li>
 * </ul>
 * <p>
 * <strong>Card Visibility:</strong>
 * GAME states retain the actual hidden identities. Use {@link #planningCopy()} before
 * exploring positions using only player knowledge.
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
    /**
     * Enum representing the operational mode of a Solitaire instance.
     * <ul>
     *   <li><strong>GAME:</strong> Real game in progress. All card information is visible as appropriate.</li>
     *   <li><strong>PLAN:</strong> Lookahead copy for AI planning. Face-down cards are masked with UNKNOWN.</li>
     * </ul>
     */
    public enum GameMode {
        /** Real game in progress. */
        GAME,
        /** Lookahead copy with masked face-down cards. */
        PLAN
    }

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
    
    /** Completed actions, including the changes needed to undo them in reverse order. */
    private final List<Action> moveHistory = new ArrayList<>();
    
    /** Tracks which cards have been revealed during play (via stock/talon cycling). */
    private final UnknownCardTracker unknownCardTracker = new UnknownCardTracker();

    /** Assumptions belong to individual UNKNOWN positions, whose cards compare equal by value. */
    private final Map<Card, UnknownCardGuess> unknownCardGuesses = new IdentityHashMap<>();
    
    /** Immutable PLAN identity pool; copies share snapshots until an assignment changes one. */
    private List<Card> remainingUnknownCards;

    /** Current operational mode (GAME or PLAN). */
    private final GameMode mode;

    /** Immutable rules shared by this game and its copies. */
    private final GameRules rules;

    /** Current stock traversal, starting at one and increasing only when waste is recycled. */
    private long stockPass = 1;

    /** Lazily computed board hash; zero is a valid key, so validity is tracked separately. */
    private long cachedStateKey;
    private boolean stateKeyValid;

    /**
     * Creates empty containers for a copy without shuffling or dealing a deck.
     *
     * @param mode the fixed operational mode for the new instance
     * @param rules the immutable configuration inherited from the source game
     * @throws NullPointerException if rules is null
     */
    private Solitaire(GameMode mode, GameRules rules) {
        this.mode = mode;
        this.rules = Objects.requireNonNull(rules, "rules");
    }

    /**
     * Constructs a new Solitaire game with a shuffled deck.
     * <p>
     * Captures JVM rule properties once through {@link GameRules#configured()}.
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
     * @throws IllegalArgumentException if the configured rule properties are invalid
     */
    public Solitaire(Deck deck) {
        this(deck, GameRules.configured());
    }

    /**
     * Constructs and deals a game using explicit rules, independently of JVM properties.
     *
     * @param deck the deck to deal from; must not be null
     * @param rules the immutable game configuration; must not be null
     * @throws NullPointerException if deck or rules is null
     */
    public Solitaire(Deck deck, GameRules rules) {
        this(GameMode.GAME, rules);
        Objects.requireNonNull(deck, "deck");
        initializeFoundation();
        dealTableau(deck);
        moveRemainingToStockpile(deck);
    }

    /**
     * Copies the current position, mode, revealed-card knowledge, and guesses.
     * <p>
     * The copy has independent pile and guess containers and shares immutable cards and
     * assumptions. It starts with empty action history, preserves a valid cached state key,
     * and does not mask cards again. PLAN copies share the immutable unseen-identity snapshot;
     * assigning or undoing a card replaces only that board's reference.
     *
     * @return an independent position in the same mode as this instance
     */
    public Solitaire copy() {
        return copy(mode);
    }

    /**
     * Creates a PLAN copy containing only visible or previously observed card identities.
     * <p>
     * Unseen tableau and stock cards become UNKNOWN placeholders; the source is unchanged.
     * On an existing PLAN state, this is equivalent to {@link #copy()} and preserves committed
     * assumptions. Players resolve revealed UNKNOWN slots through {@link #assumeCard(Card, Card)}.
     *
     * @return an independent planning position with empty action history
     */
    public Solitaire planningCopy() {
        return copy(GameMode.PLAN);
    }

    /**
     * Copies board containers and knowledge into the requested mode.
     * <p>
     * Only the transition from GAME to PLAN masks hidden identities and discards the cached
     * state key. Copies within the same mode retain both identities and a valid cached key.
     *
     * @param targetMode the mode for the copy; callers preserve the current mode or request PLAN
     * @return an independent board with no copied action history
     */
    private Solitaire copy(GameMode targetMode) {
        Solitaire clone = new Solitaire(targetMode, rules);
        clone.stockPass = stockPass;
        boolean maskHiddenCards = mode == GameMode.GAME && targetMode == GameMode.PLAN;
        for (int i = 0; i < tableau.size(); i++) {
            List<Card> pile = tableau.get(i);
            List<Card> clonePile = new ArrayList<>(pile);
            if (maskHiddenCards) {
                int hiddenCount = pile.size() - tableauFaceUp.get(i);
                for (int j = 0; j < hiddenCount; j++) {
                    clonePile.set(j, observedCard(pile.get(j)));
                }
            }
            clone.tableau.add(clonePile);
        }
        clone.tableauFaceUp.addAll(tableauFaceUp);
        for (List<Card> pile : foundation) {
            clone.foundation.add(new ArrayList<>(pile));
        }
        if (maskHiddenCards) {
            for (Card card : stockpile) clone.stockpile.add(observedCard(card));
        } else {
            clone.stockpile.addAll(stockpile);
        }
        clone.talon.addAll(talon);
        clone.unknownCardTracker.copyFrom(unknownCardTracker);
        clone.unknownCardGuesses.putAll(unknownCardGuesses);
        if (maskHiddenCards) {
            // Build from the masked board, never from the real hidden identities.
            clone.remainingUnknownCards = clone.unknownCardTracker.getUnknownCards(clone);
        } else {
            clone.remainingUnknownCards = remainingUnknownCards;
            clone.cachedStateKey = cachedStateKey;
            clone.stateKeyValid = stateKeyValid;
        }
        return clone;
    }

    /**
     * Finds the first revealed UNKNOWN slot that still needs an identity.
     * <p>
     * Searches visible tableau cards in pile order, then waste cards from bottom to top.
     * Face-down tableau and stock slots remain unresolved until revealed.
     *
     * @return the UNKNOWN card instance identifying the slot, or {@code null} when no revealed
     *         slot needs an assignment or this instance is in GAME mode
     */
    public Card nextUnresolvedCard() {
        if (!isInPlanMode()) return null;
        for (int t = 0; t < tableau.size(); t++) {
            List<Card> pile = tableau.get(t);
            for (int i = pile.size() - tableauFaceUp.get(t); i < pile.size(); i++) {
                if (pile.get(i).getRank() == Rank.UNKNOWN) return pile.get(i);
            }
        }
        for (Card card : talon) {
            if (card.getRank() == Rank.UNKNOWN) return card;
        }
        return null;
    }

    /**
     * Assigns a concrete unseen identity to one revealed UNKNOWN slot on this planning branch.
     * <p>
     * The identity becomes unavailable to subsequent assignments. The assignment records the
     * original slot, prior guess, and cached key in the same history used for moves and turns,
     * so {@link #undoLastMove()} and {@link #rollbackTo(int)} restore the previous state.
     *
     * @param unknown the exact UNKNOWN instance in a visible tableau or waste slot on this board
     * @param identity a concrete card still present in the branch's unseen pool
     * @return the conditional probability {@code 1.0 / remainingCards} before this assignment,
     *         assuming uniformly distributed identities without replacement
     * @throws IllegalArgumentException if this is not a PLAN board, the slot is not a revealed
     *         UNKNOWN on this board, or the requested identity is not in the unseen pool
     */
    public double assumeCard(Card unknown, Card identity) {
        if (!isInPlanMode() || unknown == null || unknown.getRank() != Rank.UNKNOWN) {
            throw new IllegalArgumentException("Assumptions require an UNKNOWN in a PLAN board");
        }
        String pileCode = null;
        int slot = -1;
        for (int t = 0; t < tableau.size() && slot < 0; t++) {
            List<Card> pile = tableau.get(t);
            for (int i = pile.size() - tableauFaceUp.get(t); i < pile.size(); i++) {
                if (pile.get(i) == unknown) { pileCode = "T" + (t + 1); slot = i; break; }
            }
        }
        for (int i = 0; i < talon.size() && slot < 0; i++) {
            if (talon.get(i) == unknown) { pileCode = "W"; slot = i; }
        }
        if (slot < 0) throw new IllegalArgumentException("UNKNOWN must be a revealed slot on this board");
        return assignCard(pileCode, slot, identity);
    }

    /**
     * Samples one complete hidden-card arrangement on a PLAN board.
     * <p>
     * Assigns every UNKNOWN in tableau, stock and waste uniformly without replacement from
     * the remaining identity pool. Known cards and existing assumptions are preserved, and
     * face-up counts do not change. Assignments use the ordinary action history: save a
     * checkpoint before sampling and roll back to it after evaluating the arrangement.
     * Partial assignments are undone if sampling throws.
     *
     * @param random the generator used to choose identities from the remaining pool
     * @throws IllegalStateException if called on a GAME board
     * @throws NullPointerException if random is null
     */
    public void sampleUnknownCards(RandomGenerator random) {
        if (!isInPlanMode()) throw new IllegalStateException("Sampling requires a PLAN board");
        Objects.requireNonNull(random, "random");
        int checkpoint = getMoveHistorySize();
        try {
            for (int t = 0; t < tableau.size(); t++) sampleUnknownCards("T" + (t + 1), random);
            sampleUnknownCards("S", random);
            sampleUnknownCards("W", random);
        } catch (RuntimeException | Error failure) {
            rollbackTo(checkpoint);
            throw failure;
        }
    }

    /**
     * Assigns the unresolved slots in one pile without changing its shape or visibility.
     *
     * @param pileCode the tableau, stock or waste pile to sample
     * @param random the generator used for identity selection
     */
    private void sampleUnknownCards(String pileCode, RandomGenerator random) {
        List<Card> pile = resolvePile(pileCode);
        for (int slot = 0; slot < pile.size(); slot++) {
            if (pile.get(slot).getRank() != Rank.UNKNOWN) continue;
            List<Card> unseen = getUnknownCards();
            assignCard(pileCode, slot, unseen.get(random.nextInt(unseen.size())));
        }
    }

    /**
     * Records an assignment through the shared undo mechanism after the caller locates a slot.
     *
     * @param pileCode the pile containing the UNKNOWN
     * @param slot the index of that UNKNOWN within the pile
     * @param identity an available concrete identity
     * @return the conditional probability of assigning this identity
     */
    private double assignCard(String pileCode, int slot, Card identity) {
        Card unknown = resolvePile(pileCode).get(slot);
        List<Card> unseen = getUnknownCards();
        int identityIndex = unseen.indexOf(identity);
        if (identityIndex < 0) {
            throw new IllegalArgumentException("Card is already known or assigned: " + identity);
        }
        List<Card> remaining = new ArrayList<>(unseen);
        remaining.remove(identityIndex);
        Assignment assignment = new Assignment(slot, unknown, unknownCardGuesses.get(unknown), unseen);
        moveHistory.add(new Action("assume", pileCode, null, 0, -1, -1, false, 0,
                cachedStateKey, stateKeyValid, assignment));
        remainingUnknownCards = Collections.unmodifiableList(remaining);
        resolvePile(pileCode).set(slot, identity);
        unknownCardGuesses.put(unknown, new UnknownCardGuess(unknown, identity));
        stateKeyValid = false;
        return 1.0 / unseen.size();
    }

    /**
     * Checks whether an observed board agrees with this predicted position.
     * <p>
     * Pile sizes, face-up counts, and known identities must match. An UNKNOWN in the predicted
     * position accepts any identity in the same observed slot, allowing actual play to reveal
     * cards that were unknown when the move was selected. This comparison is directional and
     * does not compare guess maps or action histories.
     * Rules and finite pass counts must also match. Unlimited traversal counts do not
     * affect future legality and are ignored.
     *
     * @param observed the newly observed position; must not be null
     * @return {@code true} if the observed piles match all known constraints in this prediction
     */
    public boolean matchesObservation(Solitaire observed) {
        if (!rules.equals(observed.rules)) return false;
        if (rules.maxStockPasses() != null && stockPass != observed.stockPass) return false;
        if (!tableauFaceUp.equals(observed.tableauFaceUp)) return false;
        for (int i = 0; i < tableau.size(); i++) {
            if (!matchesPile(tableau.get(i), observed.tableau.get(i))) return false;
        }
        for (int i = 0; i < foundation.size(); i++) {
            if (!matchesPile(foundation.get(i), observed.foundation.get(i))) return false;
        }
        return matchesPile(stockpile, observed.stockpile) && matchesPile(talon, observed.talon);
    }

    /**
     * Compares one observed pile against the known identities in a predicted pile.
     *
     * @param expected the predicted cards, with UNKNOWN slots accepting any observed identity
     * @param observed the cards in the corresponding observed pile
     * @return {@code true} if pile lengths and all predicted known identities match
     */
    private static boolean matchesPile(List<Card> expected, List<Card> observed) {
        if (expected.size() != observed.size()) return false;
        for (int i = 0; i < expected.size(); i++) {
            Card card = expected.get(i);
            if (card.getRank() != Rank.UNKNOWN && !card.equals(observed.get(i))) return false;
        }
        return true;
    }

    /**
     * Retains a previously seen hidden card or masks it with a fresh UNKNOWN placeholder.
     *
     * @param card the card in a hidden tableau or stock slot
     * @return the original card if recorded as revealed, otherwise a new UNKNOWN instance
     */
    private Card observedCard(Card card) {
        return unknownCardTracker.isRevealed(card) ? card : new Card(Rank.UNKNOWN, Suit.UNKNOWN);
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
     * Returns the current operational mode of this instance.
     *
     * @return {@code GameMode.GAME} for a real game, {@code GameMode.PLAN} for a lookahead copy
     */
    public GameMode getMode() {
        return mode;
    }

    /**
     * Convenience method to check if this instance is in PLAN mode.
     *
     * @return {@code true} if this is a lookahead copy; {@code false} if this is a real game
     */
    public boolean isInPlanMode() {
        return mode == GameMode.PLAN;
    }

    /**
     * Returns identities that remain unknown or unassigned in the current state.
     * <p>
     * In GAME mode, returns the actual unseen cards in face-down tableau and stock slots.
     * In PLAN mode, returns the retained pool derived from the standard deck minus known cards.
     * Assignments remove identities and undo restores the preceding snapshot. Ordinary moves
     * and stock turns only relocate known cards or UNKNOWN slots, so they do not change this
     * pool. Returned snapshots remain unchanged after subsequent actions or undo.
     *
     * @return an unmodifiable list of remaining identities; never null, possibly empty
     */
    public List<Card> getUnknownCards() {
        if (!isInPlanMode()) return unknownCardTracker.getUnknownCards(this);
        if (remainingUnknownCards == null) {
            remainingUnknownCards = unknownCardTracker.getUnknownCards(this);
        }
        return remainingUnknownCards;
    }

    /**
     * Returns the planning branch's mutable map of UNKNOWN slots to assumptions.
     * <p>
     * Keys use object identity so distinct UNKNOWN positions remain separate. Assignments made
     * through {@link #assumeCard(Card, Card)} contain one concrete identity. Copies have independent
     * maps; undo restores the entry that preceded an assignment.
     *
     * @return a mutable identity map of UNKNOWN card instances to their guesses; never null
     */
    public Map<Card, UnknownCardGuess> getUnknownCardGuesses() {
        return unknownCardGuesses;
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
     * Returns the immutable configuration captured when this game was created.
     *
     * @return the rules shared by this game and its copies
     */
    public GameRules getRules() {
        return rules;
    }

    /**
     * Returns the current stock traversal, including the initial pass.
     *
     * @return the one-based pass number, incremented only when recycling occurs
     */
    public long getStockPass() {
        return stockPass;
    }

    /**
     * Checks whether a stock turn is legal under the remaining pass allowance.
     * <p>
     * Commands, move generation, and simulations share this check. An empty stock
     * requires a nonempty waste and permission to start another pass.
     *
     * @return true if cards can be drawn directly or after a permitted recycle
     */
    public boolean canTurnStock() {
        return !stockpile.isEmpty() || (!talon.isEmpty()
                && (rules.maxStockPasses() == null || stockPass < rules.maxStockPasses()));
    }

    /**
     * Retains the original stock-turn entry point for existing callers.
     * <p>
     * Despite its historical name, this method uses the configured draw count and pass
     * limit. Call {@link #turnStock()} when the caller needs a success result.
     */
    public void turnThree() {
        turnStock();
    }

    /**
     * Draws the configured number of cards, recycling first when required and permitted.
     * <p>
     * A partial final draw takes the remaining stock cards. Recycling preserves card order
     * and starts the next pass in the same action as the draw. Successful turns record one
     * reversible action; rejected turns leave the board, pass count, and history unchanged.
     * The current reveal tracker still records all drawn cards; observation filtering is
     * a separate implementation step.
     *
     * @return true if cards were drawn, false if no stock turn was permitted
     */
    public boolean turnStock() {
        if (!canTurnStock()) return false;
        long previousStateKey = cachedStateKey;
        boolean previousStateKeyValid = stateKeyValid;
        boolean recycled = stockpile.isEmpty();
        stateKeyValid = false;
        if (recycled) {
            stockPass++;
            // Recycle talon back to stockpile: top of talon becomes bottom of new stock.
            for (int i = talon.size() - 1; i >= 0; i--) {
                stockpile.add(talon.get(i));
            }
            talon.clear();
        }

        int movedCount = Math.min(rules.drawCount(), stockpile.size());
        int newlyRevealedMask = 0;
        for (int i = 0; i < movedCount; i++) {
            Card card = stockpile.remove(stockpile.size() - 1);
            talon.add(card);
            // Only newly revealed cards should become unknown again on undo.
            if (!unknownCardTracker.isRevealed(card)) {
                newlyRevealedMask |= 1 << i;
                unknownCardTracker.addRevealedCard(card);
            }
        }
        moveHistory.add(new Action("turn", null, null, movedCount, -1, -1,
                recycled, newlyRevealedMask, previousStateKey, previousStateKeyValid, null));
        return true;
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
        
        if (moving.getRank() == Rank.UNKNOWN) {
            return MoveResult.failure("Resolve the UNKNOWN card before moving it.");
        }
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
        Action action = new Action("move", fromNormalized, toNormalized, movedCount,
                fromType == 'T' ? tableauFaceUp.get(fromIndex) : -1,
                toType == 'T' ? tableauFaceUp.get(toIndex) : -1,
                false, 0, cachedStateKey, stateKeyValid, null);
        stateKeyValid = false;
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
        
        moveHistory.add(action);
        
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
     * Repeated calls reuse the cached value until a move or stock turn changes the board.
     * The next call computes and caches a fresh value. Undo restores the previous cache.
     * <p>
     * <strong>Important:</strong> The state key does not expose hidden card identities directly;
     * it only contributes to a hash. Different hidden card arrangements may produce different keys.
     *
     * @return a 64-bit Zobrist hash of the entire game state
     */
    public long getStateKey() {
        if (!stateKeyValid) {
            cachedStateKey = computeStateKey();
            stateKeyValid = true;
        }
        return cachedStateKey;
    }

    private long computeStateKey() {
        // Encode rules and finite pass state separately from card positions. Unlimited pass
        // counts are omitted so repeated traversals can still be detected as cycles.
        // Default rules contribute zero to preserve the empty-board key convention.
        long ruleCode = (rules.drawCount() == 1 ? 1L : 0L)
                | (rules.stockVisibility() == GameRules.StockVisibility.allDrawn ? 2L : 0L)
                | (rules.maxStockPasses() == null ? 0L : (long) rules.maxStockPasses() << 2)
                | (rules.maxStockPasses() == null ? 0L : stockPass << 5);
        long h = ruleCode * 0x9E3779B97F4A7C15L;
        int idx = 0;
        int pileIndex = 0;

        // Tableau piles (include hidden + visible).
        for (List<Card> pile : tableau) {
            h ^= pileSizeKey(pileIndex, pile.size());
            int faceUpBase = 2048 + pileIndex * 64;
            h ^= STATE_ZOBRIST[faceUpBase + tableauFaceUp.get(pileIndex)] ^ STATE_ZOBRIST[faceUpBase];
            pileIndex++;
            for (Card c : pile) h ^= cardSlotKey(idx++, c);
        }

        // Foundation piles.
        for (List<Card> pile : foundation) {
            h ^= pileSizeKey(pileIndex++, pile.size());
            for (Card c : pile) {
                h ^= cardSlotKey(idx++, c);
            }
        }

        // Pile lengths distinguish moving identical UNKNOWN placeholders between stock and waste.
        h ^= pileSizeKey(pileIndex++, stockpile.size());
        h ^= pileSizeKey(pileIndex, talon.size());
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
     * Computes the hash contribution of a pile's length independently of its card identities.
     * <p>
     * This distinguishes transfers of identical UNKNOWN placeholders between piles. Empty piles
     * contribute zero so an entirely empty board retains its zero state key.
     *
     * @param pileIndex the stable index of the tableau, foundation, stock, or waste pile
     * @param size the number of cards in that pile
     * @return the pile-length contribution to the board hash
     */
    private long pileSizeKey(int pileIndex, int size) {
        int base = 1024 + pileIndex * 64;
        return STATE_ZOBRIST[base + size] ^ STATE_ZOBRIST[base];
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
     * Undoes the last completed card move or stock turn directly from its recorded changes.
     * Restores pile order, face-up counts, revealed-card tracking, assumptions, and the hash cache.
     * No action is recorded by undo itself. Copies can undo only actions performed after copying.
     *
     * @return true if an action was undone, false if the history is empty
     */
    public boolean undoLastMove() {
        if (moveHistory.isEmpty()) {
            return false;
        }
        Action action = moveHistory.remove(moveHistory.size() - 1);
        if ("assume".equals(action.type())) {
            Assignment assignment = action.assignment();
            remainingUnknownCards = assignment.previousUnknownCards();
            resolvePile(action.from()).set(assignment.slot(), assignment.unknown());
            if (assignment.previousGuess() == null) unknownCardGuesses.remove(assignment.unknown());
            else unknownCardGuesses.put(assignment.unknown(), assignment.previousGuess());
        } else if ("turn".equals(action.type())) {
            undoTurn(action);
        } else {
            undoCardMove(action);
        }
        cachedStateKey = action.previousStateKey();
        stateKeyValid = action.previousStateKeyValid();
        return true;
    }

    /**
     * Undoes actions until the history reaches a previously saved depth.
     * Save the depth with {@link #getMoveHistorySize()}. A depth belongs to this board's
     * current history; it does not identify an old branch after that branch has been undone.
     *
     * @throws IllegalArgumentException if historyDepth is negative or beyond the current history
     */
    public void rollbackTo(int historyDepth) {
        if (historyDepth < 0 || historyDepth > moveHistory.size()) {
            throw new IllegalArgumentException("Invalid history depth: " + historyDepth);
        }
        while (moveHistory.size() > historyDepth) {
            undoLastMove();
        }
    }

    /**
     * Reports whether this board has a recorded action available for undo.
     *
     * @return {@code true} if at least one move, turn, or planning assignment can be undone
     */
    public boolean canUndo() {
        return !moveHistory.isEmpty();
    }

    /**
     * Returns the current depth of the reversible action history.
     * <p>
     * Planning assignments count as actions alongside moves and turns. Save this value before
     * simulation and pass it to {@link #rollbackTo(int)} to restore that branch's starting state.
     *
     * @return the number of recorded actions currently available for undo
     */
    public int getMoveHistorySize() {
        return moveHistory.size();
    }

    private void undoCardMove(Action action) {
        List<Card> fromPile = resolvePile(action.from());
        List<Card> toPile = resolvePile(action.to());
        // Restore the moved suffix in its original order without copying either pile.
        if (fromPile != toPile) {
            int start = toPile.size() - action.movedCount();
            for (int i = start; i < toPile.size(); i++) {
                fromPile.add(toPile.get(i));
            }
            toPile.subList(start, toPile.size()).clear();
        }
        if (action.fromFaceUp() >= 0) {
            tableauFaceUp.set(parseIndex(action.from()), action.fromFaceUp());
        }
        if (action.toFaceUp() >= 0) {
            tableauFaceUp.set(parseIndex(action.to()), action.toFaceUp());
        }
    }

    private void undoTurn(Action action) {
        // Reverse the draw first, then reverse the recycle if this turn included one.
        for (int i = action.movedCount() - 1; i >= 0; i--) {
            Card card = talon.remove(talon.size() - 1);
            stockpile.add(card);
            if ((action.newlyRevealedMask() & (1 << i)) != 0) {
                unknownCardTracker.removeRevealedCard(card);
            }
        }
        if (action.recycled()) {
            // Undoing a recycle restores the previous traversal and its remaining allowance.
            stockPass--;
            for (int i = stockpile.size() - 1; i >= 0; i--) {
                talon.add(stockpile.get(i));
            }
            stockpile.clear();
        }
    }

    /**
     * One completed action and its reversal data. Only Solitaire creates history entries,
     * after validation, so every entry has the information needed for exact undo.
     * Face-up counts are -1 for non-tableau piles. For turns, bit i of newlyRevealedMask
     * records whether drawn card i was previously unknown (at most three bits).
     * For assignments, {@code from} identifies the affected pile and {@code assignment}
     * stores its slot, placeholder, prior guess, and previous identity pool; other action types leave it null.
     */
    private record Action(String type, String from, String to, int movedCount,
                          int fromFaceUp, int toFaceUp, boolean recycled, int newlyRevealedMask,
                          long previousStateKey, boolean previousStateKeyValid, Assignment assignment) {}

    /**
     * Stores the information needed to undo one speculative card assignment.
     *
     * @param slot the original zero-based position in the action's source pile
     * @param unknown the UNKNOWN placeholder replaced by the assignment
     * @param previousGuess the prior map entry for this placeholder, or {@code null} if absent
     * @param previousUnknownCards the immutable identity pool before assignment, in its original order
     */
    private record Assignment(int slot, Card unknown, UnknownCardGuess previousGuess,
                              List<Card> previousUnknownCards) {}
}

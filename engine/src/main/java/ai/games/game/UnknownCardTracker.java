package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tracks identities observed through stock draws and newly exposed waste or tableau cards.
 * <p>
 * Observations survive covering and recycling. Search rollback removes observations introduced
 * by the discarded action; training undo explicitly retains real observations. Unknown
 * placeholders are never recorded as identities. Currently visible cards are also known
 * directly from the board, without requiring a tracker entry.
 */
public class UnknownCardTracker {
    /** Standard identities in stable suit/rank order; immutable cards can be shared by boards. */
    private static final List<Card> STANDARD_CARDS = standardCards();

    /** Identities remembered after observation, independently of their current pile. */
    private final Set<Card> revealedCards = new HashSet<>();

    /**
     * Constructs a new tracker with an empty set of revealed cards.
     * <p>
     * Called once per game during {@link Solitaire} initialization.
     */
    public UnknownCardTracker() {
        // Initially, no cards have been revealed.
    }

    /**
     * Records an observed concrete identity.
     * <p>
     * Once recorded, the card remains known when covered or recycled. Null values and
     * UNKNOWN placeholders are ignored so placeholders are never mistaken for observed identities.
     *
     * @param card the newly observed card, or null when an action exposed no concrete identity
     */
    public void addRevealedCard(Card card) {
        if (card != null && card.getRank() != Rank.UNKNOWN) {
            revealedCards.add(card);
        }
    }

    /**
     * Removes knowledge introduced by an action being rolled back.
     *
     * @param card the card first revealed by the discarded action
     */
    void removeRevealedCard(Card card) {
        revealedCards.remove(card);
    }

    /**
     * Adds the source tracker’s revealed identities to this tracker's independent set.
     * <p>
     * Board copying calls this on an empty tracker so undo in either board cannot change
     * what the other board remembers.
     *
     * @param source the tracker whose revealed-card knowledge is copied; must not be null
     */
    void copyFrom(UnknownCardTracker source) {
        revealedCards.addAll(source.revealedCards);
    }

    /**
     * Checks whether a specific card has been revealed during play.
     *
     * @param card the card to check; must not be null
     * @return {@code true} if this identity has been observed;
     *         {@code false} if no observation has been recorded
     */
    public boolean isRevealed(Card card) {
        return revealedCards.contains(card);
    }

    /**
     * Derives possible unseen identities from public knowledge and planning assignments.
     * <p>
     * Covered waste cards are included when unobserved. GAME boards contribute only visible
     * and remembered identities, so this inventory never reveals the true location of a hidden card. PLAN boards retain
     * their committed assumptions as unavailable identities.
     *
     * @param solitaire the real or planning board
     * @return an immutable list of identities not observed or assigned
     */
    public List<Card> getUnknownCards(Solitaire solitaire) {
        return possibleUnseenCards(solitaire);
    }

    /**
     * Derives possible unseen identities from observations and planning assignments.
     * <p>
     * Concrete cards in the piles and remembered revealed cards are excluded from the standard
     * 52-card deck. UNKNOWN placeholders contribute no concrete identity, so the real hidden
     * deal is never consulted.
     *
     * @param solitaire the real or planning board whose remaining identity pool is required
     * @return an unmodifiable list of identities not already known or assigned
     */
    private List<Card> possibleUnseenCards(Solitaire solitaire) {
        Set<Card> known = new HashSet<>(revealedCards);
        solitaire.getFoundation().forEach(known::addAll);
        if (solitaire.isInPlanMode()) {
            solitaire.getTableau().forEach(known::addAll);
            known.addAll(solitaire.getStockpile());
            known.addAll(solitaire.getTalon());
        } else {
            // Visible cards are public even when a test fixture did not populate the tracker.
            solitaire.getVisibleTableau().forEach(known::addAll);
            if (solitaire.getRules().stockVisibility() == GameRules.StockVisibility.allDrawn) {
                known.addAll(solitaire.getTalon());
            } else if (!solitaire.getTalon().isEmpty()) {
                known.add(solitaire.getTalon().getLast());
            }
        }
        List<Card> unseen = new ArrayList<>();
        for (Card card : STANDARD_CARDS) {
            if (!known.contains(card)) unseen.add(card);
        }
        return Collections.unmodifiableList(unseen);
    }

    /**
     * Creates the immutable standard identities once, without allocating a shuffled deck.
     *
     * @return all 52 concrete cards in the enumeration order used for planning pools
     */
    private static List<Card> standardCards() {
        List<Card> cards = new ArrayList<>(52);
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) continue;
            for (Rank rank : Rank.values()) {
                if (rank != Rank.UNKNOWN) {
                    cards.add(new Card(rank, suit));
                }
            }
        }
        return List.copyOf(cards);
    }
}

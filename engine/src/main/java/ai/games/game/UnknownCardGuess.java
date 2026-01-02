package ai.games.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Represents a guess or possible cards for an UNKNOWN card during PLAN mode lookahead.
 * <p>
 * In PLAN mode (lookahead search), when the AI encounters a face-down UNKNOWN card,
 * it may need to make assumptions about what the card could be. This class tracks
 * those possibilities:
 * <ul>
 *   <li><b>Single possibility:</b> We've narrowed it down (e.g., "must be Black 5 Clubs")
 *   <li><b>Two possibilities:</b> Could be either of two cards (e.g., "Black 5 Clubs or Spades")
 * </ul>
 * <p>
 * Once a guess is recorded for an UNKNOWN card instance, it can be used throughout
 * the search tree to validate moves and avoid exploring impossible branches.
 * <p>
 * Guesses are created lazily when needed (during move validation or legal move generation)
 * and persist in the game state across the lookahead tree via deep copying.
 */
public class UnknownCardGuess {
    /** The unknown card instance being tracked (key for lookups). */
    private final Card unknownCard;
    
    /** 
     * List of possible cards this UNKNOWN could be (1-2 cards).
     * Typically two possibilities when the suit is ambiguous (e.g., both Black 5s)
     * or one possibility if we've narrowed it down.
     */
    private final List<Card> possibilities;

    /**
     * Constructs an UnknownCardGuess with a single possibility.
     *
     * @param unknownCard the UNKNOWN card instance being tracked (must have UNKNOWN rank/suit)
     * @param singlePossibility the one card this UNKNOWN could be
     * @throws NullPointerException if either parameter is null
     */
    public UnknownCardGuess(Card unknownCard, Card singlePossibility) {
        this.unknownCard = Objects.requireNonNull(unknownCard, "unknownCard");
        this.possibilities = new ArrayList<>();
        this.possibilities.add(Objects.requireNonNull(singlePossibility, "singlePossibility"));
    }

    /**
     * Constructs an UnknownCardGuess with multiple possibilities (typically 2).
     *
     * @param unknownCard the UNKNOWN card instance being tracked (must have UNKNOWN rank/suit)
     * @param possibilityList the list of cards this UNKNOWN could be (1-2 cards)
     * @throws NullPointerException if either parameter is null
     * @throws IllegalArgumentException if possibilityList is empty
     */
    public UnknownCardGuess(Card unknownCard, List<Card> possibilityList) {
        this.unknownCard = Objects.requireNonNull(unknownCard, "unknownCard");
        if (possibilityList == null || possibilityList.isEmpty()) {
            throw new IllegalArgumentException("possibilityList must not be null or empty");
        }
        this.possibilities = new ArrayList<>(possibilityList);
    }

    /**
     * Returns the UNKNOWN card instance this guess is for.
     *
     * @return the unknown card
     */
    public Card getUnknownCard() {
        return unknownCard;
    }

    /**
     * Returns an unmodifiable list of possible cards this UNKNOWN could be.
     *
     * @return an unmodifiable list of possibilities (1-2 cards typically)
     */
    public List<Card> getPossibilities() {
        return Collections.unmodifiableList(possibilities);
    }

    /**
     * Checks if a specific card is a possibility for this UNKNOWN.
     *
     * @param card the card to check
     * @return {@code true} if this card is in the possibilities list; {@code false} otherwise
     */
    public boolean canBe(Card card) {
        return possibilities.contains(card);
    }

    /**
     * Returns the number of possibilities.
     *
     * @return 1 if definitely determined, 2 if ambiguous, etc.
     */
    public int getPossibilityCount() {
        return possibilities.size();
    }

    /**
     * Checks if this guess has a single definite possibility (no ambiguity).
     *
     * @return {@code true} if exactly one possibility; {@code false} otherwise
     */
    public boolean isDefinite() {
        return possibilities.size() == 1;
    }

    /**
     * Checks if this guess has exactly two ambiguous possibilities.
     *
     * @return {@code true} if exactly two possibilities; {@code false} otherwise
     */
    public boolean isAmbiguous() {
        return possibilities.size() == 2;
    }

    /**
     * Returns a string representation for logging and debugging.
     *
     * @return a string like "UnknownCardGuess(? -> [5♣, 5♠])"
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("UnknownCardGuess(");
        sb.append(unknownCard.shortName()).append(" -> [");
        for (int i = 0; i < possibilities.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(possibilities.get(i).shortName());
        }
        sb.append("])");
        return sb.toString();
    }

    /**
     * Checks equality based on the unknown card instance and possibilities list.
     *
     * @param o the object to compare
     * @return {@code true} if both have the same unknown card and same possibilities
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof UnknownCardGuess)) return false;
        UnknownCardGuess that = (UnknownCardGuess) o;
        return unknownCard.equals(that.unknownCard) && possibilities.equals(that.possibilities);
    }

    /**
     * Returns a hash code based on unknown card and possibilities.
     *
     * @return the hash code
     */
    @Override
    public int hashCode() {
        return Objects.hash(unknownCard, possibilities);
    }
}

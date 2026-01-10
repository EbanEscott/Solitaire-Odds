package ai.games.unit.helpers;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Reusable scenario factory for tests.
 *
 * <p><strong>Why this exists</strong>
 * <ul>
 *   <li><strong>Consistency:</strong> many tests want the same handful of canonical board positions.
 *       Centralising them prevents drift and makes failures easier to reason about.</li>
 *   <li><strong>Readability:</strong> scenario names communicate intent better than repeating long
 *       builder calls throughout the suite.</li>
 *   <li><strong>Strictness:</strong> scenarios are built via {@link SolitaireBuilder}, so they are
 *       always complete (52 cards) and validated.</li>
 * </ul>
 *
 * <p><strong>Design note</strong>
 * <p>This factory intentionally contains only "common" scenarios. Tests that require unusual or
 * highly specific layouts should use {@link SolitaireBuilder} directly.
 */
public final class SolitaireFactory {

    private SolitaireFactory() {
    }

    /**
     * A strict but minimal state: all 52 cards are in the stockpile.
     *
     * <p><strong>Why:</strong> useful as a baseline for tests that only care about stock/talon
     * transitions (e.g., {@link ai.games.game.Solitaire#turnThree()}) without involving tableau.
     */
    public static Solitaire stockOnly() {
        return SolitaireBuilder.newGame().build();
    }

    /**
     * Endgame state that is exactly one legal move away from winning.
     *
     * <p>Setup:
     * <ul>
     *   <li>F1 = A♣..K♣</li>
     *   <li>F2 = A♦..K♦</li>
     *   <li>F3 = A♠..K♠</li>
     *   <li>F4 = A♥..Q♥</li>
     *   <li>T1 = K♥ (face-up)</li>
          *   <li>Stock and waste (talon) empty</li>
     * </ul>
     *
     * <p><strong>Why:</strong> ideal for deterministic "win detection" and AI regression tests.
     */
    public static Solitaire oneMoveFromWin() {
        return SolitaireBuilder
                .newGame()
                .tableau("T1", "K♥")
                .foundation("F1",
                        "A♣", "2♣", "3♣", "4♣", "5♣", "6♣", "7♣", "8♣", "9♣", "10♣", "J♣", "Q♣", "K♣")
                .foundation("F2",
                        "A♦", "2♦", "3♦", "4♦", "5♦", "6♦", "7♦", "8♦", "9♦", "10♦", "J♦", "Q♦", "K♦")
                .foundation("F3",
                        "A♠", "2♠", "3♠", "4♠", "5♠", "6♠", "7♠", "8♠", "9♠", "10♠", "J♠", "Q♠", "K♠")
                .foundation("F4",
                        "A♥", "2♥", "3♥", "4♥", "5♥", "6♥", "7♥", "8♥", "9♥", "10♥", "J♥", "Q♥")
                .stock()
                .waste()
                .build();
    }

        /**
         * Endgame state that is two legal moves away from winning (two kings are still on the tableau).
         *
         * <p><strong>Why:</strong> used by tests that require at least two distinct non-quit moves
         * (e.g., drift-detection tests that intentionally apply a different legal move).
         */
        public static Solitaire twoMovesFromWin() {
                return SolitaireBuilder
                                .newGame()
                                .tableau("T1", "K♠")
                                .tableau("T2", "K♥")
                                .foundation("F1", foundationUpTo(Suit.CLUBS, Rank.KING))
                                .foundation("F2", foundationUpTo(Suit.DIAMONDS, Rank.KING))
                                .foundation("F3", foundationUpTo(Suit.SPADES, Rank.QUEEN))
                                .foundation("F4", foundationUpTo(Suit.HEARTS, Rank.QUEEN))
                                .stock()
                                .waste()
                                .build();
        }

        /**
         * Completely won board: all 52 cards are on the foundations.
         *
         * <p><strong>Why:</strong> used as a stable starting point for applying reverse moves when
         * reconstructing endgame training states.
         *
         * <p>Foundation suit order matches {@link Suit#values()} (excluding {@link Suit#UNKNOWN}) to
         * preserve existing reverse-move expectations (e.g. "move F1 K♣ ...").
         */
        public static Solitaire wonGame() {
                SolitaireBuilder b = SolitaireBuilder.newGame().stock().waste();

                int foundationIndex = 1;
                for (Suit suit : Suit.values()) {
                        if (suit == Suit.UNKNOWN) {
                                continue;
                        }
                        List<String> pile = new java.util.ArrayList<>();
                        for (Rank rank : Rank.values()) {
                                if (rank == Rank.UNKNOWN) {
                                        continue;
                                }
                                pile.add(new Card(rank, suit).shortName());
                        }
                        b.foundation("F" + foundationIndex, pile.toArray(String[]::new));
                        foundationIndex++;
                }

                return b.build();
        }

        /**
         * Returns a legal foundation sequence for the given suit from Ace up to {@code maxRank}.
         *
         * <p>Useful for test setup where you want, e.g., spades in foundation up to Q♠.
         */
        public static String[] foundationUpTo(Suit suit, Rank maxRank) {
                if (suit == Suit.UNKNOWN || maxRank == Rank.UNKNOWN) {
                        throw new IllegalArgumentException("Unknown suit/rank not allowed in GAME mode foundations");
                }

                java.util.List<String> cards = new java.util.ArrayList<>();
                for (Rank rank : Rank.values()) {
                        if (rank == Rank.UNKNOWN) {
                                continue;
                        }
                        cards.add(new Card(rank, suit).shortName());
                        if (rank == maxRank) {
                                break;
                        }
                }

                return cards.toArray(String[]::new);
        }

    /**
     * Common tableau-flip scenario.
     *
     * <p>Setup:
     * <ul>
     *   <li>T1 = K♦ (face-down), 5♠ (face-up)</li>
     *   <li>T2 = 6♥ (face-up)</li>
     *   <li>T3..T7 empty</li>
          *   <li>Waste (talon) empty</li>
          *   <li>Stock contains the remaining cards (auto-filled by {@link SolitaireBuilder})</li>
     * </ul>
     *
     * <p><strong>Why:</strong> used to test that moving the last face-up card flips the next
     * face-down card as per Klondike rules.
     */
    public static Solitaire flipAfterMovingLastFaceUpCard() {
        return SolitaireBuilder
                .newGame()
                .tableau("T1", 1, "K♦", "5♠")
                .tableau("T2", "6♥")
                .stock()
                .waste()
                .build();
    }

    /**
     * Creates a complete 52-card state while keeping stock and waste (talon) exactly as specified.
     *
     * <p><strong>Why:</strong> {@link SolitaireBuilder} auto-fills unspecified cards into stock.
     * For tests that assert stock/waste sizes and ordering (e.g. {@link Solitaire#turnThree()}),
     * we need a way to keep those piles precise while still satisfying the "full 52 unique cards"
     * invariant.
     *
     * <p>Implementation detail: all remaining cards are placed into tableau pile T7. If any cards
     * are placed there, T7 is given a face-up count of 1.
     *
     * @param stockBottomToTop bottom-to-top stock ordering; last element is drawn first
     * @param wasteBottomToTop bottom-to-top waste (talon) ordering; last element is the playable top card
     */
    public static Solitaire withExactStockAndWaste(String[] stockBottomToTop, String[] wasteBottomToTop) {
        Set<Card> specified = new HashSet<>();
        for (String c : stockBottomToTop) {
            specified.add(SolitaireBuilder.parseCard(c));
        }
        for (String c : wasteBottomToTop) {
            specified.add(SolitaireBuilder.parseCard(c));
        }

        List<Card> fullDeck = SolitaireBuilder.fullDeckInStableOrder();
        String[] remainder = fullDeck.stream()
                .filter(c -> !specified.contains(c))
                .map(Card::shortName)
                .toArray(String[]::new);

        // Dump remaining cards into T7 so stock/waste remain exact.
        return SolitaireBuilder
                .newGame()
                .stock(stockBottomToTop)
                .waste(wasteBottomToTop)
                .tableau("T7", remainder.length == 0 ? 0 : 1, remainder)
                .build();
    }
}

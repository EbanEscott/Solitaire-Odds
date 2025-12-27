package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireTestHelper;
import ai.games.unit.helpers.TestGameStateBuilder;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Legal move coverage with deterministic board states.
 *
 * <p>This test class validates all legal move scenarios across different move types and game states.
 * Note: Tests use partial game states (empty columns, minimal cards) and do NOT validate
 * assertFullDeckState, as that is only needed for complex game states.
 *
 * <p><b>Tests and their intentions:</b>
 * <ul>
 *   <li><b>aceMovesToEmptyFoundation</b> - Tableau Ace to empty foundation (baseline rule)</li>
 *   <li><b>twoOnAceFoundationShouldBeLegal</b> - Foundation progression: rank increment with same suit</li>
 *   <li><b>tableauAlternatingDescendingIsLegal</b> - Tableau → Tableau: alternating colors, descending rank</li>
 *   <li><b>topVisibleTableauCardMovesLegally</b> - Only top visible card moves in single-card move (null cardCode)</li>
 *   <li><b>bottomVisibleTableauCardDrivesPileMoveRule</b> - Multi-card stack: bottom card validates move legality</li>
 *   <li><b>canMoveVisibleStackWhenBottomCardFitsDestination</b> - Multi-card stack moves when bottom card is valid</li>
 *   <li><b>talonToFoundationIncrementIsLegal</b> - Waste (talon) → Foundation legal progression</li>
 *   <li><b>wasteToTableauIsLegal</b> - Waste → Tableau single card move</li>
 *   <li><b>kingLedStackToEmpty</b> - King-led multi-card stack → empty tableau column</li>
 *   <li><b>multiCardStackFoundationBlocked</b> (Phase 2) - Valid internal sequence moves onto foundation destination</li>
 *   <li><b>wasteToTableauMultipleChoices</b> (Phase 2) - Same waste card has multiple valid tableau destinations</li>
 * </ul>
 */
class LegalMovesTest {

    @Test
    void aceMovesToEmptyFoundation() {
        // T1 has A♣ face up; empty foundation accepts it.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void twoOnAceFoundationShouldBeLegal() {
        // Reproduces the screenshot case: F1 has A♥, T6 has 2♥ face up; should be legal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(),
                pile(new Card(Rank.TWO, Suit.HEARTS)), empty()), Arrays.asList(0, 0, 0, 0, 0, 1, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.HEARTS)), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T6", null, "F1"));
    }

    @Test
    void tableauAlternatingDescendingIsLegal() {
        // T1 has 7♠, T2 has 8♥; moving 7♠ onto 8♥ is legal (alternating color, descending).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.HEARTS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void topVisibleTableauCardMovesLegally() {
        // T3 has two face-up cards: 9♠ below (covered), Q♣ on top (nearest/active). T7 has K♥. Only the top/active Q♣ can move.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        empty(),
                        pile(new Card(Rank.NINE, Suit.SPADES), new Card(Rank.QUEEN, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.KING, Suit.HEARTS))
                ),
                Arrays.asList(0, 0, 2, 0, 0, 0, 1));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T3", null, "T7"));
    }

    @Test
    void bottomVisibleTableauCardDrivesPileMoveRule() {
        // T3 has J♦, Q♣ (both face-up); when moving the entire visible stack, J♦ (bottom) on K♥ is illegal (same color).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        empty(),
                        pile(new Card(Rank.JACK, Suit.DIAMONDS), new Card(Rank.QUEEN, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.KING, Suit.HEARTS))
                ),
                Arrays.asList(0, 0, 2, 0, 0, 0, 1));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        // When moving the entire visible stack from J♦, the move is illegal because J♦ (red) cannot go on K♥ (red, same color).
        assertFalse(solitaire.moveCard("T3", "J♦", "T7"));
    }

    @Test
    void canMoveVisibleStackWhenBottomCardFitsDestination() {
        // Source T6 has 8♥, 7♣, 6♥ face up (bottom is 8♥). Destination T2 has 9♣ face up; moving the visible stack is legal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        pile(new Card(Rank.NINE, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.EIGHT, Suit.HEARTS), new Card(Rank.SEVEN, Suit.CLUBS), new Card(Rank.SIX, Suit.HEARTS)),
                        empty()
                ),
                Arrays.asList(0, 1, 0, 0, 0, 3, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.DIAMONDS)), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T6", "8♥", "T2"));
        // Destination should now have moved stack on top.
        assertEquals(4, solitaire.getTableau().get(1).size());
    }

    @Test
    void talonToFoundationIncrementIsLegal() {
        // F2 has A♦; talon top is 2♦; moving waste to foundation should succeed.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), pile(new Card(Rank.ACE, Suit.DIAMONDS)), empty(), empty()));
        SolitaireTestHelper.setTalon(solitaire, pile(new Card(Rank.TWO, Suit.DIAMONDS)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        assertTrue(solitaire.moveCard("W", null, "F2"));
    }

    @Test
    void wasteToTableauIsLegal() {
        // T1 has 6♣ (black); waste has 5♥ (red) on top (valid move: 5 onto 6, different colors, descending).
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.SIX, Suit.CLUBS));
        for (int i = 1; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        SolitaireTestHelper.setTalon(solitaire, pile(new Card(Rank.FIVE, Suit.HEARTS)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        assertTrue(solitaire.moveCard("W", null, "T1"), "5♥ should move onto 6♣ (different colors)");
    }

    @Test
    void kingLedStackToEmpty() {
        // T1 has K♠, Q♣, J♠ (valid King-led stack); T2 is empty (target).
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0,
                new Card(Rank.KING, Suit.SPADES),
                new Card(Rank.QUEEN, Suit.CLUBS),
                new Card(Rank.JACK, Suit.SPADES));
        TestGameStateBuilder.seedTableauStack(solitaire, 1);  // Empty
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        assertTrue(solitaire.moveCard("T1", "K♠", "T2"), "King-led stack should move to empty column");
    }

    @Test
    void multiCardStackFoundationBlocked() {
        // T1 has 5♠, 4♥, 3♣ (valid internal sequence); T2 has 6♥; move succeeds (5♠ black on 6♥ red, descending).
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0,
                new Card(Rank.FIVE, Suit.SPADES),
                new Card(Rank.FOUR, Suit.HEARTS),
                new Card(Rank.THREE, Suit.CLUBS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.SIX, Suit.HEARTS));
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // Moving 5♠ (black) onto 6♥ (red): valid alternating colors and descending rank
        assertTrue(solitaire.moveCard("T1", "5♠", "T2"), "5♠ should move onto 6♥ (valid alternating/descending)");
    }

    @Test
    void wasteToTableauMultipleChoices() {
        // Waste has 7♠; T1 has 8♥, T2 has 8♣ (two valid destinations for same card).
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.EIGHT, Suit.HEARTS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.EIGHT, Suit.CLUBS));
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        SolitaireTestHelper.setTalon(solitaire, pile(new Card(Rank.SEVEN, Suit.SPADES)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        // 7♠ can move to either 8♥ or 8♣; move to T1
        assertTrue(solitaire.moveCard("W", null, "T1"), "7♠ should move onto 8♥ (red on black)");
    }

    private static List<Card> empty() {
        return SolitaireTestHelper.emptyPile();
    }

    private static List<Card> pile(Card... cards) {
        return SolitaireTestHelper.pile(cards);
    }

    private static void seedTableau(Solitaire solitaire, List<List<Card>> piles, List<Integer> faceUpCounts) {
        SolitaireTestHelper.setTableau(solitaire, piles, faceUpCounts);
    }

    private static void seedFoundation(Solitaire solitaire, List<List<Card>> piles) {
        SolitaireTestHelper.setFoundation(solitaire, piles);
    }
}

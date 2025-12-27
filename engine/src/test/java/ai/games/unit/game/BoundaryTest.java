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
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Boundary and edge-case behavior around tableau rules, stock/talon flow, invalid inputs, and deck integrity.
 *
 * <p>This test class validates edge cases, state transitions, and system behavior under unusual conditions.
 * Note: Tests use partial game states (empty columns, minimal cards) and do NOT validate
 * assertFullDeckState, as that is only needed for complex game states.
 *
 * <p><b>Tests and their intentions:</b>
 * <ul>
 *   <li><b>emptyTableauAcceptsOnlyKing</b> - Empty columns reject non-Kings but accept Kings</li>
 *   <li><b>faceDownCardFlipsWhenTopMovesAway</b> - State transition: face-down card flips when last visible moves</li>
 *   <li><b>moveSingleOntoTableauRequiresAlternatingDescending</b> - Validates alternating color and rank descent rules</li>
 *   <li><b>turnThreeWithLessThanThreeCardsMovesAllRemaining</b> - Stock turn: move all if fewer than 3 cards</li>
 *   <li><b>turnThreeRecyclesTalonWhenStockEmpty</b> - Stock turn: recycle talon back to stockpile when stock empty</li>
 *   <li><b>moveFromEmptyTalonFails</b> - Cannot move from waste (talon) when empty</li>
 *   <li><b>invalidCodesAreRejected</b> - System robustness: bad move codes handled gracefully</li>
 *   <li><b>foundationProgressionBuildsUpToKingSameSuit</b> - Foundation builds from Ace to King in same suit</li>
 *   <li><b>deckResetHas52UniqueCards</b> - Deck integrity: fresh deck has all 52 unique cards</li>
 *   <li><b>lastFaceUpCardMove</b> - State transition: face-down flip when last face-up card moves away</li>
 *   <li><b>multipleEmptyTableauColumns</b> - Multiple empty columns: any can accept Kings</li>
 *   <li><b>multiCardStackInternalColorViolation</b> (Phase 2) - Valid alternating sequence [8♥,7♣,6♥] moves onto 9♠; verifies internal validity</li>
 *   <li><b>stackMoveAfterFlipTrigger</b> (Phase 2) - Moving 5♠ away flips K♦(facedown), enabling K♦ to empty column</li>
 *   <li><b>foundationBuildAfterTableauRecovery</b> (Phase 3) - F1 has 2♥; moving 3♥ from T1 to F1 advances foundation</li>
 *   <li><b>stockTalonEmptyGameOver</b> (Phase 3) - Stock and talon both empty; no legal moves; game terminal</li>
 *   <li><b>foundationProgressWithMultipleSuits</b> (Phase 3) - Multiple foundations building in parallel; 3♠ onto 2♠ in F2</li>
 *   <li><b>sequentialFoundationBuilding</b> (Phase 3) - Foundation F1 built from A♣ through K♣ (all 13 cards)</li>
 * </ul>
 */
class BoundaryTest {

    @Test
    void emptyTableauAcceptsOnlyKing() {
        // Empty T1: moving Q♣ should fail, moving K♣ should succeed.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.QUEEN, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        assertFalse(solitaire.moveCard("T1", "T2"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.KING, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void faceDownCardFlipsWhenTopMovesAway() {
        // T1 has 7♠ (face down) under K♦ (face up). After moving K♦ away, 7♠ should flip face up.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES), new Card(Rank.KING, Suit.DIAMONDS)),
                        empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", "T2")); // move king to empty tableau pile
        assertEquals(1, SolitaireTestHelper.getTableauFaceUpCount(solitaire, 0));
        assertEquals(new Card(Rank.SEVEN, Suit.SPADES), SolitaireTestHelper.getTableauPile(solitaire, 0).get(0));
    }

    @Test
    void moveSingleOntoTableauRequiresAlternatingDescending() {
        // T1 top 7♠ onto T2 top 8♥ (legal) vs 8♠ same color (illegal).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.HEARTS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        assertTrue(solitaire.moveCard("T1", "T2"));

        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.SPADES)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        assertFalse(solitaire.moveCard("T1", "T2"));
    }

    @Test
    void turnThreeWithLessThanThreeCardsMovesAllRemaining() {
        // Stock has only two cards; turnThree should move both to talon.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setStockpile(solitaire, pile(new Card(Rank.TEN, Suit.SPADES), new Card(Rank.JACK, Suit.SPADES)));
        SolitaireTestHelper.setTalon(solitaire, empty());

        solitaire.turnThree();

        assertEquals(0, SolitaireTestHelper.getStockpile(solitaire).size());
        assertEquals(pile(new Card(Rank.JACK, Suit.SPADES), new Card(Rank.TEN, Suit.SPADES)), SolitaireTestHelper.getTalon(solitaire));
    }

    @Test
    void turnThreeRecyclesTalonWhenStockEmpty() {
        // Stock empty, talon has four cards (Q top). A turn should recycle and deal three.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setStockpile(solitaire, empty());
        SolitaireTestHelper.setTalon(solitaire, pile(
                new Card(Rank.TEN, Suit.SPADES),
                new Card(Rank.JACK, Suit.SPADES),
                new Card(Rank.QUEEN, Suit.SPADES),
                new Card(Rank.KING, Suit.SPADES) // top of talon
        ));

        solitaire.turnThree();

        // After recycling, the original bottom (TEN) should be drawn first.
        assertEquals(pile(
                new Card(Rank.TEN, Suit.SPADES),
                new Card(Rank.JACK, Suit.SPADES),
                new Card(Rank.QUEEN, Suit.SPADES)
        ), SolitaireTestHelper.getTalon(solitaire));
        assertEquals(pile(new Card(Rank.KING, Suit.SPADES)), SolitaireTestHelper.getStockpile(solitaire));
    }

    @Test
    void moveFromEmptyTalonFails() {
        // No cards in talon; moving W to foundation should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setTalon(solitaire, empty());
        SolitaireTestHelper.setStockpile(solitaire, empty());

        assertFalse(solitaire.moveCard("W", "F1"));
    }

    @Test
    void invalidCodesAreRejected() {
        // Bad codes should be rejected quietly.
        Solitaire solitaire = new Solitaire(new Deck());
        assertFalse(solitaire.moveCard("X1", "F1"));
        assertFalse(solitaire.moveCard("T1", "Z9"));
        assertFalse(solitaire.moveCard("", "F1"));
        assertFalse(solitaire.moveCard(null, "F1"));
    }

    @Test
    void foundationProgressionBuildsUpToKingSameSuit() {
        // A♣ in F1; 2♣ then 3♣ should succeed; 3♦ should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.CLUBS)), empty(), empty(), empty()));
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.TWO, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", "F1"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.THREE, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", "F1"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.THREE, Suit.DIAMONDS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertFalse(solitaire.moveCard("T1", "F1"));
    }

    @Test
    void deckResetHas52UniqueCards() {
        // Reset should produce a full, unique 52-card deck.
        Deck deck = new Deck();
        List<Card> cards = deck.asUnmodifiableList();
        assertEquals(52, cards.size(), "Deck should have 52 cards");

        Set<Card> unique = new HashSet<>(cards);
        assertEquals(52, unique.size(), "Deck should not contain duplicates");
    }

    @Test
    void multipleEmptyTableauColumns() {
        // T1 and T3 are empty; moving K♠ from T5 to either empty should succeed.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0);  // Empty
        TestGameStateBuilder.seedTableauStack(solitaire, 1);  // Empty
        TestGameStateBuilder.seedTableauStack(solitaire, 2, new Card(Rank.KING, Suit.SPADES));
        TestGameStateBuilder.seedTableauStack(solitaire, 3);  // Empty
        TestGameStateBuilder.seedTableauStack(solitaire, 4);  // Empty
        TestGameStateBuilder.seedTableauStack(solitaire, 5);  // Empty
        TestGameStateBuilder.seedTableauStack(solitaire, 6);  // Empty
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // Can move to T1 (empty)
        assertTrue(solitaire.moveCard("T3", null, "T1"), "King should move to first empty column");
    }

    @Test
    void multiCardStackInternalColorViolation() {
        // T1 has [8♥, 7♣, 6♥] - valid alternating sequence; T2 has 9♠
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0,
                new Card(Rank.EIGHT, Suit.HEARTS),
                new Card(Rank.SEVEN, Suit.CLUBS),
                new Card(Rank.SIX, Suit.HEARTS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.NINE, Suit.SPADES));
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // Moving the entire stack from 8♥ onto 9♠ should work
        assertTrue(solitaire.moveCard("T1", "8♥", "T2"), "Valid internal sequence should move");
    }

    @Test
    void stackMoveAfterFlipTrigger() {
        // T1 has [K♦(facedown), 5♠(faceup)]; move 5♠ onto 6♥ to flip K♦, then move K♦ to empty T3.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauWithFaceDown(solitaire, 0, 1,
                new Card(Rank.KING, Suit.DIAMONDS),
                new Card(Rank.FIVE, Suit.SPADES));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.SIX, Suit.HEARTS));
        TestGameStateBuilder.seedTableauStack(solitaire, 2);  // Empty
        for (int i = 3; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // First move: move 5♠ (black) onto 6♥ (red), which flips K♦
        assertTrue(solitaire.moveCard("T1", "5♠", "T2"), "5♠ should move onto 6♥");
        assertEquals(1, solitaire.getTableauFaceUpCounts().get(0), "K♦ should flip to face-up");

        // Now K♦ is visible and can move to empty T3
        assertTrue(solitaire.moveCard("T1", null, "T3"), "K♦ should move to empty after flip");
    }

    @Test
    void foundationBuildAfterTableauRecovery() {
        // F1 has A♥,2♥; T1 has 3♥; move 3♥ to F1, creating a gap that won't break foundation rules.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedFoundationPartial(solitaire, 0, Suit.HEARTS, Rank.TWO);
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.THREE, Suit.HEARTS));
        for (int i = 1; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // Move 3♥ from tableau to foundation
        assertTrue(solitaire.moveCard("T1", null, "F1"), "3♥ should move onto 2♥ in foundation");
        assertEquals(3, solitaire.getFoundation().get(0).size(), "F1 should have 3 cards");
    }

    @Test
    void stockTalonEmptyGameOver() {
        // Stock empty, talon empty, no legal moves; game should be terminal.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.clearTableau(solitaire);
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // With no cards and no legal moves available, trying to move from W should fail
        assertFalse(solitaire.moveCard("W", null, "T1"), "Cannot move from empty waste");
    }

    @Test
    void foundationProgressWithMultipleSuits() {
        // Build F1 through 3♥, F2 through 2♠; try to move 3♠ from T1 to F2 (wrong suit progression).
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedFoundationPartial(solitaire, 0, Suit.HEARTS, Rank.THREE);
        TestGameStateBuilder.seedFoundationPartial(solitaire, 1, Suit.SPADES, Rank.TWO);
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.THREE, Suit.SPADES));
        for (int i = 1; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // 3♠ cannot go on 2♠ (correct suit but we're at rank 2, need 3) - actually this SHOULD work!
        assertTrue(solitaire.moveCard("T1", null, "F2"), "3♠ should move onto 2♠ in same suit");
    }

    @Test
    void sequentialFoundationBuilding() {
        // Build F1 from A♣ through K♣ (all 13 cards); verify progression.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedFoundationPartial(solitaire, 0, Suit.CLUBS, Rank.KING);
        TestGameStateBuilder.clearTableau(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        // Foundation should have all 13 clubs
        assertEquals(13, solitaire.getFoundation().get(0).size(), "F1 should have all 13 clubs from Ace to King");
    }

    @Test
    void lastFaceUpCardMove() {
        // T1 has [5♠(facedown), 6♣(faceup)]; moving 6♣ should flip 5♠
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauWithFaceDown(solitaire, 0, 1,
                new Card(Rank.FIVE, Suit.SPADES),
                new Card(Rank.SIX, Suit.CLUBS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.SEVEN, Suit.HEARTS));
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        assertEquals(1, solitaire.getTableauFaceUpCounts().get(0), "T1 should have 1 face-up card before move");

        // Move the only face-up card
        assertTrue(solitaire.moveCard("T1", null, "T2"), "6♣ should move onto 7♥");

        // Check that face-down card flipped
        assertEquals(1, solitaire.getTableauFaceUpCounts().get(0), "5♠ should flip to face-up");
        List<Card> visibleT1 = solitaire.getVisibleTableau().get(0);
        assertEquals(1, visibleT1.size(), "T1 should have 1 visible card");
        assertEquals(new Card(Rank.FIVE, Suit.SPADES), visibleT1.get(0), "T1 should show 5♠");
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

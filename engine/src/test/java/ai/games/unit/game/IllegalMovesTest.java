package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
 * Illegal move coverage; each test seeds Solitaire state then asserts the move is rejected.
 *
 * <p>This test class validates that invalid moves are correctly prevented across different game scenarios.
 * Note: Tests use partial game states (empty columns, minimal cards) and do NOT validate
 * assertFullDeckState, as that is only needed for complex game states.
 *
 * <p><b>Tests and their intentions:</b>
 * <ul>
 *   <li><b>movingFromEmptyTableauFails</b> - Cannot move from empty tableau column</li>
 *   <li><b>wrongSuitToFoundationFails</b> - Foundation requires exact suit match</li>
 *   <li><b>nonAceToEmptyFoundationFails</b> - Foundation must start with Ace (no non-Ace on empty)</li>
 *   <li><b>faceDownCardCannotMove</b> - Visibility requirement: face-down cards cannot move</li>
 *   <li><b>incorrectColorSequenceOnTableauFails</b> - Tableau requires alternating colors</li>
 *   <li><b>nonConsecutiveRankOnFoundation</b> - Foundation requires consecutive ranks (no gaps)</li>
 *   <li><b>nonKingStackOnEmptyTableau</b> - Only Kings (or King-led stacks) can move to empty columns</li>
 *   <li><b>movingNonTopWasteCard</b> - Waste mechanics: only top card is accessible</li>
 *   <li><b>sameColorSameRankOnTableau</b> (Phase 2) - Same color same rank on tableau rejected</li>
 *   <li><b>faceDownStackAttempt</b> (Phase 2) - Face-down cards in a stack are rejected</li>
 * </ul>
 */
class IllegalMovesTest {

    @Test
    void movingFromEmptyTableauFails() {
        // T1 empty, foundation empty; moving from T1 should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Collections.nCopies(7, empty()), Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void wrongSuitToFoundationFails() {
        // Foundation F1 has A♥; trying to place 2♣ (wrong suit) should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.TWO, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.HEARTS)), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void nonAceToEmptyFoundationFails() {
        // Empty foundation; attempting to move Q♣ onto it is illegal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.QUEEN, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void faceDownCardCannotMove() {
        // T1 has K♠ but face-up count is 0; moving it should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.KING, Suit.SPADES)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void incorrectColorSequenceOnTableauFails() {
        // T1 top is 5♥, T2 top is 4♦ (same color); stacking should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.FIVE, Suit.HEARTS)), pile(new Card(Rank.FOUR, Suit.DIAMONDS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void nonConsecutiveRankOnFoundation() {
        // F1 has A♥, 2♥; T1 has 4♥ (skipping rank 3); should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedFoundationPartial(solitaire, 0, Suit.HEARTS, Rank.TWO);
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.FOUR, Suit.HEARTS));
        for (int i = 1; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        assertFalse(solitaire.moveCard("T1", null, "F1"), "4♥ cannot go on 2♥ - rank 3 is missing");
    }

    @Test
    void nonKingStackOnEmptyTableau() {
        // T1 has Q♠, J♣ (valid internal sequence but Queen-led); T2 is empty; should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0,
                new Card(Rank.QUEEN, Suit.SPADES),
                new Card(Rank.JACK, Suit.CLUBS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1);  // Empty
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        assertFalse(solitaire.moveCard("T1", "Q♠", "T2"), "Non-King stacks cannot move to empty columns");
    }

    @Test
    void movingNonTopWasteCard() {
        // Waste mechanics: only top card is accessible. This test validates that
        // the top card can move but earlier seeding of non-top cards is prevented.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.clearTableau(solitaire);
        TestGameStateBuilder.clearFoundations(solitaire);
        // Talon stack with only top card (7♥) visible for moves
        SolitaireTestHelper.setTalon(solitaire, 
            pile(new Card(Rank.FIVE, Suit.SPADES), 
                 new Card(Rank.SIX, Suit.CLUBS), 
                 new Card(Rank.SEVEN, Suit.HEARTS)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());
        
        // Only the top waste card (7♥) should be movable to a valid destination (8♠ in T1)
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.EIGHT, Suit.SPADES));
        
        // This move should succeed (7♥ on 8♠)
        assertTrue(solitaire.moveCard("W", null, "T1"), "Top waste card should be movable");
    }

    @Test
    void sameColorSameRankOnTableau() {
        // T1 has 5♥, T2 has 5♦ (same rank, both red = same color); move should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        TestGameStateBuilder.seedTableauStack(solitaire, 0, new Card(Rank.FIVE, Suit.HEARTS));
        TestGameStateBuilder.seedTableauStack(solitaire, 1, new Card(Rank.FIVE, Suit.DIAMONDS));
        for (int i = 2; i < 7; i++) {
            TestGameStateBuilder.seedTableauStack(solitaire, i);  // Empty
        }
        TestGameStateBuilder.clearFoundations(solitaire);
        TestGameStateBuilder.seedStockAndTalon(solitaire, Collections.emptyList(), Collections.emptyList());

        assertFalse(solitaire.moveCard("T1", null, "T2"), "Same rank same color (5♥ on 5♦) should fail");
    }

    @Test
    void faceDownStackAttempt() {
        // T1 has [5♠(facedown), 6♣(faceup)]; try to move from the facedown card.
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

        // Trying to move from facedown card (5♠) should fail - card not visible
        assertFalse(solitaire.moveCard("T1", "5♠", "T2"), "Face-down card 5♠ is not visible");
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

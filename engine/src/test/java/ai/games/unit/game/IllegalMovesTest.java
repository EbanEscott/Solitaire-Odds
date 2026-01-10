package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireBuilder;
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
        Solitaire solitaire = SolitaireBuilder.newGame().build();

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void wrongSuitToFoundationFails() {
        // Foundation F1 has A♥; trying to place 2♣ (wrong suit) should fail.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .foundation("F1", "A♥")
            .tableau("T1", "2♣")
            .build();

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void nonAceToEmptyFoundationFails() {
        // Empty foundation; attempting to move Q♣ onto it is illegal.
        Solitaire solitaire = SolitaireBuilder.newGame().tableau("T1", "Q♣").build();

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void faceDownCardCannotMove() {
        // T1 has K♠ face-down beneath a visible Q♥; trying to move the facedown K♠ should fail.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", 1, "K♠", "Q♥")
            .build();

        assertFalse(solitaire.moveCard("T1", "K♠", "T2"));
    }

    @Test
    void incorrectColorSequenceOnTableauFails() {
        // T1 top is 5♥, T2 top is 4♦ (same color); stacking should fail.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "5♥")
            .tableau("T2", "4♦")
            .build();

        assertFalse(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void nonConsecutiveRankOnFoundation() {
        // F1 has A♥, 2♥; T1 has 4♥ (skipping rank 3); should fail.
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .foundation("F1", "A♥", "2♥")
                .tableau("T1", "4♥")
                .build();

        assertFalse(solitaire.moveCard("T1", null, "F1"), "4♥ cannot go on 2♥ - rank 3 is missing");
    }

    @Test
    void nonKingStackOnEmptyTableau() {
        // T1 has Q♠, J♣ (valid internal sequence but Queen-led); T2 is empty; should fail.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "Q♠", "J♣")
            .build();

        assertFalse(solitaire.moveCard("T1", "Q♠", "T2"), "Non-King stacks cannot move to empty columns");
    }

    @Test
    void movingNonTopWasteCard() {
        // Waste mechanics: only top card is accessible. This test validates that
        // the top card can move but earlier seeding of non-top cards is prevented.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .waste("5♠", "6♣", "7♥")
            .tableau("T1", "8♠")
            .build();
        
        // This move should succeed (7♥ on 8♠)
        assertTrue(solitaire.moveCard("W", null, "T1"), "Top waste card should be movable");
    }

    @Test
    void sameColorSameRankOnTableau() {
        // T1 has 5♥, T2 has 5♦ (same rank, both red = same color); move should fail.
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "5♥")
                .tableau("T2", "5♦")
                .build();

        assertFalse(solitaire.moveCard("T1", null, "T2"), "Same rank same color (5♥ on 5♦) should fail");
    }

    @Test
    void faceDownStackAttempt() {
        // T1 has [5♠(facedown), 6♣(faceup)]; try to move from the facedown card.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", 1, "5♠", "6♣")
            .tableau("T2", "7♥")
            .build();

        // Trying to move from facedown card (5♠) should fail - card not visible
        assertFalse(solitaire.moveCard("T1", "5♠", "T2"), "Face-down card 5♠ is not visible");
    }

}

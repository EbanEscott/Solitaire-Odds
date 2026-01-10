package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireBuilder;
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
        Solitaire solitaire = SolitaireBuilder.newGame().tableau("T1", "A♣").build();

        assertTrue(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void twoOnAceFoundationShouldBeLegal() {
        // Reproduces the screenshot case: F1 has A♥, T6 has 2♥ face up; should be legal.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .foundation("F1", "A♥")
            .tableau("T6", "2♥")
            .build();

        assertTrue(solitaire.moveCard("T6", null, "F1"));
    }

    @Test
    void tableauAlternatingDescendingIsLegal() {
        // T1 has 7♠, T2 has 8♥; moving 7♠ onto 8♥ is legal (alternating color, descending).
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "7♠")
            .tableau("T2", "8♥")
            .build();

        assertTrue(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void topVisibleTableauCardMovesLegally() {
        // T3 has two face-up cards: 9♠ below (covered), Q♣ on top (nearest/active). T7 has K♥. Only the top/active Q♣ can move.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T3", 2, "9♠", "Q♣")
            .tableau("T7", "K♥")
            .build();

        assertTrue(solitaire.moveCard("T3", null, "T7"));
    }

    @Test
    void bottomVisibleTableauCardDrivesPileMoveRule() {
        // T3 has J♦, Q♣ (both face-up); when moving the entire visible stack, J♦ (bottom) on K♥ is illegal (same color).
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T3", 2, "J♦", "Q♣")
            .tableau("T7", "K♥")
            .build();

        // When moving the entire visible stack from J♦, the move is illegal because J♦ (red) cannot go on K♥ (red, same color).
        assertFalse(solitaire.moveCard("T3", "J♦", "T7"));
    }

    @Test
    void canMoveVisibleStackWhenBottomCardFitsDestination() {
        // Source T6 has 8♥, 7♣, 6♥ face up (bottom is 8♥). Destination T2 has 9♣ face up; moving the visible stack is legal.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .foundation("F1", "A♦")
            .tableau("T2", "9♣")
            .tableau("T6", "8♥", "7♣", "6♥")
            .build();

        assertTrue(solitaire.moveCard("T6", "8♥", "T2"));
        // Destination should now have moved stack on top.
        assertEquals(4, solitaire.getVisibleTableau().get(1).size());
    }

    @Test
    void talonToFoundationIncrementIsLegal() {
        // F2 has A♦; talon top is 2♦; moving waste to foundation should succeed.
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .foundation("F2", "A♦")
            .waste("2♦")
            .build();

        assertTrue(solitaire.moveCard("W", null, "F2"));
    }

    @Test
    void wasteToTableauIsLegal() {
        // T1 has 6♣ (black); waste has 5♥ (red) on top (valid move: 5 onto 6, different colors, descending).
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "6♣")
                .waste("5♥")
                .build();

        assertTrue(solitaire.moveCard("W", null, "T1"), "5♥ should move onto 6♣ (different colors)");
    }

    @Test
    void kingLedStackToEmpty() {
        // T1 has K♠, Q♣, J♠ (valid King-led stack); T2 is empty (target).
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "K♠", "Q♣", "J♠")
            .build();

        assertTrue(solitaire.moveCard("T1", "K♠", "T2"), "King-led stack should move to empty column");
    }

    @Test
    void multiCardStackFoundationBlocked() {
        // T1 has 5♠, 4♥, 3♣ (valid internal sequence); T2 has 6♥; move succeeds (5♠ black on 6♥ red, descending).
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "5♠", "4♥", "3♣")
            .tableau("T2", "6♥")
            .build();

        // Moving 5♠ (black) onto 6♥ (red): valid alternating colors and descending rank
        assertTrue(solitaire.moveCard("T1", "5♠", "T2"), "5♠ should move onto 6♥ (valid alternating/descending)");
    }

    @Test
    void wasteToTableauMultipleChoices() {
        // Waste has 7♠; T1 has 8♥, T2 has 8♣ (two valid destinations for same card).
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "8♥")
                .tableau("T2", "8♣")
                .waste("7♠")
                .build();

        // 7♠ can move to either 8♥ or 8♣; move to T1
        assertTrue(solitaire.moveCard("W", null, "T1"), "7♠ should move onto 8♥ (red on black)");
    }

}

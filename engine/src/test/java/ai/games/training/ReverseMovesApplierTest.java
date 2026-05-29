package ai.games.training;

import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireBuilder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for reverse-move state reconstruction.
 */
class ReverseMovesApplierTest {

    /**
     * Reverse tableau moves must preserve the hidden/visible split on both piles.
     */
    @Test
    void applyReverseMoveKeepsTableauFaceUpCountsConsistent() {
        Solitaire board = SolitaireBuilder.newGame()
                .tableau("T1", 3, "9♠", "8♥", "7♣", "6♦")
                .tableau("T2", 1, "5♠", "4♥")
                .build();

        boolean success = ReverseMovesApplier.applyReverseMove(board, "move T1 7♣ T2");

        assertTrue(success, "Reverse move should apply successfully");
        assertEquals(1, board.getTableauFaceUpCounts().get(0), "T1 should keep only its original visible suffix");
        assertEquals(3, board.getTableauFaceUpCounts().get(1), "T2 should gain visibility only for moved cards");
        SolitaireBuilder.assertValidGameState(board);
    }
}
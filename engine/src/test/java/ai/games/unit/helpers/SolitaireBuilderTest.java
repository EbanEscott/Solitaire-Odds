package ai.games.unit.helpers;

import ai.games.game.Solitaire;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SolitaireBuilderTest {

    @Test
    void exampleStyle_buildsCompleteStateAndMovesWork() {
        Solitaire game = SolitaireBuilder
                .newGame()
                .tableau("T1", "K♠")
                .tableau("T2", 2, "Q♥", "J♣")
                .foundation("F1", "A♥")
                .waste("3♠", "7♦")
                .build();

        // Empty tableau accepts only King: moving J♣ to empty column should fail.
        assertFalse(game.moveCard("T2", "T3"));

        // Moving K♠ to empty column should succeed.
        assertTrue(game.moveCard("T1", "T3"));

        assertEquals(1, game.getMoveHistorySize());
    }
}

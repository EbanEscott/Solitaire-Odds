package ai.games.unit.player;

import ai.games.game.Solitaire;
import ai.games.player.ai.RuleBasedHeuristicsPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Regression for the pilot's covered-card foundation command. */
class RuleBasedFoundationTest {
    @Test
    void movesCoveredFoundationCandidateOnlyAsATableauStack() {
        Solitaire board = SolitaireBuilder.newGame()
                .foundation("F1", "A♣", "2♣")
                .tableau("T1", "4♥")
                .tableau("T7", "3♣", "2♥").build();
        RuleBasedHeuristicsPlayer player = new RuleBasedHeuristicsPlayer();
        assertEquals("turn", player.nextCommand(board, "", ""));
        assertTrue(board.turnStock());
        String command = player.nextCommand(board, "", "");
        assertEquals("move T7 3♣ T1", command);
        assertTrue(board.moveCard("T7", "3♣", "T1"));
    }
}

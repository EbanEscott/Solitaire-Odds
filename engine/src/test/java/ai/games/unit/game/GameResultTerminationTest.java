package ai.games.unit.game;

import ai.games.Game;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Verifies benchmark termination reporting through the actual game loop. */
class GameResultTerminationTest {
    @Test
    void reportsQuitClosedInputAndIterationCap() {
        String previous = System.getProperty("max.moves.per.game");
        try {
            System.setProperty("max.moves.per.game", "2");
            assertEquals("quit", play((board, moves, feedback) -> "quit").getTerminationReason());
            assertEquals("input_closed", play((board, moves, feedback) -> null).getTerminationReason());
            Game.GameResult capped = play((board, moves, feedback) -> "turn");
            assertEquals("move_cap", capped.getTerminationReason());
            assertEquals(2, capped.getMoves());
            assertFalse(capped.isWon());
        } finally {
            if (previous == null) System.clearProperty("max.moves.per.game");
            else System.setProperty("max.moves.per.game", previous);
        }
    }

    /**
     * Runs a locally supplied player with guidance disabled and standard rules.
     *
     * @param player the command source
     * @return the game-loop result
     */
    private static Game.GameResult play(Player player) {
        Game game = new Game(player);
        game.setGuidanceEnabled(false);
        return game.play(new Solitaire(new Deck(), GameRules.DEFAULT));
    }
}

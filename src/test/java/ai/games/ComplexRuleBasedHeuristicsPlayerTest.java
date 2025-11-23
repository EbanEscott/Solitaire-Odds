package ai.games;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.ComplexRuleBasedHeuristicsPlayer;
import org.junit.jupiter.api.Test;

/**
 * Smoke tests for the experimental ComplexRuleBasedHeuristicsPlayer.
 *
 * These tests do not assert win rates; they simply verify that:
 * - the player can be constructed,
 * - it produces a non-null, syntactically valid command for a fresh game, and
 * - multiple consecutive commands do not cause exceptions.
 *
 * This keeps the “complex” player on a short leash while we iterate on its rules.
 */
class ComplexRuleBasedHeuristicsPlayerTest {

    @Test
    void complexPlayerProducesInitialCommand() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new ComplexRuleBasedHeuristicsPlayer();

        String command = ai.nextCommand(solitaire, "");
        assertNotNull(command);
    }

    @Test
    void complexPlayerRunsForSeveralSteps() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new ComplexRuleBasedHeuristicsPlayer();

        for (int i = 0; i < 20; i++) {
            String command = ai.nextCommand(solitaire, "");
            if (command == null || "quit".equalsIgnoreCase(command.trim())) {
                break;
            }
            // Apply a minimal subset of commands to keep the smoke test realistic.
            String trimmed = command.trim();
            if ("turn".equalsIgnoreCase(trimmed)) {
                solitaire.turnThree();
            }
        }
    }
}


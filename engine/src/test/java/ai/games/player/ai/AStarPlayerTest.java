package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.AStarPlayer;
import ai.games.unit.helpers.FoundationCountHelper;
import ai.games.unit.helpers.TestGameStateBuilder;
import org.junit.jupiter.api.Test;

/**
 * Basic behaviour tests for {@link AStarPlayer}.
 */
class AStarPlayerTest {

    private static final int MAX_TEST_STEPS = 2000;

    @Test
    void aStarImprovesNearlyWonGame() {
        Solitaire solitaire = TestGameStateBuilder.seedNearlyWonGameVariant();
        Player ai = new AStarPlayer();

        int startFoundation = FoundationCountHelper.totalFoundation(solitaire);

        for (int i = 0; i < 10 && FoundationCountHelper.totalFoundation(solitaire) < 52; i++) {
            String command = ai.nextCommand(solitaire, "", "");
            applyCommand(solitaire, command);
        }

        int endFoundation = FoundationCountHelper.totalFoundation(solitaire);
        assertTrue(endFoundation > startFoundation, "A* should improve foundation count on simple setup");
    }

    @Test
    void aStarDoesNotLoopForeverOnRandomGame() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new AStarPlayer();

        int steps = 0;
        while (!isTerminal(solitaire) && steps < MAX_TEST_STEPS) {
            String command = ai.nextCommand(solitaire, "", "");
            assertNotNull(command, "A* player should always return a command until game exits");
            if ("quit".equalsIgnoreCase(command.trim())) {
                break;
            }
            applyCommand(solitaire, command);
            steps++;
        }

        assertTrue(steps < MAX_TEST_STEPS, "A* player should not run indefinitely on a random game");
    }

    private boolean isTerminal(Solitaire solitaire) {
        return FoundationCountHelper.totalFoundation(solitaire) == 52
                || (solitaire.getStockpile().isEmpty()
                && solitaire.getTalon().isEmpty());
    }

    private void applyCommand(Solitaire solitaire, String command) {
        if (command == null) {
            return;
        }
        String trimmed = command.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else {
                solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
    }
}

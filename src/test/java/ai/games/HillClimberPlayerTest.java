package ai.games;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.HillClimberPlayer;
import ai.games.player.Player;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link HillClimberPlayer} focusing on improvement behaviour, backtracking,
 * random restarts, and basic safety (no infinite loops, no crashes).
 */
class HillClimberPlayerTest {

    private static final int MAX_TEST_STEPS = 2000;

    @Test
    void hillClimberAdvancesOnSimpleNearlyWonGame() {
        Solitaire solitaire = GreedySearchPlayerTestHelper.seedNearlyWonGameVariant();
        Player ai = new HillClimberPlayer(123L);

        int startFoundation = FoundationCountHelper.totalFoundation(solitaire);

        for (int i = 0; i < 10 && FoundationCountHelper.totalFoundation(solitaire) < 52; i++) {
            String command = ai.nextCommand(solitaire, "");
            applyCommand(solitaire, command);
        }

        int endFoundation = FoundationCountHelper.totalFoundation(solitaire);
        assertTrue(endFoundation > startFoundation, "Hill climber should improve foundation count on simple setup");
    }

    @Test
    void hillClimberDoesNotLoopForeverOnRandomGame() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new HillClimberPlayer(98765L);

        int steps = 0;
        while (!isTerminal(solitaire) && steps < MAX_TEST_STEPS) {
            String command = ai.nextCommand(solitaire, "");
            assertNotNull(command, "Hill climber should always return a command until game exits");
            if ("quit".equalsIgnoreCase(command.trim())) {
                break;
            }
            applyCommand(solitaire, command);
            steps++;
        }

        assertTrue(steps < MAX_TEST_STEPS, "Hill climber should not run indefinitely on a random game");
    }

    @Test
    void hillClimberExploresDifferentNeighboursUsingRestarts() {
        Solitaire solitaire = new Solitaire(new Deck());
        HillClimberPlayer ai = new HillClimberPlayer(42L);

        Set<String> uniqueCommands = new HashSet<>();
        for (int i = 0; i < 50 && !isTerminal(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "");
            assertNotNull(command, "Hill climber should provide commands for exploration");
            uniqueCommands.add(command.trim().toLowerCase());
            if ("quit".equalsIgnoreCase(command)) {
                break;
            }
            applyCommand(solitaire, command);
        }

        assertTrue(uniqueCommands.size() > 1, "Random restarts / hill-climb steps should explore different moves");
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


package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple completion test for the greedy AI: seeds a near-won state and expects the last move.
 */
class GreedySearchPlayerTest {
    private static final Logger log = LoggerFactory.getLogger(GreedySearchPlayerTest.class);
    private static final int MAX_TEST_STEPS = 1000;

    @Test
    void greedyAiFinishesGame() {
        logTestHeader("greedyAiFinishesGame");
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (nearly-won test):\n{}", stripAnsi(solitaire.toString()));
        }

        runSingleMoveCompletion(solitaire, ai);

        if (log.isDebugEnabled()) {
            log.debug("Final board (nearly-won test):\n{}", stripAnsi(solitaire.toString()));
        }

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void greedyAiWinsKnownMidGameState() {
        logTestHeader("greedyAiWinsKnownMidGameState");
        // Seed a winnable mid-game so the greedy AI has to make several choices.
        Solitaire solitaire = seedMidGameWinnable();
        GreedySearchPlayer.resetForNewGame();
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (mid-game test):\n{}", stripAnsi(solitaire.toString()));
        }

        for (int i = 0; i < MAX_TEST_STEPS && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiWinsKnownMidGameState: step {} command={}", i, command);
            }
            if (command == null) {
                break;
            }
            String trimmed = command.trim();
            if ("quit".equalsIgnoreCase(trimmed)) {
                break;
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
        }

        assertTrue(isWon(solitaire), "Greedy AI should win the seeded mid-game state");
    }

    @Test
    void greedyAiDoesNotQuitWithMovesAvailableInRandomGame() {
        logTestHeader("greedyAiDoesNotQuitWithMovesAvailableInRandomGame");
        GreedySearchPlayer.resetForNewGame();
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (random game test):\n{}", stripAnsi(solitaire.toString()));
        }

        for (int step = 0; step < MAX_TEST_STEPS && !isWon(solitaire); step++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiDoesNotQuitWithMovesAvailableInRandomGame: step {} command={}", step, command);
            }
            if (command == null) {
                break;
            }
            String trimmed = command.trim();
            if ("quit".equalsIgnoreCase(trimmed)) {
                List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
                boolean hasNonQuitMove = legal.stream().anyMatch(m -> !"quit".equalsIgnoreCase(m.trim()));
                assertTrue(!hasNonQuitMove, "Greedy quit while non-quit moves were still legal: " + legal);
                break;
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
        }
    }

    private void runSingleMoveCompletion(Solitaire solitaire, Player ai) {
        for (int i = 0; i < 5 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiFinishesGame: step {} command={}", i, command);
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
        }
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

    private Solitaire seedNearlyWonGame() {
        return SolitaireFactory.oneMoveFromWin();
    }

    private Solitaire seedMidGameWinnable() {
        return SolitaireBuilder
            .newGame()
            .foundation("F1", "A♠")
            .tableau("T1", "3♠")
            .tableau("T2", "2♠")
            .tableau("T3", "K♥")
            .tableau("T4", "J♣")
            .tableau("T5", "Q♦")
            .waste("4♠")
            .build();
    }

    private int totalFoundation(Solitaire solitaire) {
        int total = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total;
    }

    private boolean isWon(Solitaire solitaire) {
        return totalFoundation(solitaire) == 52;
    }

    private static String stripAnsi(String input) {
        return input.replaceAll("\\u001B\\[[;\\d]*m", "");
    }

    private static void logTestHeader(String testName) {
        if (!log.isDebugEnabled()) {
            return;
        }
        String banner = """
============================================================
   ____                 _           _        _             
  / ___| ___ _ __   ___| |__   __ _| |_ _ __(_)_ __   __ _ 
 | |  _ / _ \\ '_ \\ / __| '_ \\ / _` | __| '__| | '_ \\ / _` |
 | |_| |  __/ | | | (__| | | | (_| | |_| |  | | | | | (_| |
  \\____|\\___|_| |_|\\___|_| |_|\\__,_|\\__|_|  |_|_| |_|\\__, |
                                                      |___/ 
============================================================
GREEDY TEST START: %s
============================================================""".formatted(testName);
        log.debug("\n{}", banner);
    }
}

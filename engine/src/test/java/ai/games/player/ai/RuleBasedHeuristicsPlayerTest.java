package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Basic end-to-end test for the rule-based AI: seeds a near-complete game and expects the AI
 * to issue the final move to finish.
 */
class RuleBasedHeuristicsPlayerTest {
    private static final int MAX_TEST_STEPS = 1000;

    @Test
    void ruleBasedAiFinishesGame() {
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new RuleBasedHeuristicsPlayer();

        runSingleMoveCompletion(solitaire, ai);

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void ruleBasedAiWinsKnownMidGameState() {
        // Seed a winnable mid-game state: AI should push toward finishing given deterministic setup.
        Solitaire solitaire = seedMidGameWinnable();
        Player ai = new RuleBasedHeuristicsPlayer();

        // Allow several moves to reach completion.
        for (int i = 0; i < MAX_TEST_STEPS && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            applyCommand(solitaire, command);
        }

        assertTrue(isWon(solitaire), "AI should win the seeded mid-game state");
    }

    private void runSingleMoveCompletion(Solitaire solitaire, Player ai) {
        // Allow a few iterations in case AI chooses a turn first; but this scenario needs one move.
        for (int i = 0; i < 5 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            applyCommand(solitaire, command);
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
            .foundation("F1", "A♥")
            .tableau("T1", "3♥")
            .tableau("T2", "2♥")
            .tableau("T3", "K♠")
            .tableau("T4", "J♦")
            .tableau("T5", "Q♣")
            .waste("4♥")
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
}

package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.List;
import org.junit.jupiter.api.Test;

class BeamSearchPlayerTest {

    private static final int MAX_TEST_STEPS = 1000;

    @Test
    void beamSearchImprovesNearlyWonGame() {
        Solitaire solitaire = SolitaireFactory.oneMoveFromWin();
        Player ai = new BeamSearchPlayer(3, 8, 123L);

        int startFoundation = totalFoundation(solitaire);

        for (int i = 0; i < 10 && totalFoundation(solitaire) < 52; i++) {
            String command = ai.nextCommand(solitaire, "", "");
            applyCommand(solitaire, command);
        }

        int endFoundation = totalFoundation(solitaire);
        assertTrue(endFoundation > startFoundation, "Beam search should make progress toward completion");
    }

    @Test
    void beamSearchDoesNotLoopForeverOnRandomGame() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new BeamSearchPlayer(3, 8, 42L);

        int steps = 0;
        while (!isWon(solitaire) && steps < MAX_TEST_STEPS) {
            String command = ai.nextCommand(solitaire, "", "");
            if (command == null || "quit".equalsIgnoreCase(command.trim())) {
                break;
            }
            applyCommand(solitaire, command);
            steps++;
        }

        assertTrue(steps < MAX_TEST_STEPS, "Beam search should not run indefinitely on a random game");
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

    private boolean isWon(Solitaire solitaire) {
        return totalFoundation(solitaire) == 52;
    }

    private int totalFoundation(Solitaire solitaire) {
        int total = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total;
    }

    // Uses a strict, validated scenario from SolitaireFactory.
}

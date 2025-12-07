package ai.games;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.alpha.AlphaSolitaireClient;
import ai.games.player.ai.alpha.AlphaSolitairePlayer;
import org.junit.jupiter.api.Test;

/**
 * Basic integration test for {@link AlphaSolitairePlayer} that exercises the
 * HTTP call to the Python service.
 *
 * <p>For this test to fully exercise the network path, ensure the Python
 * service is running locally on http://127.0.0.1:8000 before executing:
 *
 *   cd /Users/ebo/Code/solitaire
 *   source .venv/bin/activate
 *   python3 service.py --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 */
class AlphaSolitairePlayerTest {

    @Test
    void nextCommandReturnsNonEmptyCommand() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new AlphaSolitairePlayer(new AlphaSolitaireClient("http://127.0.0.1:8000"));

        String command = ai.nextCommand(solitaire, "", "");

        assertNotNull(command, "AlphaSolitairePlayer should always return a command string");
        assertFalse(command.trim().isEmpty(), "AlphaSolitairePlayer command should not be blank");
    }
}

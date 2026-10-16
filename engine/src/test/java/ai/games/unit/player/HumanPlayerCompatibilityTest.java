package ai.games.unit.player;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.config.TrainingModeProperties;
import ai.games.player.HumanPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class HumanPlayerCompatibilityTest {
    @Test
    void consoleStillReturnsOriginalTextCommandsAndEof() {
        InputStream originalInput = System.in;
        try {
            System.setIn(new ByteArrayInputStream("  MOVE t1 ac f1  \nturn\nquit\n"
                    .getBytes(StandardCharsets.UTF_8)));
            HumanPlayer player = new HumanPlayer(new TrainingModeProperties());
            var state = SolitaireBuilder.newGame().tableau("T1", "A♣").build();

            assertEquals("  MOVE t1 ac f1  ", player.nextCommand(state, "", ""));
            assertEquals("turn", player.nextCommand(state, "", ""));
            assertEquals("quit", player.nextCommand(state, "", ""));
            assertNull(player.nextCommand(state, "", ""));
        } finally {
            System.setIn(originalInput);
        }
    }
}

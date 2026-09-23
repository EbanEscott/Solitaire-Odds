package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class CodexCliPlayerTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("codex.model");
        System.clearProperty("codex.reasoning.effort");
    }

    @Test
    void readsConfiguredModelAndReasoningEffort() {
        System.setProperty("codex.model", "gpt-test");
        System.setProperty("codex.reasoning.effort", "low");

        assertEquals("gpt-test", CodexCliPlayer.configuredModelName());
        assertEquals("low", CodexCliPlayer.configuredReasoningEffort());
    }

    @Test
    void usesCatalogReasoningEffortWhenPropertyIsAbsent() {
        System.clearProperty("codex.reasoning.effort");

        assertEquals("medium", CodexCliPlayer.configuredReasoningEffort("gpt-5.6-sol"));
        assertEquals("low", CodexCliPlayer.configuredReasoningEffort("gpt-6-astra"));
    }

    @Test
    void parsesStructuredCommand() throws IOException {
        assertEquals("turn", CodexCliPlayer.parseCommand("{\"command\":\"turn\"}"));
    }

    @Test
    void rejectsResponseWithoutCommand() {
        assertThrows(IOException.class, () -> CodexCliPlayer.parseCommand("not json"));
        assertThrows(IllegalStateException.class, () -> CodexCliPlayer.parseCommand("{}"));
    }
}

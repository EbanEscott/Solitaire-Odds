package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class CopilotCliPlayerTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("copilot.model");
    }

    @Test
    void readsConfiguredModel() {
        System.setProperty("copilot.model", "claude-test");

        assertEquals("claude-test", CopilotCliPlayer.configuredModelName());
    }

    @Test
    void parsesCompleteResponseAndUsage() throws IOException {
        String events = """
                {"type":"assistant.message_delta","data":{"deltaContent":"ignored"}}
                {"type":"assistant.message","data":{"model":"claude-haiku-4.5","content":"{\\"command\\":\\"turn\\"}"}}
                {"type":"result","sessionId":"session-123","exitCode":0,"usage":{"premiumRequests":0.33}}
                """;

        CopilotCliPlayer.CliResponse response = CopilotCliPlayer.parseResponse(events);

        assertEquals("{\"command\":\"turn\"}", response.content());
        assertEquals("claude-haiku-4.5", response.model());
        assertEquals("session-123", response.sessionId());
        assertEquals(0.33, response.premiumRequests(), 0.0001);
    }

    @Test
    void parsesPlainAndFencedCommands() throws IOException {
        assertEquals("turn", CopilotCliPlayer.parseCommand("{\"command\":\"turn\"}"));
        assertEquals(
                "move W T1",
                CopilotCliPlayer.parseCommand("```json\n{\"command\":\"move W T1\"}\n```"));
    }

    @Test
    void rejectsIncompleteEventsAndCommands() {
        assertThrows(
                IllegalStateException.class,
                () -> CopilotCliPlayer.parseResponse("{\"type\":\"result\",\"sessionId\":\"x\"}"));
        assertThrows(IOException.class, () -> CopilotCliPlayer.parseCommand("not json"));
        assertThrows(IllegalStateException.class, () -> CopilotCliPlayer.parseCommand("{}"));
    }

    @Test
    void extractsFailuresAndClassifiesOnlyTransientOnes() {
        String events = """
                {"type":"session.error","data":{"message":"Service unavailable"}}
                """;

        assertEquals("Service unavailable", CopilotCliPlayer.failureDetails(events, "fallback"));
        assertEquals("fallback", CopilotCliPlayer.failureDetails("not-json", "fallback"));
        assertTrue(CopilotCliPlayer.isTransientFailure("HTTP 503 service unavailable"));
        assertTrue(CopilotCliPlayer.isTransientFailure("connection reset"));
        assertFalse(CopilotCliPlayer.isTransientFailure("AI credit limit reached"));
        assertFalse(CopilotCliPlayer.isTransientFailure("authentication failed"));
    }

    @Test
    void usesP0StrategyWithoutHandAuthoredAdvice() {
        assertTrue(CopilotCliPlayer.STRATEGY_PROMPT.contains("strategy you already know"));
        assertTrue(CopilotCliPlayer.STRATEGY_PROMPT.contains("Java game engine"));
        assertFalse(CopilotCliPlayer.STRATEGY_PROMPT.contains("Decision Priorities"));
    }

    @Test
    void correctionPromptProvidesOnlyProtocolAndLegalMoves() {
        String prompt = CopilotCliPlayer.correctionPrompt(
                List.of("turn", "move W T1"), "I cannot do that");

        assertTrue(prompt.contains("I cannot do that"));
        assertTrue(prompt.contains("- turn"));
        assertTrue(prompt.contains("- move W T1"));
        assertTrue(prompt.contains("{\"command\":\"<listed value>\"}"));
        assertFalse(prompt.contains("reveal face-down"));
    }
}

package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import java.io.IOException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class CodexCliPlayerTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("codex.model");
        System.clearProperty("codex.reasoning.effort");
        System.clearProperty(GamePrompts.PROMPT_PROFILE_PROPERTY);
        System.clearProperty(GamePrompts.LEGACY_PROMPT_PROFILE_PROPERTY);
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
    void parsesSessionIdFromJsonEvents() throws IOException {
        String events = """
                {"type":"thread.started","thread_id":"session-123"}
                {"type":"turn.started"}
                {"type":"turn.completed"}
                """;

        assertEquals("session-123", CodexCliPlayer.parseSessionId(events));
    }

    @Test
    void extractsStructuredCodexFailureFromJsonEvents() {
        String events = """
                {"type":"event_msg","payload":{"type":"task_started"}}
                {"type":"event_msg","payload":{"type":"task_complete","error":{"message":"Selected model is at capacity. Please try a different model.","codex_error_info":"server_overloaded"}}}
                """;

        assertEquals(
                "Selected model is at capacity. Please try a different model. [server_overloaded]",
                CodexCliPlayer.failureDetails(events, ""));
    }

    @Test
    void fallsBackToStderrForCodexFailureDetails() {
        assertEquals(
                "transport failed",
                CodexCliPlayer.failureDetails("not-json", "transport failed"));
    }

    @Test
    void retriesOnlyTransientCodexFailures() {
        assertTrue(CodexCliPlayer.isTransientFailure("Selected model is at capacity [server_overloaded]"));
        assertTrue(CodexCliPlayer.isTransientFailure("request failed [internal_server_error]"));
        assertTrue(CodexCliPlayer.isTransientFailure("no diagnostics emitted"));
        assertTrue(CodexCliPlayer.isTransientFailure("HTTP 503 service unavailable"));
        assertTrue(CodexCliPlayer.isTransientFailure("stream disconnected before completion"));
        assertFalse(CodexCliPlayer.isTransientFailure("usage limit reached [usage_limit_exceeded]"));
        assertFalse(CodexCliPlayer.isTransientFailure("authentication failed [unauthorized]"));
    }

    @Test
    void asksForExistingStrategyBeforeStartingTheGame() {
        assertTrue(CodexCliPlayer.STRATEGY_PROMPT.contains("strategy you already know"));
        assertFalse(CodexCliPlayer.STRATEGY_PROMPT.contains("Decision Priorities"));
    }

    @Test
    void sendsGameInterfaceOnlyWithTheFirstBoard() {
        Solitaire solitaire = new Solitaire(new Deck());

        String firstPrompt = CodexCliPlayer.buildTurnPrompt(
                solitaire, "", java.util.List.of("turn"), true);
        String resumedPrompt = CodexCliPlayer.buildTurnPrompt(
                solitaire, "", java.util.List.of("turn"), false);

        assertTrue(firstPrompt.contains("# Game interface"));
        assertFalse(resumedPrompt.contains("# Game interface"));
        assertFalse(firstPrompt.contains("Decision Priorities"));
        assertTrue(firstPrompt.contains("# Current board"));
        assertTrue(resumedPrompt.contains("# Current board"));
        assertTrue(resumedPrompt.contains("# Complete legal-move list\n- turn"));
    }

    @Test
    void rejectsResponseWithoutCommand() {
        assertThrows(IOException.class, () -> CodexCliPlayer.parseCommand("not json"));
        assertThrows(IllegalStateException.class, () -> CodexCliPlayer.parseCommand("{}"));
    }

    @Test
    void rejectsEventsWithoutSessionId() {
        assertThrows(
                IllegalStateException.class,
                () -> CodexCliPlayer.parseSessionId("{\"type\":\"turn.started\"}"));
    }
}

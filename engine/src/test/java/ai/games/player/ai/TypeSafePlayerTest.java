package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class TypeSafePlayerTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @AfterEach
    void clearProperties() {
        System.clearProperty("typesafe.model");
        System.clearProperty("typesafe.baseUrl");
        System.clearProperty(GamePrompts.PROMPT_PROFILE_PROPERTY);
        System.clearProperty(GamePrompts.LEGACY_PROMPT_PROFILE_PROPERTY);
    }

    @Test
    void readsConfiguredModelAndEndpoint() {
        System.setProperty("typesafe.model", "jev-test");
        System.setProperty("typesafe.baseUrl", "https://example.test/systemone");

        assertEquals("jev-test", TypeSafePlayer.configuredModelName());
        assertEquals("https://example.test/systemone", TypeSafePlayer.configuredBaseUrl());
    }

    @Test
    void sendsStructuredStateAndMapsTypedChoiceToCommand() throws Exception {
        AtomicReference<String> capturedPayload = new AtomicReference<>();
        TypeSafePlayer.TypeSafeTransport transport = payload -> {
            capturedPayload.set(payload);
            return new TypeSafePlayer.TypeSafeHttpResponse(
                    200,
                    """
                    {
                      "model": "jev-1.13.0",
                      "answers": {
                        "next_move": {
                          "type": "choice",
                          "choice": "move_0",
                          "confidence": 0.75,
                          "probabilities": {"move_0": 0.75, "move_1": 0.25}
                        }
                      },
                      "usage": {"input_tokens": 321, "output_tokens": 20}
                    }
                    """,
                    0L);
        };
        TypeSafePlayer player = new TypeSafePlayer("jev-1.13.0", transport, 1, 0L);
        Solitaire solitaire = new Solitaire(new Deck());
        String expected = ai.games.player.LegalMovesHelper.listLegalMoves(solitaire).getFirst();

        String command = player.nextCommand(solitaire, "ignored guidance", "");

        assertEquals(expected, command);
        JsonNode request = JSON.readTree(capturedPayload.get());
        assertEquals("jev-1.13.0", request.path("model").asText());
        JsonNode state = request.path("state");
        JsonNode board = state.path("current_turn").path("observed_board");
        assertTrue(board.path("tableau").isObject());
        assertTrue(board.path("tableau").path("T1").path("face_up_bottom_to_top").isArray());
        assertTrue(board.path("foundations").path("F1").has("top"));
        assertTrue(board.path("waste").has("visible_top"));
        assertFalse(board.path("waste").has("cards"));
        assertTrue(state.path("previous_turns").isArray());
        assertFalse(state.has("rules"));
        assertEquals(
                expected,
                request.path("questions").path("next_move").path("criteria")
                        .path("move_0").path("command").asText());
        assertFalse(
                request.path("questions").path("next_move").path("criteria")
                        .path("move_0").path("effect").asText().isBlank());
        assertFalse(request.path("questions").path("next_move").path("criteria")
                .path("move_0").has("strategic_assessment"));
        assertEquals(321L, player.getTotalInputTokens());
        assertEquals(0.75, (double) player.getLastDecisionMetadata().get("confidence"), 0.0001);
    }

    @Test
    void retainsObservedBoardsAndCommandsAcrossStatelessRequests() throws Exception {
        AtomicReference<String> secondPayload = new AtomicReference<>();
        int[] calls = {0};
        TypeSafePlayer.TypeSafeTransport transport = payload -> {
            calls[0]++;
            if (calls[0] == 2) {
                secondPayload.set(payload);
            }
            return new TypeSafePlayer.TypeSafeHttpResponse(
                    200,
                    """
                    {"model":"jev-1.13.0","answers":{"next_move":{"type":"choice",
                    "choice":"move_0","confidence":1.0,"probabilities":{"move_0":1.0}}},
                    "usage":{"input_tokens":100,"output_tokens":10}}
                    """,
                    0L);
        };
        TypeSafePlayer player = new TypeSafePlayer("jev-1.13.0", transport, 1, 0L);
        Solitaire solitaire = new Solitaire(new Deck());
        String firstCommand = player.nextCommand(solitaire, "", "");

        player.nextCommand(solitaire, "", "");

        JsonNode history = JSON.readTree(secondPayload.get()).path("state").path("previous_turns");
        assertEquals(1, history.size());
        assertEquals(firstCommand, history.get(0).path("command").asText());
        assertEquals(4, history.get(0).path("F").size());
        assertEquals(7, history.get(0).path("T").size());
        assertEquals(7, history.get(0).path("hidden").size());
        assertEquals(2, history.get(0).path("waste").size());
        assertEquals(2, JSON.readTree(secondPayload.get())
                .path("state").path("current_turn").path("turn").asInt());
    }

    @Test
    void detailedProfileAddsRulesAndDecisionPriorities() throws Exception {
        System.setProperty(GamePrompts.PROMPT_PROFILE_PROPERTY, "detailed");
        AtomicReference<String> capturedPayload = new AtomicReference<>();
        TypeSafePlayer.TypeSafeTransport transport = payload -> {
            capturedPayload.set(payload);
            return new TypeSafePlayer.TypeSafeHttpResponse(
                    200,
                    """
                    {"model":"jev-1.13.0","answers":{"next_move":{"type":"choice",
                    "choice":"move_0","confidence":1.0,"probabilities":{"move_0":1.0}}},
                    "usage":{"input_tokens":100,"output_tokens":10}}
                    """,
                    0L);
        };
        TypeSafePlayer player = new TypeSafePlayer("jev-1.13.0", transport, 1, 0L);

        player.nextCommand(new Solitaire(new Deck()), "", "");

        JsonNode state = JSON.readTree(capturedPayload.get()).path("state");
        assertEquals("Build four same-suit foundations from Ace through King.",
                state.path("rules").path("objective").asText());
        assertEquals(8, state.path("decision_priorities").size());
        assertEquals(0, state.path("progress_summary").path("consecutive_stock_turns").asInt());
        JsonNode question = JSON.readTree(capturedPayload.get()).path("questions").path("next_move");
        assertTrue(question.path("instructions").asText().contains("already_seen"));
        assertTrue(question.path("criteria").path("move_0").path("strategic_assessment")
                .has("face_down_cards_revealed"));
        assertTrue(question.path("criteria").path("move_0").path("strategic_assessment")
                .has("required_by_progress_rule"));
        assertEquals("detailed", player.getExperimentMetadata().get("prompt_profile"));
        assertEquals("P1.5", player.getExperimentMetadata().get("prompt_version"));
        assertEquals("P1.5", player.getExperimentMetadata().get("policy_version"));
    }

    @Test
    void parsesChoiceProbabilitiesAndRejectsUnknownOptions() {
        Map<String, String> options = Map.of("move_0", "turn", "move_1", "quit");
        String valid = """
                {"model":"jev-1.13.0","answers":{"next_move":{"choice":"move_0",
                "confidence":0.6,"probabilities":{"move_0":0.7,"move_1":0.3}}},
                "usage":{"input_tokens":50,"output_tokens":12}}
                """;

        TypeSafePlayer.TypeSafeDecision decision = TypeSafePlayer.parseDecision(valid, options);

        assertEquals("move_0", decision.option());
        assertEquals(0.7, decision.probabilities().get("move_0"), 0.0001);
        assertThrows(
                IllegalStateException.class,
                () -> TypeSafePlayer.parseDecision(valid.replace("move_0\"", "invented\""), options));
    }

    @Test
    void classifiesOnlyRateLimitsAndServerErrorsAsTransient() {
        assertTrue(TypeSafePlayer.isTransientStatus(429));
        assertTrue(TypeSafePlayer.isTransientStatus(503));
        assertFalse(TypeSafePlayer.isTransientStatus(400));
        assertFalse(TypeSafePlayer.isTransientStatus(401));
    }

    @Test
    void validatesConfiguration() {
        TypeSafePlayer.TypeSafeTransport unused = payload ->
                new TypeSafePlayer.TypeSafeHttpResponse(200, "{}", 0L);

        assertThrows(IllegalArgumentException.class, () -> new TypeSafePlayer("", unused, 1, 0L));
        assertThrows(IllegalArgumentException.class, () -> new TypeSafePlayer("jev", unused, 0, 0L));
        assertThrows(IllegalArgumentException.class, () -> new TypeSafePlayer("jev", unused, 1, -1L));
    }
}

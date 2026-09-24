package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import org.springframework.ai.ollama.api.ThinkOption;

class OllamaPlayerTest {

    @Test
    void usesSelfAuthoredStrategyAndRetainsOnlyRecentTurns() {
        RecordingChatModel model = new RecordingChatModel(
                "Move useful cards to foundations and expose hidden tableau cards.",
                "{\"command\":\"turn\"}",
                "{\"command\":\"turn\"}");
        OllamaPlayer player = new OllamaPlayer(model, "test-model", 1, "game-1");
        Solitaire solitaire = new Solitaire(new Deck());

        assertEquals("turn", player.nextCommand(solitaire, "ignored recommendation", "first feedback"));
        assertEquals("turn", player.nextCommand(solitaire, "ignored recommendation", "second feedback"));

        List<Message> retained = player.retainedMessages();
        assertEquals(3, retained.size());
        assertTrue(retained.get(0) instanceof SystemMessage);
        assertTrue(retained.get(0).getText().contains("Move useful cards"));
        assertTrue(retained.get(0).getText().contains("# Game interface"));
        assertFalse(retained.stream().anyMatch(message -> message.getText().contains("first feedback")));
        assertTrue(retained.stream().anyMatch(message -> message.getText().contains("second feedback")));

        assertEquals(3, model.prompts.size());
        assertTrue(model.prompts.get(0).getContents().contains("strategy you already know"));
        assertTrue(model.prompts.get(1).getContents().contains("# Complete legal-move list"));
        assertFalse(model.prompts.get(1).getContents().contains("ignored recommendation"));
    }

    @Test
    void commandSchemaContainsOnlyCurrentLegalMoves() {
        Map<String, Object> schema = OllamaPlayer.commandSchema(List.of("turn", "move W T1"));

        @SuppressWarnings("unchecked")
        Map<String, Object> properties = (Map<String, Object>) schema.get("properties");
        @SuppressWarnings("unchecked")
        Map<String, Object> command = (Map<String, Object>) properties.get("command");

        assertEquals(List.of("turn", "move W T1"), command.get("enum"));
        assertEquals(false, schema.get("additionalProperties"));
    }

    @Test
    void parsesStructuredCommandAndRejectsMalformedResponses() {
        assertEquals("move W T1", OllamaPlayer.parseCommand("{\"command\":\"move W T1\"}"));
        assertThrows(IllegalStateException.class, () -> OllamaPlayer.parseCommand("not json"));
        assertThrows(IllegalStateException.class, () -> OllamaPlayer.parseCommand("{}"));
    }

    @Test
    void rejectsAnEmptyMemoryWindow() {
        RecordingChatModel model = new RecordingChatModel("unused");

        assertThrows(
                IllegalArgumentException.class,
                () -> new OllamaPlayer(model, "test-model", 0, "game-1"));
    }

    @Test
    void configuresThinkingExplicitlyAndRejectsUnknownModes() {
        OllamaChatOptions.Builder disabled = OllamaChatOptions.builder();
        OllamaPlayer.configureThinking(disabled, "off");
        assertEquals(ThinkOption.ThinkBoolean.DISABLED, disabled.build().getThinkOption());

        OllamaChatOptions.Builder medium = OllamaChatOptions.builder();
        OllamaPlayer.configureThinking(medium, "medium");
        assertEquals(ThinkOption.ThinkLevel.MEDIUM, medium.build().getThinkOption());

        OllamaChatOptions.Builder automatic = OllamaChatOptions.builder();
        OllamaPlayer.configureThinking(automatic, "auto");
        assertEquals(null, automatic.build().getThinkOption());

        assertThrows(
                IllegalArgumentException.class,
                () -> OllamaPlayer.configureThinking(OllamaChatOptions.builder(), "extreme"));
    }

    private static final class RecordingChatModel implements ChatModel {
        private final Queue<String> responses;
        private final List<Prompt> prompts = new ArrayList<>();

        private RecordingChatModel(String... responses) {
            this.responses = new ArrayDeque<>(List.of(responses));
        }

        @Override
        public ChatResponse call(Prompt prompt) {
            prompts.add(prompt);
            String response = responses.remove();
            return new ChatResponse(List.of(new Generation(new AssistantMessage(response))));
        }
    }
}

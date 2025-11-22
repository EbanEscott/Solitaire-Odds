package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Ollama-backed AI player using Spring AI. Requires a local Ollama server with a chat model available.
 * Responds with a single command: "turn", "quit", or "move FROM [CARD] TO" (e.g., move T6 Q♣ F2).
 */
@Component
@Profile("ai-ollama")
public class OllamaPlayer extends AIPlayer implements Player {
    private final ChatClient chatClient;
    private static final String LOCAL_OLLAMA_URL = "http://localhost:11434";
    private static final String DEFAULT_MODEL = "llama3";
    private static final String SYSTEM_PROMPT = """
            You are playing Klondike Solitaire. Output exactly one command per reply:
            - "turn" to flip from stock to talon.
            - "quit" to stop.
            - "move FROM [CARD] TO" to move cards (e.g., "move W T3", "move T7 Q♣ F1", "move T6 5♠ T2").
            Rules:
            - Tableau build: alternating colors, descending ranks; empty tableau accepts only a King stack.
            - Foundation build: same suit, ascending from Ace to King; move only single top cards to foundation.
            - You can move a visible stack starting at a chosen face-up card onto a tableau target if legal.
            - Always respond with just the command, no explanation.
            """;

    /**
     * Default constructor for non-Spring contexts (e.g., tests). Builds a client that talks to the
     * local Ollama instance on {@value LOCAL_OLLAMA_URL}.
     */
    public OllamaPlayer() {
        this(buildLocalChatClient());
    }

    @Autowired
    public OllamaPlayer(ChatClient.Builder builder) {
        this(builder.build());
    }

    private OllamaPlayer(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    @Override
    public String nextCommand(Solitaire solitaire) {
        String board = solitaire.toString();
        String response = chatClient.prompt()
                .system(SYSTEM_PROMPT)
                .user(board)
                .call()
                .content();
        return response == null ? "quit" : response.trim();
    }

    private static ChatClient buildLocalChatClient() {
        OllamaApi api = OllamaApi.builder()
                .baseUrl(LOCAL_OLLAMA_URL)
                .build();

        OllamaChatModel model = OllamaChatModel.builder()
                .ollamaApi(api)
                .defaultOptions(OllamaChatOptions.builder()
                        .model(DEFAULT_MODEL)
                        .build())
                .build();

        return ChatClient.builder(model).build();
    }
}

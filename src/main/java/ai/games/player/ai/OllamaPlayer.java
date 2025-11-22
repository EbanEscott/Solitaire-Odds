package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Ollama-backed AI player using Spring AI. Requires a local Ollama server with
 * a chat model available.
 * Responds with a single command: "turn", "quit", or "move FROM [CARD] TO"
 * (e.g., move T6 Q♣ F2).
 */
@Component
@Profile("ai-ollama")
public class OllamaPlayer extends AIPlayer implements Player {
    private final ChatClient chatClient;
    private static final String LOCAL_OLLAMA_URL = "http://localhost:11434";
    private static final String DEFAULT_MODEL = "llama3";
    private static final String SYSTEM_PROMPT = """
            You are an expert Klondike Solitaire player. Your job is to choose the best next move from the current board.

            Board layout you will receive (example):

            FOUNDATION
              F1, F2, F3, F4 are the four suit piles at the top.
              Each F pile shows its current top card, or “--” if empty.

            TABLEAU
              T1..T7 are the seven tableau columns in the middle.
              Each tableau column shows a vertical stack.
              Face-up cards are shown explicitly with rank+suit (e.g., 7♦, Q♣).
              Face-down/hidden cards are shown as "..▼" or similar.
              The “top” playable card of a tableau column is the face-up card closest to the player (the lowest visible card in that column).

            STOCKPILE & TALON
              STOCK is the undealt pile. It may show a count like “12 down”.
              TALON (waste) shows its top card (playable) plus a count in brackets.
              The top of the talon is referenced as W.

            Example mapping from a board:
            - “T6 K♣” means in tableau column 6, a visible K♣ is present and can be referenced.
            - “W 3♦” means the talon top card is 3♦.
            - “F1 --” means foundation 1 is empty.

            Output format (exactly one line, no extra text):
            - turn
            - quit
            - move <FROM> <TO>

            Where:
            - FROM is one of:
              - W (top of talon)
              - T1..T7 (tableau, either the top face-up card or a specific visible card you name)
              - F1..F4 (foundation top card, single-card moves only)
            - TO is one of:
              - T1..T7
              - F1..F4

            Examples:
            - move W T3
            - move T7 Q♣ F1
            - move T6 5♠ T2

            You must NOT default to "turn". Only turn when no beneficial legal move exists.

            Decision policy (apply in order):
            1) If any move to a FOUNDATION is legal AND does not block progress, do it.
               - Move Aces and Twos up immediately when possible.
               - Prefer a foundation move if it does not remove the only card enabling tableau progress.
               - Avoid foundation moves that trap needed lower cards in tableau.

            2) If a TABLEAU move reveals a face-down card, prioritise it.
               - Uncovering hidden cards is the strongest type of move.
               - Prefer moves that uncover a new card over purely cosmetic rearrangements.

            3) Create or preserve empty tableau columns for Kings.
               - If a tableau column is empty, try to move a King (or King-led stack) into it.
               - Do not empty a column unless you have a King available soon.

            4) Improve tableau structure.
               - Build descending rank, alternating colour stacks.
               - Prefer moving longer correct stacks.
               - Prefer moves that free low cards (A–5) toward foundation.

            5) Use the TALON intelligently.
               - If W can move to foundation or tableau under the rules above, do it before turning again.

            6) If no legal move improves the position:
               - turn
               - If stock is empty and no legal moves exist, quit.

            Legality reminders:
            - Tableau: descending rank, alternating colour. Only Kings can be placed on empty tableau.
            - Foundation: ascending rank, same suit, starting at Ace. Only single top cards move to foundation.
            - You may move a visible stack starting at any named face-up card in a tableau column if the destination tableau card allows it.

            Final constraint:
            Respond with exactly one command line, nothing else.
            """;

    /**
     * Default constructor for non-Spring contexts (e.g., tests). Builds a client
     * that talks to the
     * local Ollama instance on {@value LOCAL_OLLAMA_URL}.
     */
    public OllamaPlayer() {
        this(buildLocalChatClient(DEFAULT_MODEL));
    }

    @Autowired
    public OllamaPlayer(@Value("${ollama.model:" + DEFAULT_MODEL + "}") String modelName) {
        this(buildLocalChatClient(modelName));
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

    private static ChatClient buildLocalChatClient(String modelName) {
        OllamaApi api = OllamaApi.builder()
                .baseUrl(LOCAL_OLLAMA_URL)
                .build();

        OllamaChatModel model = OllamaChatModel.builder()
                .ollamaApi(api)
                .defaultOptions(OllamaChatOptions.builder()
                        .model(modelName)
                        .build())
                .build();

        return ChatClient.builder(model).build();
    }
}

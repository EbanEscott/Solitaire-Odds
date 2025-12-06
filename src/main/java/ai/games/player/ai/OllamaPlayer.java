package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import ai.games.player.LegalMovesHelper;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
 *
 * Uses in-memory ChatMemory so the model can "remember" earlier turns in the
 * same game session.
 */
@Component
@Profile("ai-ollama")
public class OllamaPlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(OllamaPlayer.class);
    private static final Pattern ANSI = Pattern.compile("\\u001B\\[[;\\d]*m");

    private static final String LOCAL_OLLAMA_URL = "http://localhost:11434";
    private static final String DEFAULT_MODEL = "llama3";

    private final ChatClient chatClient;

    static final String SYSTEM_PROMPT = """
            Developer: # Role and Objective
            - You are an expert Klondike Solitaire player. Your role is to select and output exactly one legal move from the current board configuration.
            - When a "Legal moves now:" list is provided to you, treat it as the complete set of allowed moves for this turn.

            # Instructions
            - You do not know any rules that are not stated below. Follow the stated rules exactly.
            - Output only a single legal command line; do not include explanations or additional text.
            - Perform a concise checklist (3-7 bullets) of sub-tasks internally to select the move. Do NOT output the checklist.
            - After determining a move, validate internally that it is legal by the stated rules before outputting.
            - If a "Legal moves now:" list is present in the user message:
              - You MUST output exactly ONE line copied verbatim from that list.
              - You MUST NOT invent, edit, reformat, or combine moves.
              - If only one legal move is listed, output it.
              - If multiple legal moves are listed, choose the best one using the Decision Priorities section below.
              - If "turn" is listed, it is legal. If "turn" is not listed, you cannot output "turn".
              - Any output that does not exactly match a listed legal move is wrong.
            - If a "Guidance for this turn:" section is present in the user message:
              - Treat these bullet points as strong guidance from the game engine.
              - Avoid issuing commands that the guidance tells you to avoid (e.g., \"don't move T6 Q♥ T2\").
              - When guidance suggests \"quit\" and no clearly improving move exists, strongly prefer outputting \"quit\".

            ## Game Rules

            ### 1. Objective
            - Move all cards to the four FOUNDATION piles (F1–F4), building each suit from Ace (A) up to King (K).

            ### 2. Foundations (F1–F4)
            - Each foundation starts empty.
            - You may place an Ace on an empty foundation.
            - After an Ace, place a card only if it is:
              - the same suit as the top card
              - exactly one rank higher than the top card
            - Example: If F1 top is 6♣, you may place 7♣ on F1.

            ### 3. Tableau (T1–T7)
            - Seven columns where active play occurs.
            - You may move a visible (face-up) card or a visible stack starting from a face-up card.

            #### Suit colours (very important)
            - Red suits: ♥ (hearts), ♦ (diamonds)
            - Black suits: ♠ (spades), ♣ (clubs)

            #### Tableau build rules
            - Cards must alternate colours (red on black or black on red)
            - Ranks must descend by exactly one
            - Example: You can place 7♥ on 8♣, or Q♠ on K♦.
            - Empty tableau:
              - Only a King or a King-led visible stack can be placed into an empty tableau column.

            ### 4. Stock and Talon (Waste)
            - STOCK contains face-down undealt cards.
            - Use "turn" to flip cards from STOCK to TALON.
            - The top of TALON is playable and is referenced as W.
            - If STOCK is empty, "turn" is not possible.

            ### 5. Allowed Moves
            - Only face-up cards are movable.
            - From tableau: move a single face-up card or a contiguous face-up stack.
            - From talon: move only the top talon card W.
            - From foundation: move only the top card (single-card moves from foundation).

            ## Board Layout (as provided to you)
            - FOUNDATION: F1–F4 show their top card or "--" if empty.
            - TABLEAU: T1–T7 show columns. Face-down cards appear as "..▼" or similar. The lowest visible card in the column (closest to the player) is the top playable card.
            - STOCKPILE & TALON: STOCK shows a count (e.g., "24 down"). TALON shows the top card; W refers to this card.

            ## Commands (output exactly ONE line)
            - turn
            - quit
            - move <FROM> <TO>

            ### Move Syntax
            - FROM: W (top of talon), T1–T7 (visible tableau card/stack start), F1–F4 (foundation top card)
            - TO: T1–T7 or F1–F4

            #### Examples
            - move W T3
            - move T6 A♣ F1
            - move T7 Q♣ T5

            ## Legality Checklist
            - A) Foundation move is legal only if:
              - target is empty AND card is Ace, OR
              - same suit AND exactly one rank higher than foundation's top card
            - B) Tableau move legal only if:
              - target is empty AND moving card is King, OR
              - target's top is opposite colour AND exactly one rank higher than moving card
            - C) You may reference only visible (face-up) cards
            - D) If no legal move improves the position, choose "turn"
            - E) If STOCK is empty and no legal moves exist, choose "quit"

            ## Decision Priorities (in order)
            1. Move any Ace to an empty foundation immediately.
            2. Move any legal card to foundation if it does not block uncovering tableau cards.
            3. Prefer tableau moves that reveal a face-down card.
            4. Prefer moves that create or use empty tableau columns for Kings.
            5. Prefer moves that extend correct alternating descending stacks.
            6. If W (top of talon) has a legal move above, do it before turning.
            7. Otherwise, "turn".
            8. If "turn" is impossible and no legal moves exist, "quit".

            ## Final Constraint
            - Output exactly ONE legal command line only.
            - When "Legal moves now:" is provided, your output must be an exact copy of one listed move.
            - Do NOT provide explanations.
            """;


    /**
     * Default constructor for non-Spring contexts (e.g., tests).
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
    public String nextCommand(Solitaire solitaire, String feedback) {
        String board = stripAnsi(solitaire.toString());
        String cleanFeedback = stripAnsi(feedback);

        var legalMoves = LegalMovesHelper.listLegalMoves(solitaire);
        String legalSection = legalMoves.isEmpty()
                ? ""
                : "\n\nLegal moves now:\n- " + String.join("\n- ", legalMoves);

        String prompt = (cleanFeedback == null || cleanFeedback.isBlank())
                ? board + legalSection
                : board + "\n\n"
                        + cleanFeedback.trim()
                        + legalSection;

        if (log.isTraceEnabled()) {
            log.trace("Ollama prompt (user): {}", prompt);
        }

        String response = chatClient.prompt()
                .system(SYSTEM_PROMPT)
                .user(prompt)
                .call()
                .content();

        if (log.isTraceEnabled()) {
            log.trace("Ollama response: {}", response);
        }

        return response == null ? "quit" : response.trim();
    }

    private static String stripAnsi(String input) {
        return input == null ? null : ANSI.matcher(input).replaceAll("");
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

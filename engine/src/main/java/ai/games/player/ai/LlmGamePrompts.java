package ai.games.player.ai;

import ai.games.game.Solitaire;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Shared experiment prompts for LLM-backed Solitaire players.
 *
 * <p>The engine defines only the game interface and legal response boundary. It deliberately does
 * not supply rules or strategy: each model first states the Klondike knowledge it already has.
 */
final class LlmGamePrompts {

    static final String PROMPT_VERSION = "P0";
    private static final Pattern ANSI = Pattern.compile("\\u001B\\[[;\\d]*m");

    static final String STRATEGY_PROMPT = """
            You are about to play one complete game of Klondike Solitaire.

            Before the game begins, describe the Klondike rules and playing strategy you already know.
            Explain how you intend to choose moves and plan across the game. Rely on your existing
            knowledge: do not use tools, inspect files, or ask questions. Your response will remain in
            this conversation as your strategy for the game.
            """;

    static final String GAME_INTERFACE = """
            # Game interface
            The game now begins. Each turn provides the complete current board and the complete
            legal-move list. Locations use F1-F4 for foundations, T1-T7 for tableau columns,
            and W for the top waste card. Choose exactly one listed command without inventing
            or rewriting it. Remember earlier boards, commands, and feedback when planning later moves.
            """;

    private LlmGamePrompts() {
    }

    /** Builds one turn without adding rules or strategic recommendations. */
    static String buildTurnPrompt(
            Solitaire solitaire, String feedback, List<String> legalMoves, boolean includeInterface) {
        StringBuilder prompt = new StringBuilder();
        if (includeInterface) {
            prompt.append(GAME_INTERFACE);
        }

        prompt.append("\n# Current board\n")
                .append(stripAnsi(solitaire.toString()))
                .append("\n\n# Complete legal-move list\n");
        legalMoves.forEach(move -> prompt.append("- ").append(move).append('\n'));

        if (feedback != null && !feedback.isBlank()) {
            prompt.append("\n# Engine feedback\n").append(stripAnsi(feedback.trim())).append('\n');
        }
        prompt.append("\nReturn only the JSON object required by the response schema.");
        return prompt.toString();
    }

    /** Removes terminal colour codes before text is sent to a model. */
    static String stripAnsi(String input) {
        return input == null ? null : ANSI.matcher(input).replaceAll("");
    }
}

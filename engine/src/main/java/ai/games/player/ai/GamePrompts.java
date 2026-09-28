package ai.games.player.ai;

import ai.games.game.Solitaire;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Shared experiment prompts and policy for model-backed Solitaire players.
 *
 * <p>P0 defines only the game interface and legal response boundary, while P1.5 supplies explicit
 * rules, priorities, and progress requirements. A profile always selects these elements together.
 */
final class GamePrompts {

    static final String PROMPT_PROFILE_PROPERTY = "game.prompt.profile";
    static final String LEGACY_PROMPT_PROFILE_PROPERTY = "llm.prompt.profile";
    static final String PROMPT_VERSION = "P0";
    static final String DETAILED_POLICY_VERSION = "P1.5";
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

    static final Map<String, Object> DETAILED_RULES = detailedRules();

    static final List<String> DETAILED_DECISION_PRIORITIES = List.of(
            "Prefer moves that reveal a face-down tableau card.",
            "Move Aces and clearly safe low cards to foundations.",
            "Create and use empty tableau columns deliberately, especially for useful Kings.",
            "Play productive waste cards before turning the stock.",
            "Preserve tableau mobility; avoid foundation moves that strand needed cards.",
            "Use previous_turns to track stock and waste order across passes.",
            "Avoid reversals and repeated positions unless they enable concrete progress.",
            "Quit only when stock passes and legal rearrangements offer no productive path.");

    static final String MOVE_SELECTION_INSTRUCTIONS =
            "Choose the command you would play next in this draw-three Klondike Solitaire game. "
                    + "The objective is to move all cards to the foundations. Return one listed option.";

    static final String MOVE_SELECTION_INSTRUCTIONS_DETAILED =
            "Choose the command that makes the strongest concrete progress toward winning. "
                    + "If any option has required_by_progress_rule=true, you must choose one of those "
                    + "options. Otherwise use each strategic_assessment to prefer productive tableau "
                    + "or waste moves. Strongly avoid a resulting position already_seen in previous_turns. "
                    + "Choose turn only when no legal move makes progress and its resulting position is "
                    + "not a repeated stock-cycle state. Choose quit only when every non-quit option is "
                    + "unproductive. Return one listed option.";

    static final String STALL_INSTRUCTION =
            "When either count is high, reject moves that do not break the stall.";

    static final String STRATEGY_PROMPT_DETAILED = detailedStrategyPrompt();

    private static String detailedStrategyPrompt() {
        return """
            You are about to play one complete game of draw-three Klondike Solitaire.

            Before the game begins, form a strategy using these rules and priorities.

            Objective: %s
            Tableau: %s Only exposed tableau cards may move; uncovering a face-down card turns it face up.
            Stock: %s Previously seen waste cards can become available on later passes.

            Use this decision order as a starting point, adapting it when the position demands:
            %s

            Move selection: %s

            Describe the plan you will follow before seeing the first board. Do not use tools, inspect
            files, or ask questions. Your response will remain in this conversation throughout the game.
            """.formatted(
                DETAILED_RULES.get("objective"),
                DETAILED_RULES.get("tableau"),
                DETAILED_RULES.get("stock"),
                numberedPriorities(),
                MOVE_SELECTION_INSTRUCTIONS_DETAILED);
    }

    static final String GAME_INTERFACE_DETAILED = """
            # Game interface
            The game now begins. Each turn provides the complete current board and an exhaustive,
            engine-validated legal-move list. Foundations are F1-F4, tableau columns are T1-T7, and W
            is the playable top talon card. Tableau displays show visible cards in movable order and
            identify any face-down cards beneath them. The stock and talon display records how many
            cards remain in each location.

            Commands such as `move W T4`, `move F1 T4`, and `turn` are game protocol values, not shell
            commands. Tableau-source commands also include the displayed card token. Choose exactly
            one command from the legal-move list without inventing, abbreviating, or rewriting it.
            Return it in the required JSON object. Remember earlier boards, observed cards, commands,
            and engine feedback when planning later moves.
            """;

    private static final PromptSet P0 =
            new PromptSet(
                    "p0",
                    PROMPT_VERSION,
                    STRATEGY_PROMPT,
                    GAME_INTERFACE,
                    Map.of(),
                    List.of(),
                    MOVE_SELECTION_INSTRUCTIONS);
    private static final PromptSet DETAILED = new PromptSet(
            "detailed",
            DETAILED_POLICY_VERSION,
            STRATEGY_PROMPT_DETAILED,
            GAME_INTERFACE_DETAILED,
            DETAILED_RULES,
            DETAILED_DECISION_PRIORITIES,
            MOVE_SELECTION_INSTRUCTIONS_DETAILED);

    private GamePrompts() {
    }

    /** Returns the matched strategy and interface prompts selected for a new game. */
    static PromptSet configuredPromptSet() {
        String configuredProfile = System.getProperty(PROMPT_PROFILE_PROPERTY);
        if (configuredProfile == null) {
            configuredProfile = System.getProperty(LEGACY_PROMPT_PROFILE_PROPERTY, P0.profile());
        }
        return promptSet(configuredProfile);
    }

    /** Resolves a prompt profile while accepting its human name or version identifier. */
    static PromptSet promptSet(String configuredProfile) {
        String profile = configuredProfile == null
                ? P0.profile()
                : configuredProfile.trim().toLowerCase(Locale.ROOT);
        return switch (profile) {
            case "p0" -> P0;
            case "detailed", "p1", "p1.5" -> DETAILED;
            default -> throw new IllegalArgumentException(
                    "Unsupported game prompt profile '" + configuredProfile
                            + "'; use p0 or detailed");
        };
    }

    /** Builds one turn without adding rules or strategic recommendations. */
    static String buildTurnPrompt(
            Solitaire solitaire, String feedback, List<String> legalMoves, boolean includeInterface) {
        return buildTurnPrompt(
                solitaire, feedback, legalMoves, configuredPromptSet(), includeInterface);
    }

    /** Builds one turn from a prompt set captured when the game-scoped player was created. */
    static String buildTurnPrompt(
            Solitaire solitaire,
            String feedback,
            List<String> legalMoves,
            PromptSet promptSet,
            boolean includeInterface) {
        StringBuilder prompt = new StringBuilder();
        if (includeInterface) {
            prompt.append(promptSet.gameInterface());
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

    /** Converts a legal command into provider-independent game-interface semantics. */
    static String commandEffect(String command) {
        String[] parts = command.split(" ");
        if (command.equals("turn")) {
            return "Draw up to three cards from stock onto the waste, or recycle the waste when stock is empty.";
        }
        if (command.equals("quit")) {
            return "End the game immediately without winning.";
        }
        if (parts.length == 4 && parts[1].startsWith("T")) {
            return "Move the face-up tableau stack beginning with " + parts[2]
                    + " from " + parts[1] + " onto " + parts[3] + ".";
        }
        if (parts.length == 3 && parts[1].equals("W")) {
            return "Move the playable top waste card onto " + parts[2] + ".";
        }
        if (parts.length == 3 && parts[1].startsWith("F")) {
            return "Move the top foundation card from " + parts[1] + " back onto " + parts[2] + ".";
        }
        return "Apply this legal game command.";
    }

    /** Returns the shared detailed-policy recommendation for computed move facts. */
    static String moveRecommendation(
            String command,
            int hiddenDelta,
            int foundationDelta,
            long priorOccurrences,
            boolean returnsToCurrent,
            int consecutiveStockTurns,
            int turnsSinceProgress) {
        if (hiddenDelta > 0) {
            return "Highest priority: reveals a face-down tableau card.";
        }
        if (foundationDelta > 0) {
            return "Strong progress: adds a card to a foundation.";
        }
        if (priorOccurrences > 0 || returnsToCurrent) {
            return "Avoid: returns to an already observed position without measurable progress.";
        }
        if (command.startsWith("move W")) {
            return "Potentially productive: removes the playable waste card.";
        }
        if (command.startsWith("move T")) {
            if (turnsSinceProgress >= 8) {
                return "Low value during the current stall: rearranges tableau without immediate measurable progress.";
            }
            return "Potentially productive tableau rearrangement.";
        }
        if (command.startsWith("move F")) {
            return "Use only when moving a foundation card back enables another concrete move.";
        }
        if (command.equals("turn")) {
            if (consecutiveStockTurns >= 8) {
                return "Strongly avoid: the game is already cycling through the stock without progress.";
            }
            return "Use only to expose a new waste card when no productive move is available.";
        }
        return "No measurable immediate progress.";
    }

    /** Creates the structured form of the detailed rules for non-chat decision providers. */
    private static Map<String, Object> detailedRules() {
        Map<String, Object> rules = new LinkedHashMap<>();
        rules.put("objective", "Build four same-suit foundations from Ace through King.");
        rules.put(
                "tableau",
                "Build downward in alternating colours; movable visible stacks remain ordered; only a King-led stack enters an empty column.");
        rules.put(
                "stock",
                "Draw three cards at a time. Only the visible waste top is playable. When stock is empty, turn recycles the waste with unlimited passes.");
        return Collections.unmodifiableMap(rules);
    }

    /** Renders the canonical priority list for conversational model prompts. */
    private static String numberedPriorities() {
        StringBuilder priorities = new StringBuilder();
        for (int index = 0; index < DETAILED_DECISION_PRIORITIES.size(); index++) {
            if (index > 0) {
                priorities.append('\n');
            }
            priorities.append(index + 1)
                    .append(". ")
                    .append(DETAILED_DECISION_PRIORITIES.get(index));
        }
        return priorities.toString();
    }

    /** A versioned, matched pair used for both pre-game strategy and turn instructions. */
    record PromptSet(
            String profile,
            String version,
            String strategyPrompt,
            String gameInterface,
            Map<String, Object> rules,
            List<String> decisionPriorities,
            String moveSelectionInstructions) {
    }
}

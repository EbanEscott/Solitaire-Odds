package ai.games;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.WebApplicationType;

@SpringBootApplication
public class Game implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(Game.class);
    /** When true, emit structured per-move episode logs for training. */
    private static final boolean EPISODE_LOG_ENABLED = Boolean.getBoolean("log.episodes");

    private final Player player;

    public Game(Player player) {
        this.player = player;
    }

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(Game.class);
        app.setWebApplicationType(WebApplicationType.NONE);
        app.run(args);
    }
    
    @Override
    public void run(String... args) {
        // CLI entrypoint ignores the result; tests and other harnesses
        // can call play() directly and use the returned GameResult.
        play();
    }

    /**
     * Core game loop used by both the CLI runner and automated tests.
     *
     * <p>The loop is intentionally structured in clearly separated phases:
     * <ol>
     *     <li>Build a {@link TurnView} from the current board (guidance, recommended moves, feedback).</li>
     *     <li>Render the view to the console (board, guidance, recommended moves).</li>
     *     <li>Ask the player for the next command.</li>
     *     <li>Track ping-pong behaviour and enforce safety limits.</li>
     *     <li>Apply the command and update statistics and guidance.</li>
     * </ol>
     *
     * @return summary of whether the player won, how many successful moves
     *         were applied, and how long the game took.
     */
    public GameResult play() {
        // Each play() call starts from a fresh, shuffled deck.
        Deck deck = new Deck();
        Solitaire solitaire = new Solitaire(deck);
        boolean aiMode = player instanceof AIPlayer;
        String solverId = player.getClass().getSimpleName();
        // Textual feedback passed into the player for the next decision.
        String feedback = "";
        // "Recommended moves" string passed into the player (LLMs) each turn.
        String moves = "";
        // Detailed explanation of why the last command was illegal (if it was).
        String illegalFeedback = "";
        // Long-lived guidance entries that survive across turns (e.g., "don't keep doing X").
        java.util.Map<String, Guidance> persistentGuidance = new java.util.HashMap<>();
        // Tracks last commands so we can detect repetition and ping-pong patterns.
        PingPongState pingPongState = new PingPongState();
        int turnsSinceLastMove = 0;
        int iterations = 0;
        // Tracks stock passes and how many moves happened between stock empty events.
        StockStats stockStats = new StockStats();
        int successfulMoves = 0;
        long startNanos = System.nanoTime();
        boolean won = false;
        final int maxPingPongRepeats = 12;
        final int initialGuidanceTtlTurns = 32;
        final int maxGuidanceTtlTurns = 100;
        // Cap iterations at ~8× a typical winning game (≈120–135 moves incl. stock turns). Anything beyond this
        // is overwhelmingly likely to be looping or non-productive searching, so we bail out to keep runs finite.
        //
        // Tests can further constrain this via the "max.moves.per.game" system property, which is used to
        // align with ResultsConfig.MAX_MOVES_PER_GAME without introducing a direct dependency on test code.
        final int maxIterations = Integer.getInteger("max.moves.per.game", 10_000);

        while (true) {
            // Build the "view" of this turn (guidance + recommended moves + feedback).
            TurnView view = buildTurnView(solitaire, persistentGuidance, iterations, illegalFeedback);
            feedback = view.feedbackForPlayer;
            moves = view.movesForPrompt;

            // Render the current board and guidance for humans following along.
            printTurnView(solitaire, aiMode, view, iterations);
            if (isWon(solitaire)) {
                won = true;
                if (log.isDebugEnabled()) {
                    log.debug("🎉🤗🎉 Congrats, you moved every card to the foundations! 🎉🤗🎉");
                    log.debug("Game won by {}", player.getClass().getSimpleName());
                }
                break;
            }

            // If this is a human player, prompt on the console.
            if (!aiMode) {
                if (log.isDebugEnabled()) {
                    log.debug("Enter command (turn | move FROM TO | quit): ");
                }
            }

            // Ask the player (AI or human) for the next command using the
            // recommended moves and feedback we just prepared.
            String input = player.nextCommand(solitaire, moves, feedback);
            if (input == null) {
                if (log.isDebugEnabled()) {
                    log.debug("Input closed. Exiting for player {}", player.getClass().getSimpleName());
                }
                break;
            }

            // Normalise basic formatting artefacts (e.g., LLM copying bullet "- turn").
            input = input.trim();
            if (input.startsWith("- ")) {
                input = input.substring(2).trim();
            }

            // Log this step for training, if enabled.
            if (EPISODE_LOG_ENABLED) {
                java.util.List<String> legalMovesAtStart = LegalMovesHelper.listLegalMoves(solitaire);
                logEpisodeStep(solitaire, solverId, iterations, legalMovesAtStart, view.recommendedMoves, input, won);
            }

            // Track simple repetition and ping-pong patterns (A,B,A,B,...).
            boolean pingPongLimitHit = trackPingPongs(
                    input, aiMode, maxPingPongRepeats, pingPongState, player);
            if (pingPongLimitHit) {
                feedback = "Ping-pong limit exceeded; forcing quit.";
                illegalFeedback = "";
                break;
            }

            // Update iteration count and enforce a hard safety cap.
            iterations++;
            if (iterations > maxIterations) {
                if (log.isDebugEnabled()) {
                    log.debug(
                            "Maximum iteration limit reached ({}); stopping game loop for {} to avoid runaway execution.",
                            maxIterations,
                            player.getClass().getSimpleName());
                }
                break;
            }

            // Process the command and update the game state.
            int stockBefore = solitaire.getStockpile().size();
            CommandResult commandResult = processCommand(
                    solitaire,
                    input,
                    successfulMoves,
                    turnsSinceLastMove,
                    stockStats,
                    persistentGuidance,
                    illegalFeedback,
                    player);

            illegalFeedback = commandResult.illegalFeedback;
            successfulMoves = commandResult.successfulMoves;
            turnsSinceLastMove = commandResult.turnsSinceLastMove;
            boolean quitRequested = commandResult.quitRequested;

            // Update long-lived guidance based on this command's effects.
            updateGuidanceAfterCommand(
                    solitaire,
                    input,
                    stockBefore,
                    iterations,
                    pingPongState,
                    persistentGuidance,
                    stockStats,
                    initialGuidanceTtlTurns,
                    maxGuidanceTtlTurns);

            if (aiMode) {
                if (log.isDebugEnabled()) {
                    log.debug("AI command: {}", input);
                }
            }

            if (quitRequested) {
                break;
            }
        }
        long durationNanos = System.nanoTime() - startNanos;
        if (EPISODE_LOG_ENABLED) {
            logEpisodeSummary(solitaire, solverId, iterations, successfulMoves, won, durationNanos);
        }
        return new GameResult(won, successfulMoves, durationNanos);
    }

    /**
     * Build a {@link TurnView} for the current board state.
     *
     * <p>This method is pure with respect to game progression: it does not mutate the
     * {@link Solitaire} instance. It:
     * <ul>
     *     <li>Computes legal moves for the current position.</li>
     *     <li>Filters and formats any persistent guidance that still applies.</li>
     *     <li>Derives a filtered "recommended moves" list for this turn.</li>
     *     <li>Combines illegal-move feedback and guidance into the feedback string
     *         that will be passed into the player.</li>
     * </ul>
     */
    private TurnView buildTurnView(
            Solitaire solitaire,
            java.util.Map<String, Guidance> persistentGuidance,
            int iterations,
            String illegalFeedback) {

        // Legal moves for the *current* board, before any new command is applied.
        java.util.List<String> legalMovesAtStart = LegalMovesHelper.listLegalMoves(solitaire);

        java.util.List<String> suggestionLinesForDisplay = new java.util.ArrayList<>();
        // Fold over existing guidance entries and keep only those that are still
        // alive and relevant to at least one legal move on this turn.
        java.util.Iterator<java.util.Map.Entry<String, Guidance>> displayIt =
                persistentGuidance.entrySet().iterator();
        while (displayIt.hasNext()) {
            var entry = displayIt.next();
            String cmd = entry.getKey();
            Guidance g = entry.getValue();

            // Expire old guidance only after its TTL.
            if (!g.stillAlive(iterations)) {
                displayIt.remove();
                continue;
            }

            // Show only when relevant to the current legal move set,
            // but keep it alive even when temporarily illegal.
            if (!legalMovesAtStart.contains(cmd)) {
                continue;
            }

            if ("turn".equalsIgnoreCase(cmd)) {
                continue;
            }

            if ("quit".equalsIgnoreCase(cmd)) {
                suggestionLinesForDisplay.add("quit (" + g.reason + ")");
            } else {
                suggestionLinesForDisplay.add("don't " + cmd + " (" + g.reason + ")");
            }
        }

        // If "quit" is legal but we have not yet added any explicit guidance
        // recommending it (i.e., no persistent entry for "quit"), gently steer
        // the player away from quitting too early.
        boolean quitLegal = legalMovesAtStart.stream()
                .anyMatch(cmd -> "quit".equalsIgnoreCase(cmd));
        if (quitLegal && !persistentGuidance.containsKey("quit")) {
            suggestionLinesForDisplay.add("don't quit (keep playing; quit will be suggested only after repeated unproductive stock passes).");
        }

        String suggestionsForDisplay = "";
        if (!suggestionLinesForDisplay.isEmpty()) {
            suggestionsForDisplay = "Guidance for this turn:\n- "
                    + String.join("\n- ", suggestionLinesForDisplay);
        }

        // Build a "recommended moves" list for this turn:
        //  - Start from all legal moves except "quit".
        //  - Drop any commands (including 'quit', if present) that currently
        //    have guidance attached, since they are being discouraged.
        //  - If we have explicit guidance for "quit" (added by
        //    updateGuidanceAfterCommand after repeated unproductive stock passes),
        //    re-add "quit" as a recommended move.
        //  - Never recommend moving a lone tableau King when there are no
        //    face-down cards beneath it to reveal; such moves are legal but
        //    almost always pointless shuffling.
        java.util.List<String> recommendedMovesForThisTurn = new java.util.ArrayList<>();
        for (String move : legalMovesAtStart) {
            if (!"quit".equalsIgnoreCase(move)) {
                recommendedMovesForThisTurn.add(move);
            }
        }
        for (java.util.Map.Entry<String, Guidance> entry : persistentGuidance.entrySet()) {
            recommendedMovesForThisTurn.remove(entry.getKey());
        }
        if (persistentGuidance.containsKey("quit")
                && legalMovesAtStart.stream().anyMatch(cmd -> "quit".equalsIgnoreCase(cmd))) {
            recommendedMovesForThisTurn.add("quit");
        }

        // Filter out pointless King shuffles from tableau to tableau.
        filterPointlessTableauKingMoves(solitaire, recommendedMovesForThisTurn);

        // Build feedback text and the recommended-moves string that will be passed
        // into the player for this turn.
        String feedback;
        if (!illegalFeedback.isBlank() && !suggestionsForDisplay.isBlank()) {
            feedback = illegalFeedback + "\n\n" + suggestionsForDisplay;
        } else if (!illegalFeedback.isBlank()) {
            feedback = illegalFeedback;
        } else if (!suggestionsForDisplay.isBlank()) {
            feedback = suggestionsForDisplay;
        } else {
            feedback = "";
        }

        String movesForPrompt;
        if (!recommendedMovesForThisTurn.isEmpty()) {
            movesForPrompt = "Recommended moves now:\n- "
                    + String.join("\n- ", recommendedMovesForThisTurn);
        } else {
            movesForPrompt = "";
        }

        return new TurnView(suggestionsForDisplay, recommendedMovesForThisTurn, feedback, movesForPrompt);
    }

    /**
     * Emit a single structured JSON line describing this move and the current state.
     *
     * <p>The line is prefixed with "EPISODE_STEP " so downstream tools can
     * filter it out of mixed logs easily.</p>
     */
    private void logEpisodeStep(
            Solitaire solitaire,
            String solverId,
            int stepIndex,
            java.util.List<String> legalMoves,
            java.util.List<String> recommendedMoves,
            String chosenCommand,
            boolean won) {

        try {
            long stateKey = solitaire.getStateKey();
            java.util.List<java.util.List<Card>> visibleTableau = solitaire.getVisibleTableau();
            java.util.List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
            java.util.List<java.util.List<Card>> foundation = solitaire.getFoundation();
            java.util.List<Card> talon = solitaire.getTalon();
            java.util.List<Card> stockpile = solitaire.getStockpile();

            String gameIndex = System.getProperty("game.index");
            String gameTotal = System.getProperty("game.total");

            StringBuilder sb = new StringBuilder();
            sb.append("{\"type\":\"step\"");
            if (gameIndex != null) {
                sb.append(",\"game_index\":").append(gameIndex);
            }
            if (gameTotal != null) {
                sb.append(",\"game_total\":").append(gameTotal);
            }
            sb.append(",\"solver\":\"").append(solverId).append("\"");
            sb.append(",\"step_index\":").append(stepIndex);
            sb.append(",\"state_key\":").append(stateKey);
            sb.append(",\"won\":").append(won);

            // Encode visible tableau as short card codes plus face-down counts.
            sb.append(",\"tableau_visible\":[");
            for (int i = 0; i < visibleTableau.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('[');
                java.util.List<Card> pile = visibleTableau.get(i);
                for (int j = 0; j < pile.size(); j++) {
                    if (j > 0) {
                        sb.append(',');
                    }
                    sb.append('"').append(pile.get(j).shortName()).append('"');
                }
                sb.append(']');
            }
            sb.append(']');

            sb.append(",\"tableau_face_down\":[");
            for (int i = 0; i < faceDownCounts.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(faceDownCounts.get(i));
            }
            sb.append(']');

            // Foundation piles: full piles as short codes.
            sb.append(",\"foundation\":[");
            for (int i = 0; i < foundation.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('[');
                java.util.List<Card> pile = foundation.get(i);
                for (int j = 0; j < pile.size(); j++) {
                    if (j > 0) {
                        sb.append(',');
                    }
                    sb.append('"').append(pile.get(j).shortName()).append('"');
                }
                sb.append(']');
            }
            sb.append(']');

            // Talon (waste) cards, short codes, in order.
            sb.append(",\"talon\":[");
            for (int i = 0; i < talon.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(talon.get(i).shortName()).append('"');
            }
            sb.append(']');

            // Stockpile size (we do not expose hidden order by default to the model).
            sb.append(",\"stock_size\":").append(stockpile.size());

            // Legal moves at start of turn.
            sb.append(",\"legal_moves\":[");
            for (int i = 0; i < legalMoves.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(legalMoves.get(i)).append('"');
            }
            sb.append(']');

            // Recommended moves for this turn (after guidance filters).
            sb.append(",\"recommended_moves\":[");
            for (int i = 0; i < recommendedMoves.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(recommendedMoves.get(i)).append('"');
            }
            sb.append(']');

            // Command actually chosen and applied.
            sb.append(",\"chosen_command\":\"").append(chosenCommand).append('"');
            sb.append('}');

            if (log.isInfoEnabled()) {
                log.info("EPISODE_STEP {}", sb);
            }
        } catch (Exception e) {
            // Logging must never interfere with gameplay.
            if (log.isDebugEnabled()) {
                log.debug("Failed to log episode step", e);
            }
        }
    }

    /**
     * Emit a single structured JSON line summarising the whole game.
     */
    private void logEpisodeSummary(
            Solitaire solitaire,
            String solverId,
            int iterations,
            int successfulMoves,
            boolean won,
            long durationNanos) {

        try {
            String gameIndex = System.getProperty("game.index");
            String gameTotal = System.getProperty("game.total");

            StringBuilder sb = new StringBuilder();
            sb.append("{\"type\":\"summary\"");
            if (gameIndex != null) {
                sb.append(",\"game_index\":").append(gameIndex);
            }
            if (gameTotal != null) {
                sb.append(",\"game_total\":").append(gameTotal);
            }
            sb.append(",\"solver\":\"").append(solverId).append("\"");
            sb.append(",\"iterations\":").append(iterations);
            sb.append(",\"successful_moves\":").append(successfulMoves);
            sb.append(",\"won\":").append(won);
            sb.append(",\"duration_nanos\":").append(durationNanos);
            sb.append('}');

            if (log.isInfoEnabled()) {
                log.info("EPISODE_SUMMARY {}", sb);
            }
        } catch (Exception e) {
            if (log.isDebugEnabled()) {
                log.debug("Failed to log episode summary", e);
            }
        }
    }

    /**
     * Render the current board state to the console, along with guidance and
     * the "Recommended moves now:" section.
     *
     * <p>For AI players this is purely for humans reading the logs; for human
     * players the guidance also appears as "Feedback:" so they can see the same
     * information the AIs receive.
     */
    private void printTurnView(Solitaire solitaire, boolean aiMode, TurnView view, int iterations) {
        String gameIndex = System.getProperty("game.index");
        String gameTotal = System.getProperty("game.total");
        StringBuilder sb = new StringBuilder();
        sb.append("\n--------------------------------------------------------------------------------------------------\n");
        if (gameIndex != null && gameTotal != null) {
            sb.append("GAME ")
                    .append(gameIndex)
                    .append("/")
                    .append(gameTotal)
                    .append(" MOVE ")
                    .append(iterations + 1)
                    .append('\n');
        } else {
            sb.append("MOVE ")
                    .append(iterations + 1)
                    .append('\n');
        }
        
        // Add board (with or without colour based on aiMode)
        if (!aiMode) {
            sb.append(solitaire.toString()).append('\n');
        } else {
            sb.append(stripAnsi(solitaire.toString())).append('\n');
        }
        
        // Common guidance and recommendations for both human and AI players
        if (!view.suggestionsForDisplay.isBlank()) {
            sb.append(view.suggestionsForDisplay).append('\n');
        }
        if (!view.recommendedMoves.isEmpty()) {
            sb.append("Recommended moves now:\n");
            for (String move : view.recommendedMoves) {
                sb.append("- ").append(move).append('\n');
            }
        }
        if (!view.feedbackForPlayer.isBlank()) {
            sb.append("Feedback: ").append(view.feedbackForPlayer).append('\n');
        }
        
        // Log based on player type
        if (!aiMode) {
            log.info("{}", sb);
        } else if (log.isDebugEnabled()) {
            log.debug("{}", sb);
        }
    }

    /**
     * Update long-lived guidance entries after a command has been applied.
     *
     * <p>This method is responsible for:
     * <ul>
     *     <li>Marking commands that are repeated too often as "don't" recommendations.</li>
     *     <li>Detecting ping-pong between two commands and advising against both.</li>
     *     <li>Tracking unproductive passes through the stock and suggesting "quit"
     *         after several such passes.</li>
     * </ul>
     */
    private void updateGuidanceAfterCommand(
            Solitaire solitaire,
            String input,
            int stockBefore,
            int iterations,
            PingPongState pingPongState,
            java.util.Map<String, Guidance> persistentGuidance,
            StockStats stockStats,
            int initialGuidanceTtlTurns,
            int maxGuidanceTtlTurns) {

            // Suggestion 1: repeated identical command (same move over and over).
            if (pingPongState.sameCommandCount >= 4 && !input.equalsIgnoreCase("turn")) {
                String reason = "you have chosen this many times without making progress.";
                Guidance g = persistentGuidance.get(input);
                if (g == null) {
                    persistentGuidance.put(input, new Guidance(reason, 1, initialGuidanceTtlTurns, iterations));
                } else {
                    g.refresh(iterations, 3, maxGuidanceTtlTurns);
                }
            }

            // Suggestion 1b: ping-pong between two commands (A,B,A,B,...).
            // Use current legal moves to tune TTL: if a move is no longer legal, let it decay faster.
            java.util.List<String> legalMovesForPingPong = LegalMovesHelper.listLegalMoves(solitaire);
            if (pingPongState.pingPongCount >= 3
                    && pingPongState.lastCommand != null
                    && pingPongState.secondLastCommand != null) {
                String reason = "you are ping-ponging between two moves without improving the board.";
                for (String cmd : new String[]{ pingPongState.lastCommand, pingPongState.secondLastCommand }) {
                    int ttlBoost = legalMovesForPingPong.contains(cmd) ? 4 : 1;
                    Guidance g = persistentGuidance.get(cmd);
                    if (g == null) {
                        // Start with a relatively long time-to-live (TTL) so
                        // "don't ..." guidance is not forgotten too quickly.
                        persistentGuidance.put(cmd, new Guidance(reason, 1, initialGuidanceTtlTurns, iterations));
                    } else {
                        g.refresh(iterations, ttlBoost, maxGuidanceTtlTurns);
                    }
                }
            }

            // Suggestion 2: STOCK emptied without any successful moves since the last empty.
            int stockAfter = solitaire.getStockpile().size();
            if (input.equalsIgnoreCase("turn")
                    && stockBefore > 0
                    && stockAfter == 0) {

                if (stockStats.movesSinceLastStockEmpty == 0) {
                    stockStats.stockEmptyStrikes++;
                } else {
                    stockStats.stockEmptyStrikes = 0;
                }
                stockStats.movesSinceLastStockEmpty = 0;

                // Only start warning to quit after repeated unproductive passes.
                if (stockStats.stockEmptyStrikes >= 2) {
                    String reason = "you have turned through the entire stockpile "
                            + stockStats.stockEmptyStrikes + " times without making any moves.";
                    Guidance g = persistentGuidance.get("quit");
                    if (g == null) {
                        persistentGuidance.put("quit", new Guidance(reason, 1, initialGuidanceTtlTurns, iterations));
                    } else {
                        g.refresh(iterations, 6, maxGuidanceTtlTurns);
                    }
                }
            }
    }

    /**
     * Track simple repetition and ping-pong (A,B,A,B,...) patterns for commands.
     *
     * <p>Returns {@code true} when the ping-pong safety limit is reached and the
     * caller should force-quit the game for this AI.
     */
    private boolean trackPingPongs(
            String input,
            boolean aiMode,
            int maxPingPongRepeats,
            PingPongState state,
            Player player) {

        // Increment repetition counters.
        if (state.lastCommand != null && input.equalsIgnoreCase(state.lastCommand)) {
            state.sameCommandCount++;
        } else {
            state.sameCommandCount = 1;
        }
        if (state.secondLastCommand != null
                && input.equalsIgnoreCase(state.secondLastCommand)
                && !input.equalsIgnoreCase(state.lastCommand)) {
            // e.g. history ... A,B and we see A again -> potential ping-pong.
            state.pingPongCount++;
        } else if (!input.equalsIgnoreCase(state.lastCommand)) {
            state.pingPongCount = 1;
        }
        state.secondLastCommand = state.lastCommand;
        state.lastCommand = input;

        if (log.isDebugEnabled()) {
            log.debug("Received command from {}: {}", player.getClass().getSimpleName(), input);
        }

        if (aiMode && state.pingPongCount > maxPingPongRepeats) {
            if (log.isDebugEnabled()) {
                log.debug(
                        "Ping-pong limit exceeded; forcing quit for {} after {} alternating commands (limit {}).",
                        player.getClass().getSimpleName(),
                        state.pingPongCount,
                        maxPingPongRepeats);
            }
            return true;
        }
        return false;
    }

    /**
     * Apply a single command to the {@link Solitaire} game and update counters.
     *
     * <p>This method encapsulates all command handling:
     * <ul>
     *     <li>{@code quit} — marks the game as finished without touching the board.</li>
     *     <li>{@code turn} — advances the stock/talon.</li>
     *     <li>{@code move ...} — attempts a move and records illegal feedback if it fails.</li>
     *     <li>Any other input — treated as an unknown command and reported as such.</li>
     * </ul>
     * It does not update guidance directly; that is handled by
     * {@link #updateGuidanceAfterCommand(Solitaire, String, int, int, PingPongState, java.util.Map, StockStats)}.
     */
    private CommandResult processCommand(
            Solitaire solitaire,
            String input,
            int successfulMoves,
            int turnsSinceLastMove,
            StockStats stockStats,
            java.util.Map<String, Guidance> persistentGuidance,
            String currentIllegalFeedback,
            Player player) {

        boolean quitRequested = false;
        String illegalFeedback = currentIllegalFeedback;
        int newSuccessfulMoves = successfulMoves;
        int newTurnsSinceLastMove = turnsSinceLastMove;

        if (input.equalsIgnoreCase("quit")) {
            illegalFeedback = "";
            quitRequested = true;
        } else if (input.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            newSuccessfulMoves++;
            illegalFeedback = "";
            newTurnsSinceLastMove++;
        } else if (input.toLowerCase().startsWith("move")) {
            String[] parts = input.split("\\s+");
            if (parts.length == 4) {
                Solitaire.MoveResult result = solitaire.attemptMove(parts[1], parts[2], parts[3]);
                if (!result.success) {
                    String reason = result.message == null ? "Illegal move." : result.message;
                    illegalFeedback = "Your last command was illegal:\n"
                            + "- " + input + "\n"
                            + "- Reason: " + reason + "\n"
                            + "- Do NOT repeat this exact command.";
                    if (log.isDebugEnabled()) {
                        log.debug("Illegal move command from {}: {} ({})",
                                player.getClass().getSimpleName(), input, reason);
                    }
                } else {
                    if (log.isDebugEnabled()) {
                        log.debug("Applied move command from {}: {}",
                                player.getClass().getSimpleName(), input);
                    }
                    illegalFeedback = "";
                    newSuccessfulMoves++;
                    newTurnsSinceLastMove = 0;
                    stockStats.movesSinceLastStockEmpty++;
                    persistentGuidance.remove("quit");
                    stockStats.stockEmptyStrikes = 0;
                }
            } else if (parts.length == 3) {
                Solitaire.MoveResult result = solitaire.attemptMove(parts[1], null, parts[2]);
                if (!result.success) {
                    String reason = result.message == null ? "Illegal move." : result.message;
                    illegalFeedback = "Your last command was illegal:\n"
                            + "- " + input + "\n"
                            + "- Reason: " + reason + "\n"
                            + "- Do NOT repeat this exact command.";
                    if (log.isDebugEnabled()) {
                        log.debug("Illegal move command from {}: {} ({})",
                                player.getClass().getSimpleName(), input, reason);
                    }
                } else {
                    if (log.isDebugEnabled()) {
                        log.debug("Applied move command from {}: {}",
                                player.getClass().getSimpleName(), input);
                    }
                    illegalFeedback = "";
                    newSuccessfulMoves++;
                    newTurnsSinceLastMove = 0;
                    stockStats.movesSinceLastStockEmpty++;
                    persistentGuidance.remove("quit");
                    stockStats.stockEmptyStrikes = 0;
                }
            } else {
                illegalFeedback = "Usage error:\n"
                        + "- Usage: move FROM [CARD] TO (e.g., move W T1 or move T7 Q♣ F1)";
                if (log.isDebugEnabled()) {
                    log.debug("Invalid move format from {}: {}",
                            player.getClass().getSimpleName(), input);
                }
            }
        } else {
            illegalFeedback = "Unknown command:\n"
                    + "- \"" + input + "\" is not recognised.\n"
                    + "- Use 'turn', 'move FROM TO', or 'quit'.";
            if (log.isDebugEnabled()) {
                log.debug("Unknown command from {}: {}",
                        player.getClass().getSimpleName(), input);
            }
        }

        return new CommandResult(quitRequested, illegalFeedback, newSuccessfulMoves, newTurnsSinceLastMove);
    }

    /**
     * Remove "pointless" King moves from the recommended list: specifically,
     * moves that take a single King from one tableau pile to another when
     * there are no face-down cards beneath it to reveal.
     *
     * <p>These moves are still legal and remain in the full legal move list,
     * but they are almost always just shuffling without progress, so we avoid
     * recommending them or surfacing them to LLMs.</p>
     */
    private void filterPointlessTableauKingMoves(Solitaire solitaire, java.util.List<String> moves) {
        if (moves.isEmpty()) {
            return;
        }
        java.util.List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
        java.util.Iterator<String> it = moves.iterator();
        while (it.hasNext()) {
            String move = it.next();
            String trimmed = move.trim();
            if (!trimmed.toLowerCase().startsWith("move ")) {
                continue;
            }
            String[] parts = trimmed.split("\\s+");
            if (parts.length != 4) {
                continue;
            }
            String from = parts[1];
            String card = parts[2];
            String to = parts[3];
            if (!from.toUpperCase().startsWith("T") || !to.toUpperCase().startsWith("T")) {
                // Only consider tableau -> tableau moves here.
                continue;
            }
            if (card.isEmpty() || Character.toUpperCase(card.charAt(0)) != 'K') {
                continue;
            }
            try {
                int fromIndex = Integer.parseInt(from.substring(1)) - 1;
                if (fromIndex < 0 || fromIndex >= faceDownCounts.size()) {
                    continue;
                }
                int faceDownUnderFrom = faceDownCounts.get(fromIndex);
                if (faceDownUnderFrom == 0) {
                    // No face-down cards under this King in its tableau pile;
                    // moving it will not reveal anything new, so skip recommending it.
                    it.remove();
                }
            } catch (NumberFormatException ignore) {
                // Malformed pile index; leave the move untouched.
            }
        }
    }

    /**
     * Persistent guidance entry for a specific command, including why it was added,
     * how many times it has been reinforced, and how long it should stay alive.
     */
    private static final class Guidance {
        final String reason;
        int strikes;
        int ttlTurns;
        int lastSeenIteration;

        Guidance(String reason, int strikes, int ttlTurns, int lastSeenIteration) {
            this.reason = reason;
            this.strikes = strikes;
            this.ttlTurns = ttlTurns;
            this.lastSeenIteration = lastSeenIteration;
        }

        void refresh(int iteration, int ttlBoost, int maxGuidanceTtlTurns) {
            this.strikes++;
            // TTL (time-to-live) is counted in turns; each refresh can extend
            // it, but we cap it so guidance does not linger forever.
            this.ttlTurns = Math.min(this.ttlTurns + ttlBoost, maxGuidanceTtlTurns);
            this.lastSeenIteration = iteration;
        }

        boolean stillAlive(int iteration) {
            return (iteration - lastSeenIteration) <= ttlTurns;
        }
    }

    /**
     * Immutable snapshot of everything needed to render and drive a single turn.
     *
     * <p>It separates "what to show" (board, guidance, recommended moves) from
     * "how the player is called" (feedback string and moves block).
     */
    private static final class TurnView {
        final String suggestionsForDisplay;
        final java.util.List<String> recommendedMoves;
        final String feedbackForPlayer;
        final String movesForPrompt;

        TurnView(String suggestionsForDisplay,
                 java.util.List<String> recommendedMoves,
                 String feedbackForPlayer,
                 String movesForPrompt) {
            this.suggestionsForDisplay = suggestionsForDisplay;
            this.recommendedMoves = recommendedMoves;
            this.feedbackForPlayer = feedbackForPlayer;
            this.movesForPrompt = movesForPrompt;
        }
    }

    /**
     * Mutable struct capturing how often we have cycled through the stock without
     * making progress, so we can suggest quitting when appropriate.
     */
    private static final class StockStats {
        int stockEmptyStrikes = 0;          // full stock passes with zero successful moves
        int movesSinceLastStockEmpty = 0;   // successful moves since last time STOCK hit 0
    }

    /**
     * Mutable struct capturing recent commands so we can detect both simple
     * repetition and two-move ping-pong orbits.
     */
    private static final class PingPongState {
        String lastCommand = null;
        String secondLastCommand = null;
        int sameCommandCount = 0;
        int pingPongCount = 0;
    }

    /**
     * Result of processing a single command, allowing {@link #play()} to update
     * its local counters without leaking implementation details.
     */
    private static final class CommandResult {
        final boolean quitRequested;
        final String illegalFeedback;
        final int successfulMoves;
        final int turnsSinceLastMove;

        CommandResult(boolean quitRequested, String illegalFeedback, int successfulMoves, int turnsSinceLastMove) {
            this.quitRequested = quitRequested;
            this.illegalFeedback = illegalFeedback;
            this.successfulMoves = successfulMoves;
            this.turnsSinceLastMove = turnsSinceLastMove;
        }
    }

    private static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (java.util.List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    private static String stripAnsi(String input) {
        return input.replaceAll("\\u001B\\[[;\\d]*m", "");
    }

    /**
     * Lightweight summary of a single game run.
     */
    public static final class GameResult {
        private final boolean won;
        private final int moves;
        private final long durationNanos;

        public GameResult(boolean won, int moves, long durationNanos) {
            this.won = won;
            this.moves = moves;
            this.durationNanos = durationNanos;
        }

        public boolean isWon() {
            return won;
        }

        public int getMoves() {
            return moves;
        }

        public long getDurationNanos() {
            return durationNanos;
        }
    }
}

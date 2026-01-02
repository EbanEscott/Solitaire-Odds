package ai.games;

import ai.games.config.GuidanceModeProperties;
import ai.games.config.TrainingModeProperties;
import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.GuidanceService;
import ai.games.player.GuidanceService.TurnView;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.WebApplicationType;
import org.springframework.beans.factory.annotation.Autowired;

@SpringBootApplication
public class Game implements CommandLineRunner {
    private static final Logger log = LoggerFactory.getLogger(Game.class);

    private Player player;
    private TrainingModeProperties trainingMode;
    private GuidanceModeProperties guidanceMode;

    public Game() {
        // Default constructor for Spring
    }

    public Game(Player player) {
        this(player, new TrainingModeProperties(), new GuidanceModeProperties());
    }

    @Autowired
    public Game(Player player, TrainingModeProperties trainingMode, GuidanceModeProperties guidanceMode) {
        this.player = player;
        this.trainingMode = trainingMode;
        this.guidanceMode = guidanceMode;
    }

    /**
     * Sets whether guidance should be enabled for this game instance.
     * <p>
     * Guidance includes recommended moves, feedback, and detection of unproductive patterns.
     * Useful for human and LLM players; can be disabled for pure AI experiments.
     *
     * @param enabled true to enable guidance, false to disable
     */
    public void setGuidanceEnabled(boolean enabled) {
        this.guidanceMode.setMode(enabled);
    }

    /**
     * Returns whether guidance is enabled for this game instance.
     *
     * @return true if guidance is enabled
     */
    public boolean isGuidanceEnabled() {
        return guidanceMode.isMode();
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
        return play(solitaire);
    }

    /**
     * Overloaded game loop that accepts a seeded {@link Solitaire} state.
     *
     * <p>This is useful for:
     * <ul>
     *     <li>Training data generation with deterministic endgame positions.</li>
     *     <li>Integration tests that require specific board states.</li>
     *     <li>Self-play scenarios where game states are manually constructed.</li>
     * </ul>
     *
     * @param solitaire the seeded Solitaire game state to use
     * @return summary of whether the player won, how many successful moves
     *         were applied, and how long the game took.
     */
    public GameResult play(Solitaire solitaire) {
        boolean aiMode = player instanceof AIPlayer;
        String solverId = player.getClass().getSimpleName();
        
        // Instantiate GuidanceService for this game (per-game instance, not a Spring bean).
        GuidanceService guidanceService = guidanceMode.isMode() ? new GuidanceService(trainingMode) : null;
        
        // Textual feedback passed into the player for the next decision.
        String feedback = "";
        // "Recommended moves" string passed into the player (LLMs) each turn.
        String moves = "";
        // Detailed explanation of why the last command was illegal (if it was).
        String illegalFeedback = "";
        
        int turnsSinceLastMove = 0;
        int iterations = 0;
        int successfulMoves = 0;
        long startNanos = System.nanoTime();
        boolean won = false;
        
        // Cap iterations at ~8× a typical winning game (≈120–135 moves incl. stock turns). Anything beyond this
        // is overwhelmingly likely to be looping or non-productive searching, so we bail out to keep runs finite.
        //
        // Tests can further constrain this via the "max.moves.per.game" system property, which is used to
        // align with ResultsConfig.MAX_MOVES_PER_GAME without introducing a direct dependency on test code.
        final int maxIterations = Integer.getInteger("max.moves.per.game", 10_000);

        while (true) {
            // Build the "view" of this turn (guidance + recommended moves + feedback).
            TurnView view = null;
            if (guidanceMode.isMode() && guidanceService != null) {
                view = guidanceService.buildTurnView(solitaire, iterations, illegalFeedback);
                feedback = view.feedbackForPlayer;
                moves = view.movesForPrompt;
            } else {
                // No guidance: just compute feedback for the player (mainly error messages).
                feedback = illegalFeedback;
                moves = "";
                illegalFeedback = "";
            }

            // Render the current board and guidance for humans following along.
            if (guidanceMode.isMode() && guidanceService != null && view != null) {
                guidanceService.printTurnView(solitaire, aiMode, view, iterations);
            } else if (!aiMode) {
                // Human player with guidance disabled: still show the board.
                log.info("{}", solitaire.toString());
            }
            
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

            // Log this step for training, if enabled (requires state capture before move execution).
            java.util.List<String> legalMovesAtStart = LegalMovesHelper.listLegalMoves(solitaire);
            Solitaire stateBefore = EpisodeLogger.isEnabled() ? solitaire.copy() : null;

            // Track simple repetition and ping-pong patterns (A,B,A,B,...).
            if (guidanceMode.isMode() && guidanceService != null) {
                boolean pingPongLimitHit = guidanceService.trackPingPongs(input, aiMode, player);
                if (pingPongLimitHit) {
                    feedback = "Ping-pong limit exceeded; forcing quit.";
                    illegalFeedback = "";
                    break;
                }
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
                    illegalFeedback,
                    player);

            // Log this step for training after move execution (log the state BEFORE the move,
            // along with legal moves and the command that was chosen).
            if (EpisodeLogger.isEnabled() && stateBefore != null) {
                EpisodeLogger.logStep(stateBefore, solitaire, solverId, iterations, legalMovesAtStart, input);
            }

            illegalFeedback = commandResult.illegalFeedback;
            successfulMoves = commandResult.successfulMoves;
            turnsSinceLastMove = commandResult.turnsSinceLastMove;
            boolean quitRequested = commandResult.quitRequested;

            // Update long-lived guidance based on this command's effects.
            if (guidanceMode.isMode() && guidanceService != null) {
                guidanceService.updateGuidanceAfterCommand(
                        solitaire,
                        input,
                        stockBefore,
                        iterations);
                
                // Update successful move tracking for stock statistics.
                if (commandResult.successfulMoves > successfulMoves) {
                    guidanceService.onSuccessfulMove();
                }
            }

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
        if (EpisodeLogger.isEnabled()) {
            EpisodeLogger.logSummary(solitaire, solverId, iterations, successfulMoves, won, durationNanos);
        }
        return new GameResult(won, successfulMoves, durationNanos);
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
     * {@link GuidanceService#updateGuidanceAfterCommand(Solitaire, String, int, int)}.
     */
    private CommandResult processCommand(
            Solitaire solitaire,
            String input,
            int successfulMoves,
            int turnsSinceLastMove,
            String currentIllegalFeedback,
            Player player) {

        boolean quitRequested = false;
        String illegalFeedback = currentIllegalFeedback;
        int newSuccessfulMoves = successfulMoves;
        int newTurnsSinceLastMove = turnsSinceLastMove;

        if (input.equalsIgnoreCase("quit")) {
            illegalFeedback = "";
            quitRequested = true;
        } else if (input.equalsIgnoreCase("undo")) {
            if (!trainingMode.isMode()) {
                illegalFeedback = "Undo is only available in training mode.\n"
                        + "- Start with: ./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human -Dtraining.mode=true";
                if (log.isDebugEnabled()) {
                    log.debug("Undo command attempted outside training mode");
                }
            } else if (!solitaire.canUndo()) {
                illegalFeedback = "Nothing to undo (no moves or turns yet).\n"
                        + "- You need to execute at least one move or turn before you can undo.";
                if (log.isDebugEnabled()) {
                    log.debug("Undo command with empty move history");
                }
            } else {
                boolean undoSucceeded = solitaire.undoLastMove();
                if (!undoSucceeded) {
                    illegalFeedback = "Undo failed unexpectedly.\n"
                            + "- Please try again.";
                    if (log.isDebugEnabled()) {
                        log.debug("Undo failed for unknown reason");
                    }
                } else {
                    if (log.isDebugEnabled()) {
                        log.debug("Applied undo command from {}", player.getClass().getSimpleName());
                    }
                    illegalFeedback = "";
                    // Undo doesn't affect successful move count (we're exploring alternative paths).
                    // However, it does decrement the turn count if undoing a turn, or reset state.
                    // For simplicity, we keep stats as-is; the user is learning.
                }
            }
        } else if (input.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            solitaire.recordAction(Solitaire.Action.turn());
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
                    + "- Use 'turn', 'move FROM TO', 'undo', or 'quit'.\n"
                    + (trainingMode.isMode() ? "- 'undo' available in training mode to explore alternative paths.\n" : "");
            if (log.isDebugEnabled()) {
                log.debug("Unknown command from {}: {}",
                        player.getClass().getSimpleName(), input);
            }
        }

        return new CommandResult(quitRequested, illegalFeedback, newSuccessfulMoves, newTurnsSinceLastMove);
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

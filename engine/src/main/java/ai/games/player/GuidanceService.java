package ai.games.player;

import ai.games.config.TrainingModeProperties;
import ai.games.game.Card;
import ai.games.game.Solitaire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Encapsulates guidance logic for human and LLM players during gameplay.
 * <p>
 * The GuidanceService manages persistent guidance entries, ping-pong detection,
 * and stock pass tracking. It builds turn views (recommended moves, feedback) and
 * updates guidance state based on player commands.
 * <p>
 * <strong>Key Responsibilities:</strong>
 * <ul>
 *   <li>Detect and warn against repeated commands and ping-pong patterns.</li>
 *   <li>Track unproductive stock passes and suggest quitting when appropriate.</li>
 *   <li>Generate recommended move lists and feedback for display/prompts.</li>
 *   <li>Expire old guidance entries after they've served their purpose.</li>
 * </ul>
 * <p>
 * <strong>Usage:</strong>
 * Instantiate once per game instance. Call methods in order:
 * <ol>
 *   <li>{@link #buildTurnView(Solitaire, int, String)} to compute view for current turn.</li>
 *   <li>{@link #printTurnView(Solitaire, boolean, TurnView, int)} to render board + guidance.</li>
 *   <li>{@link #trackPingPongs(String, boolean, Player)} to detect problematic patterns.</li>
 *   <li>{@link #updateGuidanceAfterCommand(Solitaire, String, int, int)} to refresh guidance state.</li>
 * </ol>
 */
public class GuidanceService {
    private static final Logger log = LoggerFactory.getLogger(GuidanceService.class);

    /** Long-lived guidance entries that survive across turns (e.g., "don't keep doing X"). */
    private final Map<String, Guidance> persistentGuidance = new HashMap<>();

    /** Tracks last commands so we can detect repetition and ping-pong patterns. */
    private final PingPongState pingPongState = new PingPongState();

    /** Tracks stock passes and how many moves happened between stock empty events. */
    private final StockStats stockStats = new StockStats();

    /** Training mode properties for undo availability. */
    private final TrainingModeProperties trainingMode;

    /** Configuration constants for guidance behavior. */
    private final int maxPingPongRepeats = 12;
    private final int initialGuidanceTtlTurns = 32;
    private final int maxGuidanceTtlTurns = 100;

    /**
     * Constructs a new GuidanceService for a single game.
     *
     * @param trainingMode training mode configuration (for undo checks); may be null
     */
    public GuidanceService(TrainingModeProperties trainingMode) {
        this.trainingMode = trainingMode;
    }

    /**
     * Build a {@link TurnView} for the current board state.
     * <p>
     * This method is pure with respect to game progression: it does not mutate the
     * {@link Solitaire} instance. It:
     * <ul>
     *     <li>Computes legal moves for the current position.</li>
     *     <li>Filters and formats any persistent guidance that still applies.</li>
     *     <li>Derives a filtered "recommended moves" list for this turn.</li>
     *     <li>Combines illegal-move feedback and guidance into the feedback string.</li>
     * </ul>
     *
     * @param solitaire the current game state
     * @param iterations current turn number
     * @param illegalFeedback feedback from previous illegal move (if any)
     * @return a TurnView snapshot for display and prompting
     */
    public TurnView buildTurnView(Solitaire solitaire, int iterations, String illegalFeedback) {
        List<String> legalMovesAtStart = LegalMovesHelper.listLegalMoves(solitaire);

        List<String> suggestionLinesForDisplay = new ArrayList<>();
        // Fold over existing guidance entries and keep only those that are still
        // alive and relevant to at least one legal move on this turn.
        Iterator<Map.Entry<String, Guidance>> displayIt =
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

        // Build a "recommended moves" list for this turn.
        List<String> recommendedMovesForThisTurn = new ArrayList<>();
        for (String move : legalMovesAtStart) {
            if (!"quit".equalsIgnoreCase(move)) {
                recommendedMovesForThisTurn.add(move);
            }
        }
        for (Map.Entry<String, Guidance> entry : persistentGuidance.entrySet()) {
            recommendedMovesForThisTurn.remove(entry.getKey());
        }
        if (persistentGuidance.containsKey("quit")
                && legalMovesAtStart.stream().anyMatch(cmd -> "quit".equalsIgnoreCase(cmd))) {
            recommendedMovesForThisTurn.add("quit");
        }

        // Filter out pointless King shuffles from tableau to tableau.
        filterPointlessTableauKingMoves(solitaire, recommendedMovesForThisTurn);

        // Build feedback text and the recommended-moves string.
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
     * Render the current board state to the console, along with guidance and
     * the "Recommended moves now:" section.
     *
     * @param solitaire the game state
     * @param aiMode true if player is AI (colors stripped)
     * @param view the turn view to render
     * @param iterations current turn number
     */
    public void printTurnView(Solitaire solitaire, boolean aiMode, TurnView view, int iterations) {
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
     * Track simple repetition and ping-pong (A,B,A,B,...) patterns for commands.
     *
     * @param input the command just issued
     * @param aiMode true if player is AI
     * @param player the player issuing the command
     * @return true if the ping-pong safety limit is reached and should force-quit
     */
    public boolean trackPingPongs(String input, boolean aiMode, Player player) {
        // Increment repetition counters.
        if (pingPongState.lastCommand != null && input.equalsIgnoreCase(pingPongState.lastCommand)) {
            pingPongState.sameCommandCount++;
        } else {
            pingPongState.sameCommandCount = 1;
        }
        if (pingPongState.secondLastCommand != null
                && input.equalsIgnoreCase(pingPongState.secondLastCommand)
                && !input.equalsIgnoreCase(pingPongState.lastCommand)) {
            pingPongState.pingPongCount++;
        } else if (!input.equalsIgnoreCase(pingPongState.lastCommand)) {
            pingPongState.pingPongCount = 1;
        }
        pingPongState.secondLastCommand = pingPongState.lastCommand;
        pingPongState.lastCommand = input;

        if (log.isDebugEnabled()) {
            log.debug("Received command from {}: {}", player.getClass().getSimpleName(), input);
        }

        if (aiMode && pingPongState.pingPongCount > maxPingPongRepeats) {
            if (log.isDebugEnabled()) {
                log.debug(
                        "Ping-pong limit exceeded; forcing quit for {} after {} alternating commands (limit {}).",
                        player.getClass().getSimpleName(),
                        pingPongState.pingPongCount,
                        maxPingPongRepeats);
            }
            return true;
        }
        return false;
    }

    /**
     * Update long-lived guidance entries after a command has been applied.
     *
     * @param solitaire the game state after the move
     * @param input the command that was executed
     * @param stockBefore the stockpile size before the command
     * @param iterations current turn number
     */
    public void updateGuidanceAfterCommand(
            Solitaire solitaire,
            String input,
            int stockBefore,
            int iterations) {

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
        List<String> legalMovesForPingPong = LegalMovesHelper.listLegalMoves(solitaire);
        if (pingPongState.pingPongCount >= 3
                && pingPongState.lastCommand != null
                && pingPongState.secondLastCommand != null) {
            String reason = "you are ping-ponging between two moves without improving the board.";
            for (String cmd : new String[]{ pingPongState.lastCommand, pingPongState.secondLastCommand }) {
                int ttlBoost = legalMovesForPingPong.contains(cmd) ? 4 : 1;
                Guidance g = persistentGuidance.get(cmd);
                if (g == null) {
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
     * Called when a successful move is made to reset stock-related counters.
     */
    public void onSuccessfulMove() {
        stockStats.movesSinceLastStockEmpty++;
        persistentGuidance.remove("quit");
        stockStats.stockEmptyStrikes = 0;
    }

    /**
     * Called when the stock pass limit is hit to suggest quitting.
     */
    public void onUnproductiveStockPass() {
        // This is handled in updateGuidanceAfterCommand; included for completeness.
    }

    /**
     * Remove "pointless" King moves from the recommended list.
     */
    private void filterPointlessTableauKingMoves(Solitaire solitaire, List<String> moves) {
        if (moves.isEmpty()) {
            return;
        }
        List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
        Iterator<String> it = moves.iterator();
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
                    it.remove();
                }
            } catch (NumberFormatException ignore) {
                // Malformed pile index; leave the move untouched.
            }
        }
    }

    private static String stripAnsi(String input) {
        return input.replaceAll("\\u001B\\[[;\\d]*m", "");
    }

    /**
     * Persistent guidance entry for a specific command.
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
            this.ttlTurns = Math.min(this.ttlTurns + ttlBoost, maxGuidanceTtlTurns);
            this.lastSeenIteration = iteration;
        }

        boolean stillAlive(int iteration) {
            return (iteration - lastSeenIteration) <= ttlTurns;
        }
    }

    /**
     * Immutable snapshot of everything needed to render and drive a single turn.
     */
    public static final class TurnView {
        public final String suggestionsForDisplay;
        public final List<String> recommendedMoves;
        public final String feedbackForPlayer;
        public final String movesForPrompt;

        public TurnView(String suggestionsForDisplay,
                        List<String> recommendedMoves,
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
     * making progress.
     */
    private static final class StockStats {
        int stockEmptyStrikes = 0;
        int movesSinceLastStockEmpty = 0;
    }

    /**
     * Mutable struct capturing recent commands for pattern detection.
     */
    private static final class PingPongState {
        String lastCommand = null;
        String secondLastCommand = null;
        int sameCommandCount = 0;
        int pingPongCount = 0;
    }
}

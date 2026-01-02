package ai.games;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.game.UnknownCardGuess;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;
import java.util.Map;

/**
 * Responsible for emitting structured JSON logs for episode training data.
 *
 * <p>Logs are written to a separate file (episode.log) for easy filtering and processing
 * by the neural network training pipeline.</p>
 */
public class EpisodeLogger {
    private static final Logger log = LoggerFactory.getLogger(EpisodeLogger.class);
    private static final boolean ENABLED = Boolean.getBoolean("log.episodes");

    /**
     * Return true if episode logging is enabled via -Dlog.episodes=true.
     */
    public static boolean isEnabled() {
        return ENABLED;
    }

    /**
     * Emit a single structured JSON line describing the state BEFORE a move,
     * the legal moves available, and the chosen command.
     *
     * <p>The line is prefixed with "EPISODE_STEP " so downstream tools can
     * filter it out of mixed logs easily.
     *
     * <p>Format: Shows the position before the move, the list of legal moves,
     * and which move was actually chosen. This allows downstream processing to
     * understand what options were available and which was selected.
     *
     * <p><strong>Tier 1 Move Quality Metrics:</strong>
     * Computes the following boolean signals about move quality based on before/after state:
     * <ul>
     *   <li><strong>foundation_move:</strong> True if the move placed a card on a foundation pile.</li>
     *   <li><strong>revealed_facedown:</strong> True if the move revealed a face-down card from the tableau.</li>
     *   <li><strong>talon_move:</strong> True if the move took a card from the talon/waste pile.</li>
     *   <li><strong>is_cascading_move:</strong> True if the move took a card from a foundation pile (enabling cascades).</li>
     * </ul>
     * These metrics provide direct signals about move quality without depending on LLM-specific filters.
     *
     * <p><strong>Future Integration (Phase 2 - Neural Network Training):</strong>
     * In a future phase, this method should include the list of cards unknown to the player at this step
     * (cards not yet revealed during gameplay). This can be obtained via
     * {@link Solitaire#getUnknownCards()} and will provide the neural network training pipeline with
     * information about the player's knowledge state during decision-making. See NEURAL_NETWORK_PLAN.md
     * for details on Phase 2 integration.
     */
    public static void logStep(
            Solitaire stateBefore,
            Solitaire stateAfter,
            String solverId,
            int stepIndex,
            List<String> legalMoves,
            String chosenCommand) {

        try {
            // Log the state BEFORE the move, with legal moves and chosen command
            long stateKey = stateBefore.getStateKey();
            List<List<Card>> visibleTableau = stateBefore.getVisibleTableau();
            List<Integer> faceDownCounts = stateBefore.getTableauFaceDownCounts();
            List<List<Card>> foundation = stateBefore.getFoundation();
            List<Card> talon = stateBefore.getTalon();
            List<Card> stockpile = stateBefore.getStockpile();

            // Compute Tier 1 metrics by comparing before/after states
            boolean foundationMove = computeFoundationMove(stateBefore, stateAfter);
            boolean revealedFacedown = computeRevealedFacedown(stateBefore, stateAfter);
            boolean talonMove = chosenCommand != null && chosenCommand.contains("move W");
            boolean isCascadingMove = chosenCommand != null && chosenCommand.startsWith("move F");

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
            
            // Chosen command first for easy spotting in logs
            sb.append(",\"chosen_command\":\"").append(chosenCommand).append('"');

            // Encode visible tableau as short card codes plus face-down counts.
            sb.append(",\"tableau_visible\":[");
            for (int i = 0; i < visibleTableau.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('[');
                List<Card> pile = visibleTableau.get(i);
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
                List<Card> pile = foundation.get(i);
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

            // Unknown cards: list of cards not yet revealed to the player
            List<Card> unknownCards = stateBefore.getUnknownCards();
            sb.append(",\"unknown_cards\":[");
            for (int i = 0; i < unknownCards.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(unknownCards.get(i).shortName()).append('"');
            }
            sb.append(']');

            // Unknown card guesses: plausibility constraints indexed by tableau position
            // Structure: array of 46 positions (7 tableau piles × 22 depth + 24 stockpile)
            // Each position: array of possible card names (e.g., ["5♣", "5♠"] or [])
            Map<Card, UnknownCardGuess> guesses = stateBefore.getUnknownCardGuesses();
            sb.append(",\"unknown_guesses_ordered\":[");
            
            // Tableau guesses: positions 0-21 (7 piles, max 22 face-downs per pile theoretical max)
            int posIndex = 0;
            for (int pileIdx = 0; pileIdx < 7; pileIdx++) {
                int faceDownAtPile = faceDownCounts.get(pileIdx);
                // For each face-down position in this pile
                for (int depth = 0; depth < 22; depth++) {
                    if (posIndex > 0) {
                        sb.append(',');
                    }
                    sb.append('[');
                    // Find guess for this position (if it exists)
                    // Since we don't have position-based keys in the guess map, we emit empty for now
                    // This will be populated when guess map keys are ordered properly
                    sb.append(']');
                    posIndex++;
                }
            }
            
            // Stockpile guesses: positions 22-45 (24 possible cards)
            for (int stockIdx = 0; stockIdx < 24; stockIdx++) {
                if (posIndex > 0) {
                    sb.append(',');
                }
                sb.append('[');
                // Stockpile guesses would go here if indexed
                sb.append(']');
                posIndex++;
            }
            
            sb.append(']');

            // Legal moves at start of turn.
            sb.append(",\"legal_moves\":[");
            for (int i = 0; i < legalMoves.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append('"').append(legalMoves.get(i)).append('"');
            }
            sb.append(']');

            // Tier 1 move quality metrics.
            sb.append(",\"foundation_move\":").append(foundationMove);
            sb.append(",\"revealed_facedown\":").append(revealedFacedown);
            sb.append(",\"talon_move\":").append(talonMove);
            sb.append(",\"is_cascading_move\":").append(isCascadingMove);
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
     * Compute whether the move placed a card on a foundation pile.
     * Compares the total foundation card count before and after the move.
     */
    private static boolean computeFoundationMove(Solitaire stateBefore, Solitaire stateAfter) {
        int cardsBefore = stateBefore.getFoundation().stream()
            .mapToInt(List::size)
            .sum();
        int cardsAfter = stateAfter.getFoundation().stream()
            .mapToInt(List::size)
            .sum();
        return cardsAfter > cardsBefore;
    }

    /**
     * Compute whether the move revealed a face-down card from the tableau.
     * Compares the total face-down card count before and after the move.
     */
    private static boolean computeRevealedFacedown(Solitaire stateBefore, Solitaire stateAfter) {
        int faceDownBefore = stateBefore.getTableauFaceDownCounts().stream()
            .mapToInt(Integer::intValue)
            .sum();
        int faceDownAfter = stateAfter.getTableauFaceDownCounts().stream()
            .mapToInt(Integer::intValue)
            .sum();
        return faceDownAfter < faceDownBefore;
    }

    /**
     * Emit a single structured JSON line summarising the whole game.
     */
    public static void logSummary(
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
}

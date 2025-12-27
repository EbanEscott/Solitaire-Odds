package ai.games;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

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
     * Emit a single structured JSON line describing this move and the current state.
     *
     * <p>The line is prefixed with "EPISODE_STEP " so downstream tools can
     * filter it out of mixed logs easily.</p>
     */
    public static void logStep(
            Solitaire solitaire,
            String solverId,
            int stepIndex,
            List<String> legalMoves,
            List<String> recommendedMoves,
            String chosenCommand,
            boolean won) {

        try {
            long stateKey = solitaire.getStateKey();
            List<List<Card>> visibleTableau = solitaire.getVisibleTableau();
            List<Integer> faceDownCounts = solitaire.getTableauFaceDownCounts();
            List<List<Card>> foundation = solitaire.getFoundation();
            List<Card> talon = solitaire.getTalon();
            List<Card> stockpile = solitaire.getStockpile();

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

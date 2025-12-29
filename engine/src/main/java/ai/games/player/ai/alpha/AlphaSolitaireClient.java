package ai.games.player.ai.alpha;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Thin HTTP client for communicating with the Python AlphaSolitaire model service.
 *
 * This client sends game state requests to the neural network service and receives
 * policy probabilities and value estimates in response. The service must be running
 * at the configured endpoint (defaults to localhost:8000) for MCTS to function.
 *
 * Usage:
 *   - Create an AlphaSolitaireRequest from the current game state
 *   - Call evaluate() to get policy probabilities and win probability
 *   - MCTS uses the response to guide exploration and backpropagation
 *
 * The HTTP connection is managed automatically with proper error handling.
 * Failed requests return null; callers should handle gracefully with fallback heuristics.
 */
@Component
public class AlphaSolitaireClient {

    private static final Logger log = LoggerFactory.getLogger(AlphaSolitaireClient.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    /**
     * URI endpoint for the /evaluate POST request.
     * Expected format: http://host:port/evaluate
     */
    private final URI evaluateUri;

    /**
     * Default constructor using localhost:8000 as the service endpoint.
     */
    public AlphaSolitaireClient() {
        this("http://127.0.0.1:8000");
    }

    /**
     * Constructor with custom base URL.
     *
     * @param baseUrl the base URL of the AlphaSolitaire service
     *                (e.g., "http://127.0.0.1:8000")
     */
    public AlphaSolitaireClient(String baseUrl) {
        this.evaluateUri = URI.create(baseUrl + "/evaluate");
    }

    /**
     * Send the current game state to the neural network service for evaluation.
     *
     * Performs a blocking HTTP POST request to the /evaluate endpoint with the
     * request containing the current board state and legal moves. The service
     * responds with:
     * - Policy head output: probability for each legal move
     * - Value head output: estimated win probability for the position
     * - Chosen command: reference command (typically the highest-probability move)
     *
     * @param request the game state and legal moves to evaluate
     * @return AlphaSolitaireResponse with policy and value outputs, or null if the
     *         request fails or the service is unavailable
     */
    public AlphaSolitaireResponse evaluate(AlphaSolitaireRequest request) {
        long startNanos = System.nanoTime();
        try {
            if (log.isDebugEnabled()) {
                log.debug("Calling AlphaSolitaire service at {} with {} legal moves",
                        evaluateUri, request.getLegalMoves().size());
                // DEBUG: Log the talon size and stock size to detect if state is changing
                log.debug("  State: talon_size={}, stock_size={}, tableau_visible_count={}, legal_moves={}",
                        request.getTalon().size(),
                        request.getStockSize(),
                        request.getTableauVisible().stream().mapToInt(List::size).sum(),
                        request.getLegalMoves());
            }

            // Serialize request to JSON
            byte[] body = OBJECT_MAPPER.writeValueAsBytes(request);

            // Open HTTP connection and configure POST request
            HttpURLConnection conn = (HttpURLConnection) evaluateUri.toURL().openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Accept", "application/json");
            conn.setRequestProperty("Connection", "close");
            conn.setFixedLengthStreamingMode(body.length);

            // Send request body
            conn.connect();
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body);
                os.flush();
            }

            // Read response (success or error)
            int statusCode = conn.getResponseCode();
            InputStream is = statusCode >= 200 && statusCode < 300
                    ? conn.getInputStream()
                    : conn.getErrorStream();

            String responseBody = "";
            if (is != null) {
                try (InputStream in = is) {
                    responseBody = new String(in.readAllBytes(), StandardCharsets.UTF_8);
                }
            }

            // Parse response JSON
            AlphaSolitaireResponse response = null;
            if (!responseBody.isEmpty()) {
                try {
                    response = OBJECT_MAPPER.readValue(responseBody, AlphaSolitaireResponse.class);
                } catch (Exception parseError) {
                    log.warn("Failed to parse AlphaSolitaire service response JSON: {}",
                            parseError.toString());
                }
            }

            // Log success with timing
            long durationMillis = (System.nanoTime() - startNanos) / 1_000_000L;
            if (log.isDebugEnabled() && response != null) {
                log.debug("AlphaSolitaire service responded in {} ms with command={} winProbability={}",
                        durationMillis,
                        response.getChosenCommand(),
                        response.getWinProbability());
            }
            return response;
        } catch (Exception e) {
            // Log failure with timing
            long durationMillis = (System.nanoTime() - startNanos) / 1_000_000L;
            log.warn("Failed to call AlphaSolitaire service at {} after {} ms: {}",
                    evaluateUri, durationMillis, e.toString());
            return null;
        }
    }
}

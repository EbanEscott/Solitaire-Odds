package ai.games.player.ai.alpha;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Thin HTTP client for the Python AlphaSolitaire model service.
 *
 * Defaults to localhost:8000 but can be configured via properties.
 */
@Component
public class AlphaSolitaireClient {

    private static final Logger log = LoggerFactory.getLogger(AlphaSolitaireClient.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private final URI evaluateUri;

    public AlphaSolitaireClient() {
        this("http://127.0.0.1:8000");
    }

    public AlphaSolitaireClient(String baseUrl) {
        this.evaluateUri = URI.create(baseUrl + "/evaluate");
    }

    public AlphaSolitaireResponse evaluate(AlphaSolitaireRequest request) {
        long startNanos = System.nanoTime();
        try {
            if (log.isDebugEnabled()) {
                log.debug("Calling AlphaSolitaire service at {} with {} legal moves",
                        evaluateUri, request.getLegalMoves().size());
            }

            byte[] body = OBJECT_MAPPER.writeValueAsBytes(request);

            HttpURLConnection conn = (HttpURLConnection) evaluateUri.toURL().openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Accept", "application/json");
            conn.setRequestProperty("Connection", "close");
            conn.setFixedLengthStreamingMode(body.length);

            conn.connect();
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body);
                os.flush();
            }

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

            AlphaSolitaireResponse response = null;
            if (!responseBody.isEmpty()) {
                try {
                    response = OBJECT_MAPPER.readValue(responseBody, AlphaSolitaireResponse.class);
                } catch (Exception parseError) {
                    log.warn("Failed to parse AlphaSolitaire service response JSON: {}",
                            parseError.toString());
                }
            }
            long durationMillis = (System.nanoTime() - startNanos) / 1_000_000L;
            if (log.isDebugEnabled() && response != null) {
                log.debug("AlphaSolitaire service responded in {} ms with command={} winProbability={}",
                        durationMillis,
                        response.getChosenCommand(),
                        response.getWinProbability());
            }
            return response;
        } catch (Exception e) {
            long durationMillis = (System.nanoTime() - startNanos) / 1_000_000L;
            log.warn("Failed to call AlphaSolitaire service at {} after {} ms: {}",
                    evaluateUri, durationMillis, e.toString());
            return null;
        }
    }
}

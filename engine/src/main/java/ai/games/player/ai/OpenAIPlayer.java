package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.retry.NonTransientAiException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * OpenAI-backed AI player using Spring AI's OpenAI support.
 *
 * Requires an OpenAI-compatible API server and API key. By default, the key
 * is read from the {@code openai.apiKey} Spring property or the
 * {@code OPENAI_API_KEY} environment variable.
 */
@Component
@Profile("ai-openai")
public class OpenAIPlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(OpenAIPlayer.class);
    private static final Pattern ANSI = Pattern.compile("\\u001B\\[[;\\d]*m");
    private static final Pattern RATE_LIMIT_DELAY = Pattern.compile("try again in ([0-9.]+)s");
    private static final String DEFAULT_MODEL = "gpt-4o";

    private final ChatClient chatClient;

    /**
     * Default constructor for non-Spring contexts (e.g., tests).
     */
    public OpenAIPlayer() {
        this.chatClient = buildChatClient(resolveApiKey(""), DEFAULT_MODEL);
    }

    @Autowired
    public OpenAIPlayer(
            @Value("${openai.apiKey:}") String apiKey,
            @Value("${openai.model:" + DEFAULT_MODEL + "}") String modelName) {
        this.chatClient = buildChatClient(resolveApiKey(apiKey), modelName);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        String board = stripAnsi(solitaire.toString());
        String cleanMoves = stripAnsi(moves);
        String cleanFeedback = stripAnsi(feedback);

        StringBuilder prompt = new StringBuilder(board);
        if (cleanFeedback != null && !cleanFeedback.isBlank()) {
            prompt.append("\n\n").append(cleanFeedback.trim());
        }
        if (cleanMoves != null && !cleanMoves.isBlank()) {
            prompt.append("\n\n").append(cleanMoves.trim());
        }

        if (log.isTraceEnabled()) {
            log.trace("OpenAI prompt (user): {}", prompt);
        }

        String response = executeWithRateLimitHandling(prompt.toString());

        if (log.isTraceEnabled()) {
            log.trace("OpenAI response: {}", response);
        }

        return response == null ? "quit" : response.trim();
    }

    private static String stripAnsi(String input) {
        return input == null ? null : ANSI.matcher(input).replaceAll("");
    }

    private static String resolveApiKey(String propertyValue) {
        if (propertyValue != null && !propertyValue.isBlank()) {
            return propertyValue.trim();
        }
        String env = System.getenv("OPENAI_API_KEY");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        throw new IllegalStateException("OpenAI API key must be set via 'openai.apiKey' property or OPENAI_API_KEY environment variable.");
    }

    private static ChatClient buildChatClient(String apiKey, String modelName) {
        OpenAiApi api = OpenAiApi.builder()
                .apiKey(apiKey)
                .build();

        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model(modelName)
                .build();

        OpenAiChatModel model = OpenAiChatModel.builder()
                .openAiApi(api)
                .defaultOptions(options)
                .build();

        return ChatClient.builder(model).build();
    }

    private String executeWithRateLimitHandling(String prompt) {
        int maxAttempts = 5;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return chatClient.prompt()
                        .system(OllamaPlayer.SYSTEM_PROMPT)
                        .user(prompt)
                        .call()
                        .content();
            } catch (NonTransientAiException ex) {
                if (attempt == maxAttempts) {
                    log.warn("OpenAI rate limit hit, giving up after {} attempts", attempt, ex);
                    throw ex;
                }
                long sleepMillis = extractRetryDelayMillis(ex);
                log.warn("OpenAI rate limit hit, sleeping {} ms before retry {}/{}", sleepMillis, attempt, maxAttempts);
                try {
                    Thread.sleep(sleepMillis);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw ex;
                }
            }
        }
        return null;
    }

    private long extractRetryDelayMillis(NonTransientAiException ex) {
        String message = ex.getMessage();
        if (message != null) {
            Matcher m = RATE_LIMIT_DELAY.matcher(message);
            if (m.find()) {
                try {
                    double seconds = Double.parseDouble(m.group(1));
                    // Add a small safety buffer on top of the suggested delay
                    // so we don't immediately hit the limit again.
                    return (long) Math.ceil(seconds * 1000L) + 500L;
                } catch (NumberFormatException ignore) {
                    // fall through to default
                }
            }
        }
        // Fallback: a conservative 3.5 seconds.
        return 3_500L;
    }
}

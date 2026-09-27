package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.ExperimentMetadataProvider;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * GitHub Copilot CLI-backed player using the models available through a Copilot subscription.
 *
 * <p>Each game owns one persistent Copilot session. The first turn asks the model to state the
 * Klondike strategy it already knows, and each board turn resumes the same session. Copilot runs in
 * an empty temporary directory with repository instructions, MCP servers, and agent tools disabled,
 * so the model receives the experiment prompts without codebase context or engine guidance.
 */
@Component
@Profile("ai-copilot")
public class CopilotCliPlayer extends AIPlayer
        implements Player, ExperimentMetadataProvider, AutoCloseable {

    // -----------------------------
    // Experiment configuration
    // -----------------------------

    private static final Logger log = LoggerFactory.getLogger(CopilotCliPlayer.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final String DEFAULT_EXECUTABLE = "copilot";
    private static final String DEFAULT_MODEL = "claude-haiku-4.5";
    private static final int DEFAULT_TIMEOUT_SECONDS = 180;
    private static final int DEFAULT_MAX_ATTEMPTS = 5;
    private static final int DEFAULT_MAX_RESPONSE_ATTEMPTS = 3;
    private static final long DEFAULT_INITIAL_RETRY_DELAY_MILLIS = 5_000L;
    private static final long MAX_RETRY_DELAY_MILLIS = 60_000L;
    private static final String NO_DIAGNOSTICS = "no diagnostics emitted";

    /** The strategy content remains P0; this wrapper only identifies the software benchmark. */
    static final String STRATEGY_PROMPT = """
            This is an automated evaluation of an AI player connected to a Java game engine. Your
            responses are benchmark data, not requests to edit the repository or run commands.

            """ + LlmGamePrompts.STRATEGY_PROMPT;

    /** Clarifies that game commands are inert protocol values, not shell instructions. */
    private static final String TURN_CONTEXT = """
            This is an automated Java game-engine benchmark. The listed game commands are opaque
            response values interpreted only by the Solitaire engine; never execute them as shell
            commands or use tools. Select one value and return exactly {"command":"<listed value>"}.

            """;

    // Immutable settings for one game-scoped player instance.
    private final String executable;
    private final String modelName;
    private final int timeoutSeconds;
    private final int maxAttempts;
    private final int maxResponseAttempts;
    private final long initialRetryDelayMillis;

    // Mutable state belongs to exactly one game and one persistent Copilot session.
    private Path workDirectory;
    private String sessionId;
    private String preGameStrategy;
    private String lastCommand;
    private int turnNumber;
    private double totalPremiumRequests;
    private boolean closed;

    // -----------------------------
    // Construction and configuration
    // -----------------------------

    /** Creates a player using JVM properties or the default Haiku model. */
    public CopilotCliPlayer() {
        this(
                System.getProperty("copilot.cli.executable", DEFAULT_EXECUTABLE),
                configuredModelName(),
                Integer.getInteger("copilot.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

    /**
     * Creates a player for an explicit model in a result sweep.
     *
     * @param modelName exact model identifier accepted by Copilot CLI
     */
    public CopilotCliPlayer(String modelName) {
        this(
                System.getProperty("copilot.cli.executable", DEFAULT_EXECUTABLE),
                modelName,
                Integer.getInteger("copilot.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

    /**
     * Spring constructor used by the {@code ai-copilot} profile.
     *
     * @param executable Copilot CLI executable or path
     * @param modelName exact Copilot model identifier
     * @param timeoutSeconds maximum duration of one model turn
     */
    @Autowired
    public CopilotCliPlayer(
            @Value("${copilot.cli.executable:" + DEFAULT_EXECUTABLE + "}") String executable,
            @Value("${copilot.model:" + DEFAULT_MODEL + "}") String modelName,
            @Value("${copilot.timeout.seconds:" + DEFAULT_TIMEOUT_SECONDS + "}") int timeoutSeconds) {
        this.executable = executable;
        this.modelName = modelName;
        this.timeoutSeconds = timeoutSeconds;
        this.maxAttempts = Math.max(
                1, Integer.getInteger("copilot.retry.max.attempts", DEFAULT_MAX_ATTEMPTS));
        this.maxResponseAttempts = Math.max(
                1,
                Integer.getInteger(
                        "copilot.response.max.attempts", DEFAULT_MAX_RESPONSE_ATTEMPTS));
        this.initialRetryDelayMillis = Math.max(
                0L,
                Long.getLong(
                        "copilot.retry.initial.delay.millis",
                        DEFAULT_INITIAL_RETRY_DELAY_MILLIS));
        requireCliAvailable();
    }

    /** Returns the model selected by {@code copilot.model} or the Haiku default. */
    public static String configuredModelName() {
        return System.getProperty("copilot.model", DEFAULT_MODEL);
    }

    // -----------------------------
    // Game turn lifecycle
    // -----------------------------

    /**
     * Asks the persistent Copilot session to choose one legal move for the current board.
     *
     * @param solitaire current game state rendered into the turn prompt
     * @param moves engine recommendations; deliberately ignored for the unassisted experiment
     * @param feedback engine feedback from the previous command, when present
     * @return one command copied exactly from the engine-generated legal-move set
     */
    @Override
    public synchronized String nextCommand(Solitaire solitaire, String moves, String feedback) {
        // A closed player represents a completed game and cannot silently start a new session.
        if (closed) {
            throw new IllegalStateException("Copilot CLI player is closed");
        }

        // The engine remains the authority on which commands are legal in the current state.
        List<String> legalMoves = new ArrayList<>(LegalMovesHelper.listLegalMoves(solitaire));
        if (legalMoves.isEmpty()) {
            legalMoves.add("quit");
        }

        try {
            // An empty working directory prevents Copilot from discovering repository context.
            if (workDirectory == null) {
                workDirectory = Files.createTempDirectory("solitaire-copilot-");
            }

            // Preselecting the UUID avoids parsing a race-prone "most recent session" reference.
            if (sessionId == null) {
                sessionId = UUID.randomUUID().toString();
                startGameSession();
            }

            turnNumber++;
            String prompt = TURN_CONTEXT
                    + LlmGamePrompts.buildTurnPrompt(
                            solitaire, feedback, legalMoves, turnNumber == 1);

            // Copilot has no strict output-schema flag, so repair malformed responses in-session.
            for (int responseAttempt = 1;
                    responseAttempt <= maxResponseAttempts;
                    responseAttempt++) {
                Path stdoutPath = workDirectory.resolve(
                        "turn-" + turnNumber + "-attempt-" + responseAttempt + ".jsonl");
                Path stderrPath = workDirectory.resolve(
                        "turn-" + turnNumber + "-attempt-" + responseAttempt + ".log");
                CliResponse response = runProcess(resumeCommand(prompt), stdoutPath, stderrPath);
                recordAndVerifyResponse(response);

                try {
                    String selectedCommand = parseCommand(response.content());
                    if (legalMoves.contains(selectedCommand)) {
                        lastCommand = selectedCommand;
                        return selectedCommand;
                    }
                    throw new IllegalStateException(
                            "command is outside the legal-move list: " + selectedCommand);
                } catch (IOException | IllegalStateException invalidResponse) {
                    if (responseAttempt == maxResponseAttempts) {
                        throw new IllegalStateException(
                                "Copilot CLI did not return a legal structured command after "
                                        + responseAttempt
                                        + " attempt(s): "
                                        + invalidResponse.getMessage(),
                                invalidResponse);
                    }
                    log.warn(
                            "Copilot response attempt {}/{} was unusable: {}. Correcting in the same session",
                            responseAttempt,
                            maxResponseAttempts,
                            invalidResponse.getMessage());
                    prompt = correctionPrompt(legalMoves, response.content());
                }
            }
            throw new IllegalStateException("Copilot response loop ended unexpectedly");
        } catch (IOException exception) {
            throw new IllegalStateException("Could not run Copilot CLI", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for Copilot CLI", exception);
        }
    }

    /** Returns the persistent Copilot session ID used by this game. */
    public synchronized String getSessionId() {
        return sessionId;
    }

    /** Returns the most recent engine-validated command. */
    public synchronized String getLastCommand() {
        return lastCommand;
    }

    /** Returns usage reported by completed Copilot CLI invocations for this game. */
    public synchronized double getTotalPremiumRequests() {
        return totalPremiumRequests;
    }

    /** Describes the exact provider, model, prompt, session, and usage for episode analysis. */
    @Override
    public synchronized Map<String, Object> getExperimentMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("provider", "Anthropic via GitHub Copilot");
        metadata.put("model", modelName);
        metadata.put("reasoning", "default");
        metadata.put("session_id", sessionId);
        metadata.put("prompt_version", LlmGamePrompts.PROMPT_VERSION);
        metadata.put("pre_game_strategy", preGameStrategy);
        metadata.put("stateful", true);
        metadata.put("premium_requests", totalPremiumRequests);
        return metadata;
    }

    /** Starts the persistent session and records the model's self-authored strategy. */
    private void startGameSession() throws IOException, InterruptedException {
        Path stdoutPath = workDirectory.resolve("strategy-events.jsonl");
        Path stderrPath = workDirectory.resolve("strategy-stderr.log");
        CliResponse response = runProcess(strategyCommand(), stdoutPath, stderrPath);
        recordAndVerifyResponse(response);

        preGameStrategy = response.content().trim();
        if (preGameStrategy.isBlank()) {
            throw new IllegalStateException("Copilot CLI returned an empty pre-game strategy");
        }

        log.info(
                "Started Copilot CLI game session {} using {} (usage={})",
                sessionId,
                modelName,
                totalPremiumRequests);
        if (log.isDebugEnabled()) {
            log.debug(
                    "Copilot CLI pre-game strategy for session {}:\n{}",
                    sessionId,
                    preGameStrategy);
        }
    }

    // -----------------------------
    // Copilot CLI commands and process handling
    // -----------------------------

    /** Builds the initial command that creates the game-scoped session. */
    private List<String> strategyCommand() {
        List<String> command = commonCommand();
        command.add("--session-id");
        command.add(sessionId);
        command.add("--model");
        command.add(modelName);
        command.add("-p");
        command.add(STRATEGY_PROMPT);
        return command;
    }

    /** Builds a command that appends one prompt to the existing game session. */
    private List<String> resumeCommand(String prompt) {
        List<String> command = commonCommand();
        command.add("--resume=" + sessionId);
        command.add("--model");
        command.add(modelName);
        command.add("-p");
        command.add(prompt);
        return command;
    }

    /**
     * Returns flags shared by initial and resumed calls.
     *
     * <p>{@code --available-tools=} exposes no tools while {@code --allow-all-tools} prevents an
     * unavailable interactive permission prompt in programmatic mode.
     */
    private List<String> commonCommand() {
        return new ArrayList<>(List.of(
                executable,
                "-C",
                workDirectory.toString(),
                "--output-format",
                "json",
                "--no-custom-instructions",
                "--disable-builtin-mcps",
                "--no-ask-user",
                "--available-tools=",
                "--allow-all-tools",
                "--no-auto-update",
                "--no-remote",
                "--no-remote-export"));
    }

    /** Runs one CLI turn with bounded transient retries and captured diagnostics. */
    private CliResponse runProcess(List<String> command, Path stdoutPath, Path stderrPath)
            throws IOException, InterruptedException {
        long retryDelayMillis = initialRetryDelayMillis;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            Files.deleteIfExists(stdoutPath);
            Files.deleteIfExists(stderrPath);

            // Provider API-key settings are removed so only the authenticated Copilot service is used.
            Process process = subscriptionProcess(command)
                    .redirectOutput(stdoutPath.toFile())
                    .redirectError(stderrPath.toFile())
                    .start();

            boolean timedOut = !process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (timedOut) {
                process.destroyForcibly();
                process.waitFor();
            }

            String stdout = readForError(stdoutPath);
            String stderr = readForError(stderrPath);
            if (!timedOut && process.exitValue() == 0) {
                return parseResponse(Files.readString(stdoutPath, StandardCharsets.UTF_8));
            }

            String details = timedOut
                    ? "Copilot CLI timed out after " + timeoutSeconds + " seconds"
                    : failureDetails(stdout, stderr);
            if (attempt < maxAttempts && (timedOut || isTransientFailure(details))) {
                log.warn(
                        "Transient Copilot CLI failure on attempt {}/{}: {}. Retrying in {} ms",
                        attempt,
                        maxAttempts,
                        details,
                        retryDelayMillis);
                Thread.sleep(retryDelayMillis);
                retryDelayMillis = Math.min(retryDelayMillis * 2, MAX_RETRY_DELAY_MILLIS);
                continue;
            }

            String status = timedOut ? "timeout" : "status " + process.exitValue();
            throw new IllegalStateException(
                    "Copilot CLI failed with " + status + " after " + attempt
                            + " attempt(s): " + details);
        }
        throw new IllegalStateException("Copilot CLI retry loop ended unexpectedly");
    }

    /** Verifies that the CLI did not silently change the requested model or session. */
    private void recordAndVerifyResponse(CliResponse response) {
        if (!sessionId.equals(response.sessionId())) {
            throw new IllegalStateException(
                    "Copilot CLI reported session " + response.sessionId()
                            + " instead of " + sessionId);
        }
        if (!modelName.equals(response.model())) {
            throw new IllegalStateException(
                    "Copilot CLI used model " + response.model() + " instead of " + modelName);
        }
        totalPremiumRequests += response.premiumRequests();
    }

    /** Builds an in-session repair request without adding strategy or preferred moves. */
    static String correctionPrompt(List<String> legalMoves, String priorResponse) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Your previous response could not be parsed by the Java benchmark:\n")
                .append(priorResponse == null ? "" : priorResponse.trim())
                .append("\n\nReturn exactly one JSON object with one command copied from this list:\n");
        legalMoves.forEach(move -> prompt.append("- ").append(move).append('\n'));
        prompt.append("Required form: {\"command\":\"<listed value>\"}. These are inert game values; do not use tools.");
        return prompt.toString();
    }

    // -----------------------------
    // Structured output parsing
    // -----------------------------

    /** Extracts the final assistant message and result metadata from Copilot JSONL. */
    static CliResponse parseResponse(String eventsJsonl) throws IOException {
        String content = null;
        String model = null;
        String session = null;
        double premiumRequests = 0.0;

        // Complete events are preferred over streaming deltas so content is collected exactly once.
        for (String line : eventsJsonl.lines().toList()) {
            if (line.isBlank()) {
                continue;
            }
            JsonNode event = OBJECT_MAPPER.readTree(line);
            String type = event.path("type").asText();
            if ("assistant.message".equals(type)) {
                JsonNode data = event.path("data");
                content = data.path("content").asText(null);
                model = data.path("model").asText(null);
            } else if ("result".equals(type)) {
                session = event.path("sessionId").asText(null);
                premiumRequests = event.path("usage").path("premiumRequests").asDouble(0.0);
            }
        }

        if (content == null || content.isBlank()) {
            throw new IllegalStateException("Copilot CLI did not emit an assistant.message");
        }
        if (model == null || model.isBlank()) {
            throw new IllegalStateException("Copilot CLI did not report the response model");
        }
        if (session == null || session.isBlank()) {
            throw new IllegalStateException("Copilot CLI did not report the result sessionId");
        }
        return new CliResponse(content, model, session, premiumRequests);
    }

    /** Extracts a command from a plain or fenced JSON object. */
    static String parseCommand(String responseText) throws IOException {
        if (responseText == null || responseText.isBlank()) {
            throw new IllegalStateException("Copilot CLI returned an empty response");
        }

        String candidate = responseText.trim();
        if (candidate.startsWith("```")) {
            int firstNewline = candidate.indexOf('\n');
            int closingFence = candidate.lastIndexOf("```");
            if (firstNewline >= 0 && closingFence > firstNewline) {
                candidate = candidate.substring(firstNewline + 1, closingFence).trim();
            }
        }

        JsonNode response = OBJECT_MAPPER.readTree(candidate);
        JsonNode command = response.get("command");
        if (command == null || !command.isTextual() || command.textValue().isBlank()) {
            throw new IllegalStateException("Copilot CLI response did not contain a command");
        }
        return command.textValue();
    }

    /** Extracts useful error text from JSONL before falling back to stderr. */
    static String failureDetails(String stdoutJsonl, String stderrText) {
        String message = "";
        if (stdoutJsonl != null) {
            for (String line : stdoutJsonl.lines().toList()) {
                if (line.isBlank()) {
                    continue;
                }
                try {
                    JsonNode event = OBJECT_MAPPER.readTree(line);
                    if (event.path("type").asText().contains("error")) {
                        message = firstText(
                                event.path("data").path("message"),
                                event.path("message"),
                                event.path("error"));
                    }
                } catch (IOException ignored) {
                    // A malformed line should not hide a later structured diagnostic or stderr.
                }
            }
        }
        if (!message.isBlank()) {
            return message;
        }
        if (stderrText != null && !stderrText.isBlank()) {
            return stderrText.trim();
        }
        return NO_DIAGNOSTICS;
    }

    /** Returns the first non-blank textual node from a small set of diagnostic candidates. */
    private static String firstText(JsonNode... nodes) {
        for (JsonNode node : nodes) {
            if (node != null && node.isTextual() && !node.textValue().isBlank()) {
                return node.textValue();
            }
        }
        return "";
    }

    /** Identifies temporary capacity, service, timeout, and transport failures. */
    static boolean isTransientFailure(String details) {
        if (details == null) {
            return false;
        }
        String normalized = details.toLowerCase(Locale.ROOT);
        return normalized.equals(NO_DIAGNOSTICS)
                || normalized.contains("at capacity")
                || normalized.contains("temporarily unavailable")
                || normalized.contains("service unavailable")
                || normalized.contains("internal server error")
                || normalized.contains("stream disconnected")
                || normalized.contains("connection reset")
                || normalized.contains("connection refused")
                || normalized.contains("timed out")
                || normalized.contains("timeout")
                || normalized.contains("http 502")
                || normalized.contains("http 503")
                || normalized.contains("http 504");
    }

    // -----------------------------
    // Authentication isolation and lifecycle
    // -----------------------------

    /** Fails before a game is dealt when the configured Copilot executable is unavailable. */
    private void requireCliAvailable() {
        try {
            Process process = subscriptionProcess(List.of(executable, "--version"))
                    .redirectErrorStream(true)
                    .start();
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException("Timed out while checking Copilot CLI");
            }
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (process.exitValue() != 0 || !output.contains("GitHub Copilot CLI")) {
                throw new IllegalStateException(
                        "Copilot CLI is unavailable; `copilot --version` reported: " + output.trim());
            }
        } catch (IOException exception) {
            throw new IllegalStateException("Could not check Copilot CLI", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while checking Copilot CLI", exception);
        }
    }

    /** Creates a process that cannot switch from the Copilot subscription to a custom provider. */
    private static ProcessBuilder subscriptionProcess(List<String> command) {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().remove("COPILOT_PROVIDER_BASE_URL");
        processBuilder.environment().remove("COPILOT_PROVIDER_TYPE");
        processBuilder.environment().remove("COPILOT_PROVIDER_API_KEY");
        processBuilder.environment().remove("ANTHROPIC_API_KEY");
        processBuilder.environment().remove("OPENAI_API_KEY");
        return processBuilder;
    }

    /** Marks the game complete and removes temporary event and diagnostic files. */
    @PreDestroy
    @Override
    public synchronized void close() {
        closed = true;
        deleteRecursively(workDirectory);
        workDirectory = null;
    }

    /** Reads a bounded diagnostic suffix suitable for an exception message. */
    private static String readForError(Path path) throws IOException {
        String text = Files.exists(path)
                ? Files.readString(path, StandardCharsets.UTF_8).trim()
                : "";
        return text.length() <= 2_000 ? text : text.substring(text.length() - 2_000);
    }

    /** Removes a game's temporary directory from leaves to root on a best-effort basis. */
    private static void deleteRecursively(Path directory) {
        if (directory == null || !Files.exists(directory)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(directory)) {
            paths.sorted(Comparator.reverseOrder()).forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException ignored) {
                    // Temporary inference files are best-effort cleanup only.
                }
            });
        } catch (IOException ignored) {
            // Temporary inference files are best-effort cleanup only.
        }
    }

    /** One completed Copilot invocation after streaming events have been consolidated. */
    record CliResponse(
            String content, String model, String sessionId, double premiumRequests) {
    }
}

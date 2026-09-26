package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.ExperimentMetadataProvider;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Codex CLI-backed player authenticated through a ChatGPT subscription.
 *
 * <p>Each game uses one persistent Codex session. A pre-game turn asks the model to state the
 * Klondike strategy it already knows, and every gameplay turn resumes that conversation. This lets
 * the model retain its strategy, previous boards, commands, and feedback without receiving the
 * engine's hand-authored strategy. The CLI is required to report ChatGPT authentication before a
 * game can start; API-key environment variables are also removed from child processes to avoid
 * accidental API billing.
 */
@Component
@Profile("ai-codex")
public class CodexCliPlayer extends AIPlayer implements Player, ExperimentMetadataProvider, AutoCloseable {

    // -----------------------------
    // Experiment configuration
    // -----------------------------

    private static final Logger log = LoggerFactory.getLogger(CodexCliPlayer.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final Pattern ANSI = Pattern.compile("\\u001B\\[[;\\d]*m");
    private static final String DEFAULT_EXECUTABLE = "codex";
    private static final String DEFAULT_MODEL = "gpt-5.6-sol";
    private static final String DEFAULT_REASONING_EFFORT = "medium";
    private static final int DEFAULT_TIMEOUT_SECONDS = 180;
    private static final int DEFAULT_MAX_ATTEMPTS = 5;
    private static final long DEFAULT_INITIAL_RETRY_DELAY_MILLIS = 5_000L;
    private static final long MAX_RETRY_DELAY_MILLIS = 60_000L;
    private static final String NO_DIAGNOSTICS = "no diagnostics emitted";

    /** The only strategic prompt: the model must supply its own knowledge before seeing a board. */
    static final String STRATEGY_PROMPT = LlmGamePrompts.STRATEGY_PROMPT;

    // -----------------------------
    // Per-game session state
    // -----------------------------

    // Immutable experiment configuration. These values must not change midway through a game.
    private final String executable;
    private final String modelName;
    private final String reasoningEffort;
    private final int timeoutSeconds;
    private final int maxAttempts;
    private final long initialRetryDelayMillis;

    // Game-scoped session state. A CodexCliPlayer instance must never be shared between games.
    private Path workDirectory;
    private String sessionId;
    private String preGameStrategy;
    private String lastCommand;
    private int turnNumber;
    private boolean closed;

    // -----------------------------
    // Construction and configuration
    // -----------------------------

    /**
     * Creates a player from JVM properties, which is the path used by direct Java callers.
     */
    public CodexCliPlayer() {
        this(
                System.getProperty("codex.cli.executable", DEFAULT_EXECUTABLE),
                configuredModelName(),
                configuredReasoningEffort(),
                Integer.getInteger("codex.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

    /**
     * Creates a player for one explicit model/reasoning combination in a result sweep.
     *
     * @param modelName exact Codex model identifier
     * @param reasoningEffort reasoning setting passed to every turn in the session
     */
    public CodexCliPlayer(String modelName, String reasoningEffort) {
        this(
                System.getProperty("codex.cli.executable", DEFAULT_EXECUTABLE),
                modelName,
                reasoningEffort,
                Integer.getInteger("codex.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

    /**
     * Spring constructor used by the {@code ai-codex} profile.
     *
     * <p>Authentication is checked eagerly so a game fails before dealing cards when the CLI would
     * use anything other than the intended ChatGPT subscription.
     *
     * @param executable Codex CLI executable or path
     * @param modelName exact Codex model identifier
     * @param reasoningEffort reasoning setting retained throughout the game
     * @param timeoutSeconds maximum duration of each individual Codex turn
     */
    @Autowired
    public CodexCliPlayer(
            @Value("${codex.cli.executable:" + DEFAULT_EXECUTABLE + "}") String executable,
            @Value("${codex.model:" + DEFAULT_MODEL + "}") String modelName,
            @Value("${codex.reasoning.effort:" + DEFAULT_REASONING_EFFORT + "}") String reasoningEffort,
            @Value("${codex.timeout.seconds:" + DEFAULT_TIMEOUT_SECONDS + "}") int timeoutSeconds) {
        this.executable = executable;
        this.modelName = modelName;
        this.reasoningEffort = reasoningEffort;
        this.timeoutSeconds = timeoutSeconds;
        this.maxAttempts = Math.max(1, Integer.getInteger("codex.retry.max.attempts", DEFAULT_MAX_ATTEMPTS));
        this.initialRetryDelayMillis = Math.max(
                0L,
                Long.getLong("codex.retry.initial.delay.millis", DEFAULT_INITIAL_RETRY_DELAY_MILLIS));
        requireChatGptLogin();
    }

    /**
     * Returns the model selected by the {@code codex.model} JVM property or the repository default.
     */
    public static String configuredModelName() {
        return System.getProperty("codex.model", DEFAULT_MODEL);
    }

    /**
     * Returns the configured reasoning effort for the currently selected model.
     */
    public static String configuredReasoningEffort() {
        return configuredReasoningEffort(configuredModelName());
    }

    /**
     * Resolves reasoning effort, preferring an explicit JVM property over model metadata.
     *
     * @param modelName model whose catalog recommendation should be used as the fallback
     * @return reasoning effort accepted by the Codex CLI
     */
    public static String configuredReasoningEffort(String modelName) {
        // A command-line override makes deliberate reasoning-level experiments possible.
        String configured = System.getProperty("codex.reasoning.effort");
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        // Known models use their catalog recommendation; unknown models use the conservative default.
        return OpenAIModelInfo.byModelName(modelName)
                .flatMap(OpenAIModelInfo::getRecommendedCodexReasoningEffort)
                .orElse(DEFAULT_REASONING_EFFORT);
    }

    // -----------------------------
    // Game turn lifecycle
    // -----------------------------

    /**
     * Asks the persistent Codex session to choose one legal move for the current board.
     *
     * <p>The method is synchronized because the session transcript is ordered state: concurrent
     * calls could resume the same thread out of sequence and corrupt the experiment.
     *
     * @param solitaire current game state rendered into the turn prompt
     * @param moves engine recommendations; intentionally ignored so the model uses its own strategy
     * @param feedback engine feedback from the previous command, when present
     * @return one command copied from the engine-generated legal-move set
     */
    @Override
    public synchronized String nextCommand(Solitaire solitaire, String moves, String feedback) {
        // A closed player represents a completed game and must not silently start another session.
        if (closed) {
            throw new IllegalStateException("Codex CLI player is closed");
        }

        // The engine, rather than the model, remains the authority on move legality.
        List<String> legalMoves = new ArrayList<>(LegalMovesHelper.listLegalMoves(solitaire));
        if (legalMoves.isEmpty()) {
            // Keep the response schema satisfiable even in a terminal position.
            legalMoves.add("quit");
        }

        try {
            // Keep all files for this game's resumable thread together until close().
            if (workDirectory == null) {
                workDirectory = Files.createTempDirectory("solitaire-codex-");
            }

            // The pre-game strategy answer becomes the first item in the persistent conversation.
            if (sessionId == null) {
                startGameSession();
            }

            // Use separate files per move so failures can be diagnosed without overwriting prior turns.
            turnNumber++;
            Path schemaPath = workDirectory.resolve("command-schema.json");
            Path responsePath = workDirectory.resolve("response-" + turnNumber + ".json");
            Path stdoutPath = workDirectory.resolve("stdout-" + turnNumber + ".jsonl");
            Path stderrPath = workDirectory.resolve("stderr-" + turnNumber + ".log");
            writeCommandSchema(schemaPath, legalMoves);

            // Every board decision resumes the exact game thread captured during strategy setup.
            List<String> command = resumeCommand(schemaPath, responsePath);

            // Interface instructions are needed once; later turns inherit them from session context.
            String prompt = buildTurnPrompt(solitaire, feedback, legalMoves, turnNumber == 1);
            runProcess(command, prompt, stdoutPath, stderrPath);

            // Schema validation occurs in Codex, then this membership check defends our game boundary.
            String selectedCommand = parseCommand(Files.readString(responsePath, StandardCharsets.UTF_8));
            if (!legalMoves.contains(selectedCommand)) {
                throw new IllegalStateException("Codex CLI returned a command outside the legal-move list: " + selectedCommand);
            }
            // Retain the final choice so the result runner can distinguish a model-requested quit.
            lastCommand = selectedCommand;
            return selectedCommand;
        } catch (IOException exception) {
            throw new IllegalStateException("Could not run Codex CLI", exception);
        } catch (InterruptedException exception) {
            // Preserve interruption so Gradle or the application can stop a long experiment cleanly.
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for Codex CLI", exception);
        }
    }

    /**
     * Returns the persistent Codex thread ID for experiment logging and transcript lookup.
     *
     * @return session ID after strategy setup, or {@code null} before the first command
     */
    public synchronized String getSessionId() {
        return sessionId;
    }

    /**
     * Returns the most recent schema-validated command selected by the model.
     *
     * @return last command, or {@code null} before the first board decision
     */
    public synchronized String getLastCommand() {
        return lastCommand;
    }

    /** Describes the exact model protocol used by this game for episode analysis. */
    @Override
    public synchronized Map<String, Object> getExperimentMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("provider", "OpenAI");
        metadata.put("model", modelName);
        metadata.put("reasoning", reasoningEffort);
        metadata.put("session_id", sessionId);
        metadata.put("prompt_version", LlmGamePrompts.PROMPT_VERSION);
        metadata.put("pre_game_strategy", preGameStrategy);
        metadata.put("stateful", true);
        return metadata;
    }

    /**
     * Starts the game-scoped session and asks the model to formulate its own Klondike strategy.
     *
     * <p>This turn deliberately has no move schema because it produces prose that remains in the
     * conversation as the model's plan. The first board is sent only after this turn completes.
     */
    private void startGameSession() throws IOException, InterruptedException {
        // Preserve the prose response and JSON event stream separately for diagnosis during the game.
        Path strategyPath = workDirectory.resolve("strategy.txt");
        Path stdoutPath = workDirectory.resolve("strategy-events.jsonl");
        Path stderrPath = workDirectory.resolve("strategy-stderr.log");
        runProcess(strategyCommand(strategyPath), STRATEGY_PROMPT, stdoutPath, stderrPath);

        // The thread.started event gives us a concurrency-safe ID; --last is intentionally avoided.
        sessionId = parseSessionId(Files.readString(stdoutPath, StandardCharsets.UTF_8));
        preGameStrategy = Files.readString(strategyPath, StandardCharsets.UTF_8).trim();
        if (preGameStrategy.isBlank()) {
            throw new IllegalStateException("Codex CLI returned an empty pre-game strategy");
        }
        // The session ID is the durable link to Codex's stored transcript for later investigation.
        log.info("Started Codex CLI game session {} using {} with reasoning={}",
                sessionId, modelName, reasoningEffort);
        if (log.isDebugEnabled()) {
            log.debug("Codex CLI pre-game strategy for session {}:\n{}", sessionId, preGameStrategy);
        }
    }

    // -----------------------------
    // Codex CLI process commands
    // -----------------------------

    /**
     * Builds the initial non-ephemeral CLI invocation used to create a resumable thread.
     *
     * @param strategyPath destination for the model's prose strategy
     * @return process arguments for the strategy turn
     */
    private List<String> strategyCommand(Path strategyPath) {
        return List.of(
                executable,
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--model",
                modelName,
                "--config",
                "model_reasoning_effort=\"" + reasoningEffort + "\"",
                "--output-last-message",
                strategyPath.toString(),
                "--json",
                "--cd",
                workDirectory.toString(),
                "-");
    }

    /**
     * Runs one Codex process with bounded execution time, transient retries, and captured diagnostics.
     *
     * <p>A resumed request may appear twice in the Codex transcript when an overloaded request was
     * accepted far enough to append its user message before failing. Re-sending the identical board
     * does not alter the information or strategic guidance available to the model.
     *
     * @param command complete CLI argument list
     * @param prompt UTF-8 prompt written to standard input
     * @param stdoutPath destination for machine-readable JSONL events
     * @param stderrPath destination for CLI diagnostics
     */
    private void runProcess(List<String> command, String prompt, Path stdoutPath, Path stderrPath)
            throws IOException, InterruptedException {
        long retryDelayMillis = initialRetryDelayMillis;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            // Each attempt replaces its diagnostics so a successful retry cannot be confused with
            // an earlier failed process.
            Files.deleteIfExists(stdoutPath);
            Files.deleteIfExists(stderrPath);

            // Capturing both streams prevents verbose CLI output from obscuring Gradle's result summary.
            ProcessBuilder processBuilder = subscriptionProcess(command)
                    .redirectOutput(stdoutPath.toFile())
                    .redirectError(stderrPath.toFile());
            Process process = processBuilder.start();

            // Closing stdin signals that the complete non-interactive prompt has been delivered.
            try (var stdin = process.getOutputStream()) {
                stdin.write(prompt.getBytes(StandardCharsets.UTF_8));
            }

            // A per-turn timeout is transient: terminate this process, then resume the same session.
            boolean timedOut = !process.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (timedOut) {
                process.destroyForcibly();
                process.waitFor();
            }

            String details = timedOut
                    ? "Codex CLI timed out after " + timeoutSeconds + " seconds"
                    : failureDetails(readForError(stdoutPath), readForError(stderrPath));
            if (!timedOut && process.exitValue() == 0) {
                return;
            }

            // Retry only failures that are expected to clear without changing the experiment.
            if (attempt < maxAttempts && (timedOut || isTransientFailure(details))) {
                log.warn(
                        "Transient Codex CLI failure on attempt {}/{}: {}. Retrying in {} ms",
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
                    "Codex CLI failed with " + status + " after " + attempt + " attempt(s): " + details);
        }
    }

    /**
     * Extracts the most useful message and error code from Codex JSONL, falling back to stderr.
     *
     * @param stdoutJsonl Codex CLI JSON event stream
     * @param stderrText process stderr
     * @return concise diagnostics suitable for logs and exceptions
     */
    static String failureDetails(String stdoutJsonl, String stderrText) {
        String message = "";
        String code = "";
        if (stdoutJsonl != null) {
            for (String line : stdoutJsonl.lines().toList()) {
                if (line.isBlank()) {
                    continue;
                }
                try {
                    JsonNode error = OBJECT_MAPPER.readTree(line).path("payload").path("error");
                    if (error.isObject()) {
                        message = error.path("message").asText(message);
                        code = error.path("codex_error_info").asText(code);
                    }
                } catch (IOException ignored) {
                    // A malformed diagnostic line should not hide a valid error later in the stream.
                }
            }
        }
        if (!message.isBlank() || !code.isBlank()) {
            return code.isBlank() ? message : message + " [" + code + "]";
        }
        if (stderrText != null && !stderrText.isBlank()) {
            return stderrText.trim();
        }
        return NO_DIAGNOSTICS;
    }

    /**
     * Identifies failures for which repeating the identical Codex request is appropriate.
     *
     * @param details normalized process diagnostics
     * @return true for temporary capacity, service, or transport failures
     */
    static boolean isTransientFailure(String details) {
        if (details == null) {
            return false;
        }
        String normalized = details.toLowerCase(Locale.ROOT);
        return normalized.equals(NO_DIAGNOSTICS)
                || normalized.contains("server_overloaded")
                || normalized.contains("internal_server_error")
                || normalized.contains("at capacity")
                || normalized.contains("temporarily unavailable")
                || normalized.contains("service unavailable")
                || normalized.contains("stream disconnected")
                || normalized.contains("connection reset")
                || normalized.contains("connection refused")
                || normalized.contains("failed to send request")
                || normalized.contains("error sending request")
                || normalized.contains("timed out")
                || normalized.contains("timeout")
                || normalized.contains("http 502")
                || normalized.contains("http 503")
                || normalized.contains("http 504");
    }

    /**
     * Builds a resume invocation for one board decision in the current game thread.
     *
     * <p>The read-only sandbox is repeated explicitly because resume does not accept the top-level
     * {@code --sandbox} flag. The per-turn schema restricts output to moves legal on this board.
     *
     * @param schemaPath JSON schema containing the current legal moves
     * @param responsePath destination for the final structured response
     * @return process arguments for a resumed game turn
     */
    private List<String> resumeCommand(Path schemaPath, Path responsePath) {
        return List.of(
                executable,
                "exec",
                "resume",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--model",
                modelName,
                "--config",
                "model_reasoning_effort=\"" + reasoningEffort + "\"",
                "--config",
                "sandbox_mode=\"read-only\"",
                "--output-schema",
                schemaPath.toString(),
                "--output-last-message",
                responsePath.toString(),
                "--json",
                sessionId,
                "-");
    }

    // -----------------------------
    // Structured response parsing
    // -----------------------------

    /**
     * Extracts the selected command from the schema-constrained response.
     *
     * @param responseJson final response written by Codex
     * @return non-blank command text
     */
    static String parseCommand(String responseJson) throws IOException {
        JsonNode response = OBJECT_MAPPER.readTree(responseJson);
        JsonNode command = response.get("command");
        if (command == null || !command.isTextual() || command.textValue().isBlank()) {
            throw new IllegalStateException("Codex CLI response did not contain a command");
        }
        return command.textValue();
    }

    /**
     * Extracts the resumable thread ID from a {@code codex exec --json} event stream.
     *
     * @param eventsJsonl newline-delimited Codex events
     * @return exact thread ID emitted by the CLI
     */
    static String parseSessionId(String eventsJsonl) throws IOException {
        // JSON parsing is used instead of text matching so event ordering and extra fields are harmless.
        for (String line : eventsJsonl.lines().toList()) {
            if (line.isBlank()) {
                continue;
            }
            JsonNode event = OBJECT_MAPPER.readTree(line);
            if ("thread.started".equals(event.path("type").asText())) {
                JsonNode threadId = event.get("thread_id");
                if (threadId != null && threadId.isTextual() && !threadId.textValue().isBlank()) {
                    return threadId.textValue();
                }
            }
        }
        throw new IllegalStateException("Codex CLI did not report a thread.started session ID");
    }

    // -----------------------------
    // Authentication isolation
    // -----------------------------

    /**
     * Verifies that the installed CLI is authenticated through ChatGPT before any model call occurs.
     */
    private void requireChatGptLogin() {
        try {
            // Merge stderr here because login status is a short, human-oriented CLI response.
            Process process = subscriptionProcess(List.of(executable, "login", "status"))
                    .redirectErrorStream(true)
                    .start();
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException("Timed out while checking Codex CLI authentication");
            }
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            // Refuse other authentication modes instead of risking direct API charges.
            if (process.exitValue() != 0 || !output.contains("Logged in using ChatGPT")) {
                throw new IllegalStateException(
                        "Codex CLI must be logged in using ChatGPT; `codex login status` reported: " + output.trim());
            }
        } catch (IOException exception) {
            throw new IllegalStateException("Could not check Codex CLI authentication", exception);
        } catch (InterruptedException exception) {
            // Match nextCommand's interruption behavior during application startup.
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while checking Codex CLI authentication", exception);
        }
    }

    /**
     * Creates a subprocess with API-key credentials removed from its inherited environment.
     *
     * @param command CLI argument list
     * @return guarded process builder that can use only the existing ChatGPT login
     */
    private static ProcessBuilder subscriptionProcess(List<String> command) {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        // Environment removal is defense in depth in addition to the explicit login-status check.
        processBuilder.environment().remove("OPENAI_API_KEY");
        processBuilder.environment().remove("CODEX_API_KEY");
        return processBuilder;
    }

    // -----------------------------
    // Turn prompt construction
    // -----------------------------

    /**
     * Builds one gameplay prompt without adding strategic advice from the engine.
     *
     * @param solitaire current board
     * @param feedback execution feedback from the previous turn, if any
     * @param legalMoves complete set of commands accepted for this turn
     * @param firstTurn whether to establish the board notation and response contract
     * @return prompt appended to the persistent Codex conversation
     */
    static String buildTurnPrompt(
            Solitaire solitaire, String feedback, List<String> legalMoves, boolean firstTurn) {
        return LlmGamePrompts.buildTurnPrompt(solitaire, feedback, legalMoves, firstTurn);
    }

    // -----------------------------
    // Lifecycle and file helpers
    // -----------------------------

    /**
     * Marks this game complete and removes transient schemas, responses, and process logs.
     *
     * <p>The Codex transcript itself remains in Codex's session store and can be inspected through
     * the logged session ID. Spring calls this method at bean shutdown; result tests call it after
     * every game through try-with-resources.
     */
    @PreDestroy
    @Override
    public synchronized void close() {
        closed = true;
        deleteRecursively(workDirectory);
        workDirectory = null;
    }

    /**
     * Removes terminal colour codes so the model receives stable plain-text board state.
     */
    private static String stripAnsi(String input) {
        return input == null ? "" : ANSI.matcher(input).replaceAll("");
    }

    /**
     * Writes a strict response schema whose enum is the legal-move set for this exact board.
     */
    private static void writeCommandSchema(Path schemaPath, List<String> legalMoves) throws IOException {
        // Restrict the command field itself before wrapping it in the top-level response object.
        Map<String, Object> commandProperty = new LinkedHashMap<>();
        commandProperty.put("type", "string");
        commandProperty.put("enum", legalMoves);

        // Reject omitted commands and extra explanatory fields to keep game input deterministic.
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", Map.of("command", commandProperty));
        schema.put("required", List.of("command"));
        schema.put("additionalProperties", false);
        OBJECT_MAPPER.writeValue(schemaPath.toFile(), schema);
    }

    /**
     * Reads a bounded diagnostic suffix suitable for inclusion in an exception message.
     */
    private static String readForError(Path path) throws IOException {
        String text = Files.readString(path, StandardCharsets.UTF_8).trim();
        return text.length() <= 2_000 ? text : text.substring(text.length() - 2_000);
    }

    /**
     * Deletes a game's temporary directory from leaves to root on a best-effort basis.
     */
    private static void deleteRecursively(Path directory) {
        if (directory == null || !Files.exists(directory)) {
            return;
        }
        // Reverse sorting ensures files and child directories are removed before their parents.
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
}

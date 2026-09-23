package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
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
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Codex CLI-backed player authenticated through a ChatGPT subscription.
 *
 * <p>Each turn runs in a fresh ephemeral session so the complete board state is the only gameplay
 * context. The CLI is required to report ChatGPT authentication before a game can start; API-key
 * environment variables are also removed from child processes to avoid accidental API billing.
 */
@Component
@Profile("ai-codex")
public class CodexCliPlayer extends AIPlayer implements Player {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final Pattern ANSI = Pattern.compile("\\u001B\\[[;\\d]*m");
    private static final String DEFAULT_EXECUTABLE = "codex";
    private static final String DEFAULT_MODEL = "gpt-5.6-sol";
    private static final String DEFAULT_REASONING_EFFORT = "medium";
    private static final int DEFAULT_TIMEOUT_SECONDS = 180;

    private final String executable;
    private final String modelName;
    private final String reasoningEffort;
    private final int timeoutSeconds;

    public CodexCliPlayer() {
        this(
                System.getProperty("codex.cli.executable", DEFAULT_EXECUTABLE),
                configuredModelName(),
                configuredReasoningEffort(),
                Integer.getInteger("codex.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

    public CodexCliPlayer(String modelName, String reasoningEffort) {
        this(
                System.getProperty("codex.cli.executable", DEFAULT_EXECUTABLE),
                modelName,
                reasoningEffort,
                Integer.getInteger("codex.timeout.seconds", DEFAULT_TIMEOUT_SECONDS));
    }

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
        requireChatGptLogin();
    }

    public static String configuredModelName() {
        return System.getProperty("codex.model", DEFAULT_MODEL);
    }

    public static String configuredReasoningEffort() {
        return configuredReasoningEffort(configuredModelName());
    }

    public static String configuredReasoningEffort(String modelName) {
        String configured = System.getProperty("codex.reasoning.effort");
        if (configured != null && !configured.isBlank()) {
            return configured.trim();
        }
        return OpenAIModelInfo.byModelName(modelName)
                .flatMap(OpenAIModelInfo::getRecommendedCodexReasoningEffort)
                .orElse(DEFAULT_REASONING_EFFORT);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        List<String> legalMoves = new ArrayList<>(LegalMovesHelper.listLegalMoves(solitaire));
        if (legalMoves.isEmpty()) {
            legalMoves.add("quit");
        }

        Path workDirectory = null;
        try {
            workDirectory = Files.createTempDirectory("solitaire-codex-");
            Path schemaPath = workDirectory.resolve("command-schema.json");
            Path responsePath = workDirectory.resolve("response.json");
            Path stdoutPath = workDirectory.resolve("stdout.log");
            Path stderrPath = workDirectory.resolve("stderr.log");
            writeCommandSchema(schemaPath, legalMoves);

            List<String> command = List.of(
                    executable,
                    "exec",
                    "--ephemeral",
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
                    "--output-schema",
                    schemaPath.toString(),
                    "--output-last-message",
                    responsePath.toString(),
                    "--cd",
                    workDirectory.toString(),
                    "-");

            ProcessBuilder processBuilder = subscriptionProcess(command)
                    .redirectOutput(stdoutPath.toFile())
                    .redirectError(stderrPath.toFile());
            Process process = processBuilder.start();
            try (var stdin = process.getOutputStream()) {
                stdin.write(buildPrompt(solitaire, moves, feedback, legalMoves).getBytes(StandardCharsets.UTF_8));
            }

            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException("Codex CLI timed out after " + timeoutSeconds + " seconds");
            }
            if (process.exitValue() != 0) {
                throw new IllegalStateException(
                        "Codex CLI exited with status " + process.exitValue() + ": " + readForError(stderrPath));
            }

            String selectedCommand = parseCommand(Files.readString(responsePath, StandardCharsets.UTF_8));
            if (!legalMoves.contains(selectedCommand)) {
                throw new IllegalStateException("Codex CLI returned a command outside the legal-move list: " + selectedCommand);
            }
            return selectedCommand;
        } catch (IOException exception) {
            throw new IllegalStateException("Could not run Codex CLI", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for Codex CLI", exception);
        } finally {
            deleteRecursively(workDirectory);
        }
    }

    static String parseCommand(String responseJson) throws IOException {
        JsonNode response = OBJECT_MAPPER.readTree(responseJson);
        JsonNode command = response.get("command");
        if (command == null || !command.isTextual() || command.textValue().isBlank()) {
            throw new IllegalStateException("Codex CLI response did not contain a command");
        }
        return command.textValue();
    }

    private void requireChatGptLogin() {
        try {
            Process process = subscriptionProcess(List.of(executable, "login", "status"))
                    .redirectErrorStream(true)
                    .start();
            if (!process.waitFor(10, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException("Timed out while checking Codex CLI authentication");
            }
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (process.exitValue() != 0 || !output.contains("Logged in using ChatGPT")) {
                throw new IllegalStateException(
                        "Codex CLI must be logged in using ChatGPT; `codex login status` reported: " + output.trim());
            }
        } catch (IOException exception) {
            throw new IllegalStateException("Could not check Codex CLI authentication", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while checking Codex CLI authentication", exception);
        }
    }

    private static ProcessBuilder subscriptionProcess(List<String> command) {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.environment().remove("OPENAI_API_KEY");
        processBuilder.environment().remove("CODEX_API_KEY");
        return processBuilder;
    }

    private static String buildPrompt(
            Solitaire solitaire, String moves, String feedback, List<String> legalMoves) {
        StringBuilder prompt = new StringBuilder(OllamaPlayer.SYSTEM_PROMPT)
                .append("\n\n# Response format override\n")
                .append("For this Codex CLI evaluation, the JSON response schema supersedes the instructions ")
                .append("to output a bare command line. Put exactly one listed command in the command field.")
                .append("\n\n# Current board\n")
                .append(stripAnsi(solitaire.toString()))
                .append("\n\n# Complete legal-move list\n");
        legalMoves.forEach(move -> prompt.append("- ").append(move).append('\n'));
        if (feedback != null && !feedback.isBlank()) {
            prompt.append("\n# Engine feedback\n").append(stripAnsi(feedback.trim())).append('\n');
        }
        if (moves != null && !moves.isBlank()) {
            prompt.append("\n# Engine recommendations\n").append(stripAnsi(moves.trim())).append('\n');
        }
        prompt.append("\nReturn only the JSON object required by the response schema.");
        return prompt.toString();
    }

    private static String stripAnsi(String input) {
        return input == null ? "" : ANSI.matcher(input).replaceAll("");
    }

    private static void writeCommandSchema(Path schemaPath, List<String> legalMoves) throws IOException {
        Map<String, Object> commandProperty = new LinkedHashMap<>();
        commandProperty.put("type", "string");
        commandProperty.put("enum", legalMoves);

        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", Map.of("command", commandProperty));
        schema.put("required", List.of("command"));
        schema.put("additionalProperties", false);
        OBJECT_MAPPER.writeValue(schemaPath.toFile(), schema);
    }

    private static String readForError(Path path) throws IOException {
        String text = Files.readString(path, StandardCharsets.UTF_8).trim();
        return text.length() <= 2_000 ? text : text.substring(text.length() - 2_000);
    }

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
}

package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import org.springframework.ai.ollama.api.ThinkOption;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Stateful Ollama-backed Solitaire player using Spring AI.
 *
 * <p>Each instance represents one game. Before seeing the deal, the model states the Klondike
 * strategy it already knows. That model-authored strategy is pinned for the game while a bounded
 * window retains recent boards, commands, and feedback. The engine supplies the complete legal
 * move list but no rules, strategic guidance, or recommended move.
 */
@Component
@Profile("ai-ollama")
public class OllamaPlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(OllamaPlayer.class);
    private static final ObjectMapper JSON = new ObjectMapper();

    private static final String LOCAL_OLLAMA_URL = "http://localhost:11434";
    private static final String DEFAULT_MODEL = "llama3";
    private static final int DEFAULT_MEMORY_TURNS = 8;
    private static final int DEFAULT_CONTEXT_TOKENS = 32_768;
    private static final int DEFAULT_MAX_OUTPUT_TOKENS = 1_024;
    private static final String DEFAULT_THINKING = "off";

    private final ChatClient chatClient;
    private final ChatMemory chatMemory;
    private final String conversationId;
    private final String modelName;
    private final int memoryTurns;

    private boolean strategyInitialized;
    private int turnNumber;

    /** Creates a local player with system-property defaults for command-line and test use. */
    public OllamaPlayer() {
        this(DEFAULT_MODEL);
    }

    /** Creates a local player for one explicitly selected Ollama model. */
    public OllamaPlayer(String modelName) {
        this(
                modelName,
                Integer.getInteger("ollama.memory.turns", DEFAULT_MEMORY_TURNS),
                Integer.getInteger("ollama.context.tokens", DEFAULT_CONTEXT_TOKENS),
                Integer.getInteger("ollama.max.output.tokens", DEFAULT_MAX_OUTPUT_TOKENS),
                System.getProperty("ollama.thinking", DEFAULT_THINKING));
    }

    /** Creates the Spring-managed player from application or JVM properties. */
    @Autowired
    public OllamaPlayer(
            @Value("${ollama.model:" + DEFAULT_MODEL + "}") String modelName,
            @Value("${ollama.memory.turns:" + DEFAULT_MEMORY_TURNS + "}") int memoryTurns,
            @Value("${ollama.context.tokens:" + DEFAULT_CONTEXT_TOKENS + "}") int contextTokens,
            @Value("${ollama.max.output.tokens:" + DEFAULT_MAX_OUTPUT_TOKENS + "}") int maxOutputTokens,
            @Value("${ollama.thinking:" + DEFAULT_THINKING + "}") String thinking) {
        this(
                buildLocalChatModel(modelName, contextTokens, maxOutputTokens, thinking),
                modelName,
                memoryTurns,
                UUID.randomUUID().toString());
    }

    /** Package-visible constructor used by focused tests with an in-process chat model. */
    OllamaPlayer(ChatModel chatModel, String modelName, int memoryTurns, String conversationId) {
        if (memoryTurns < 1) {
            throw new IllegalArgumentException("Ollama memory turns must be at least 1");
        }
        this.modelName = modelName;
        this.memoryTurns = memoryTurns;
        this.conversationId = conversationId;

        // One pinned system message plus one user/assistant pair for each retained gameplay turn.
        this.chatMemory = MessageWindowChatMemory.builder()
                .maxMessages(1 + (memoryTurns * 2))
                .build();
        MessageChatMemoryAdvisor memoryAdvisor = MessageChatMemoryAdvisor.builder(chatMemory)
                .conversationId(conversationId)
                .build();
        this.chatClient = ChatClient.builder(chatModel)
                .defaultAdvisors(memoryAdvisor)
                .build();
    }

    /**
     * Chooses one engine-validated legal move while retaining recent game context.
     *
     * @param solitaire authoritative current game state
     * @param moves engine recommendations, intentionally ignored for this unguided experiment
     * @param feedback execution feedback from the preceding command, if any
     * @return one command copied from the complete legal-move list
     */
    @Override
    public synchronized String nextCommand(Solitaire solitaire, String moves, String feedback) {
        // The model's own strategy is established before it sees the first deal.
        if (!strategyInitialized) {
            initializeStrategy();
        }

        // Generate legal actions independently of GuidanceService so guidance can remain disabled.
        List<String> legalMoves = new ArrayList<>(LegalMovesHelper.listLegalMoves(solitaire));
        if (legalMoves.isEmpty()) {
            legalMoves.add("quit");
        }

        turnNumber++;
        String prompt = LlmGamePrompts.buildTurnPrompt(solitaire, feedback, legalMoves, false);
        if (log.isTraceEnabled()) {
            log.trace("Ollama turn {} prompt for conversation {}:\n{}", turnNumber, conversationId, prompt);
        }

        // Ollama enforces the dynamic enum before Spring AI returns the JSON response.
        String response = chatClient.prompt()
                .user(prompt)
                .options(OllamaChatOptions.builder()
                        .format(commandSchema(legalMoves))
                        .build())
                .call()
                .content();
        String selectedCommand = parseCommand(response);

        // Keep the engine as the final authority even when structured output is enabled.
        if (!legalMoves.contains(selectedCommand)) {
            throw new IllegalStateException(
                    "Ollama returned a command outside the legal-move list: " + selectedCommand);
        }
        if (log.isTraceEnabled()) {
            log.trace("Ollama turn {} response for conversation {}: {}", turnNumber, conversationId, response);
        }
        return selectedCommand;
    }

    /** Asks for existing knowledge, then pins that answer above the rolling turn window. */
    private void initializeStrategy() {
        String strategy = chatClient.prompt()
                .user(LlmGamePrompts.STRATEGY_PROMPT)
                .call()
                .content();
        if (strategy == null || strategy.isBlank()) {
            throw new IllegalStateException("Ollama returned an empty pre-game strategy");
        }

        // Replace the temporary strategy exchange with one preserved system message. Spring AI's
        // MessageWindowChatMemory evicts old user/assistant turns but always retains this message.
        chatMemory.clear(conversationId);
        chatMemory.add(conversationId, new SystemMessage(persistentSystemPrompt(strategy.trim())));
        strategyInitialized = true;

        log.info("Started Ollama game conversation {} using {} with {} retained turns",
                conversationId, modelName, memoryTurns);
        if (log.isDebugEnabled()) {
            log.debug("Ollama pre-game strategy for conversation {}:\n{}", conversationId, strategy.trim());
        }
    }

    /** Combines model-authored strategy with the non-strategic game interface. */
    static String persistentSystemPrompt(String strategy) {
        return "# Strategy you described before the game\n"
                + strategy
                + "\n\n"
                + LlmGamePrompts.GAME_INTERFACE;
    }

    /** Creates the JSON schema that restricts the model to this turn's legal commands. */
    static Map<String, Object> commandSchema(List<String> legalMoves) {
        Map<String, Object> command = new LinkedHashMap<>();
        command.put("type", "string");
        command.put("enum", List.copyOf(legalMoves));

        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("command", command);

        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("type", "object");
        schema.put("properties", properties);
        schema.put("required", List.of("command"));
        schema.put("additionalProperties", false);
        return schema;
    }

    /** Extracts the command from Ollama's structured response. */
    static String parseCommand(String response) {
        if (response == null || response.isBlank()) {
            throw new IllegalStateException("Ollama returned an empty move response");
        }
        try {
            JsonNode command = JSON.readTree(response).get("command");
            if (command == null || !command.isTextual() || command.asText().isBlank()) {
                throw new IllegalStateException("Ollama response did not contain a command: " + response);
            }
            return command.asText();
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Could not parse Ollama move response: " + response, exception);
        }
    }

    /** Exposes the game-scoped conversation identifier for experiment diagnostics. */
    public String getConversationId() {
        return conversationId;
    }

    /** Returns a snapshot of retained messages for package-level verification. */
    List<Message> retainedMessages() {
        return chatMemory.get(conversationId);
    }

    /** Builds the local model with an explicit context budget and visible overflow failures. */
    private static ChatModel buildLocalChatModel(
            String modelName, int contextTokens, int maxOutputTokens, String thinking) {
        if (contextTokens < 1) {
            throw new IllegalArgumentException("Ollama context tokens must be at least 1");
        }
        if (maxOutputTokens < 1) {
            throw new IllegalArgumentException("Ollama maximum output tokens must be at least 1");
        }
        OllamaApi api = OllamaApi.builder()
                .baseUrl(LOCAL_OLLAMA_URL)
                .build();
        OllamaChatOptions.Builder options = OllamaChatOptions.builder()
                .model(modelName)
                .numCtx(contextTokens)
                .numPredict(maxOutputTokens)
                .truncate(false);
        configureThinking(options, thinking);
        return OllamaChatModel.builder()
                .ollamaApi(api)
                .defaultOptions(options.build())
                .build();
    }

    /** Applies a model-compatible thinking mode while allowing explicit Ollama auto-detection. */
    static void configureThinking(OllamaChatOptions.Builder options, String thinking) {
        String normalized = thinking == null ? DEFAULT_THINKING : thinking.trim().toLowerCase();
        switch (normalized) {
            case "auto" -> {
                // Leave the option unset so Ollama chooses the model's default behavior.
            }
            case "off", "false", "none" -> options.disableThinking();
            case "on", "true" -> options.enableThinking();
            case "low" -> options.thinkOption(ThinkOption.ThinkLevel.LOW);
            case "medium" -> options.thinkOption(ThinkOption.ThinkLevel.MEDIUM);
            case "high" -> options.thinkOption(ThinkOption.ThinkLevel.HIGH);
            default -> throw new IllegalArgumentException(
                    "Unsupported Ollama thinking mode '" + thinking
                            + "'; use auto, off, on, low, medium, or high");
        }
    }
}

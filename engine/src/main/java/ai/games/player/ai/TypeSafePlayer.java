package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.DecisionMetadataProvider;
import ai.games.player.ExperimentMetadataProvider;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Klondike player backed by TypeSafe AI's Jev System One model.
 *
 * <p>Jev is used as a typed decision model rather than a chat model. Every turn supplies the
 * complete current board plus compact human-visible observations and commands from earlier turns,
 * while legal engine commands become opaque Choice options. The engine remains the authority for
 * legality.
 */
@Component
@Profile("ai-typesafe")
public class TypeSafePlayer extends AIPlayer
        implements Player, ExperimentMetadataProvider, DecisionMetadataProvider {

    private static final Logger log = LoggerFactory.getLogger(TypeSafePlayer.class);
    private static final ObjectMapper JSON = new ObjectMapper();

    static final String DEFAULT_MODEL = "jev-1.13.0";
    static final String DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone";
    private static final int DEFAULT_MAX_ATTEMPTS = 5;
    private static final long DEFAULT_INITIAL_RETRY_MILLIS = 500L;
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);

    private final String modelName;
    private final TypeSafeTransport transport;
    private final int maxAttempts;
    private final long initialRetryMillis;
    private final GamePrompts.PromptSet promptSet;
    private final List<TurnObservation> turnHistory = new ArrayList<>();

    private Map<String, Object> lastDecisionMetadata = Map.of();
    private long totalInputTokens;
    private long totalOutputTokens;
    private long totalLatencyMillis;

    /** Creates a player from JVM properties and the {@code TYPESAFE_API_KEY} environment variable. */
    public TypeSafePlayer() {
        this(
                configuredModelName(),
                new JdkTypeSafeTransport(
                        configuredBaseUrl(),
                        resolveApiKey(System.getProperty("typesafe.apiKey", "")),
                        DEFAULT_TIMEOUT),
                Integer.getInteger("typesafe.retry.max.attempts", DEFAULT_MAX_ATTEMPTS),
                Long.getLong("typesafe.retry.initial.delay.millis", DEFAULT_INITIAL_RETRY_MILLIS));
    }

    /** Creates the Spring-managed player from application properties or environment configuration. */
    @Autowired
    public TypeSafePlayer(
            @Value("${typesafe.model:" + DEFAULT_MODEL + "}") String modelName,
            @Value("${typesafe.baseUrl:" + DEFAULT_BASE_URL + "}") String baseUrl,
            @Value("${typesafe.apiKey:}") String apiKey,
            @Value("${typesafe.retry.max.attempts:" + DEFAULT_MAX_ATTEMPTS + "}") int maxAttempts,
            @Value("${typesafe.retry.initial.delay.millis:" + DEFAULT_INITIAL_RETRY_MILLIS + "}") long initialRetryMillis) {
        this(
                modelName,
                new JdkTypeSafeTransport(baseUrl, resolveApiKey(apiKey), DEFAULT_TIMEOUT),
                maxAttempts,
                initialRetryMillis);
    }

    /** Package-visible constructor used by focused tests with an in-process transport. */
    TypeSafePlayer(
            String modelName,
            TypeSafeTransport transport,
            int maxAttempts,
            long initialRetryMillis) {
        if (modelName == null || modelName.isBlank()) {
            throw new IllegalArgumentException("TypeSafe model name must not be blank");
        }
        if (maxAttempts < 1) {
            throw new IllegalArgumentException("TypeSafe retry attempts must be at least 1");
        }
        if (initialRetryMillis < 0) {
            throw new IllegalArgumentException("TypeSafe retry delay must not be negative");
        }
        this.modelName = modelName.trim();
        this.transport = transport;
        this.maxAttempts = maxAttempts;
        this.initialRetryMillis = initialRetryMillis;
        this.promptSet = GamePrompts.configuredPromptSet();
    }

    /** Returns the model selected by the {@code typesafe.model} JVM property. */
    public static String configuredModelName() {
        return System.getProperty("typesafe.model", DEFAULT_MODEL).trim();
    }

    /** Returns the endpoint selected by the {@code typesafe.baseUrl} JVM property. */
    public static String configuredBaseUrl() {
        return System.getProperty("typesafe.baseUrl", DEFAULT_BASE_URL).trim();
    }

    /** Returns the prompt profile that a newly constructed player will use. */
    public static String configuredPromptProfile() {
        return GamePrompts.configuredPromptSet().profile();
    }

    /** Returns the shared policy version selected for a newly constructed player. */
    public static String configuredPolicyVersion() {
        return GamePrompts.configuredPromptSet().version();
    }

    /**
     * Selects one command from the engine's legal-move list using a Jev Choice question.
     *
     * @param solitaire authoritative visible game state
     * @param moves engine recommendations, intentionally ignored for this unguided experiment
     * @param feedback execution feedback from the preceding command, if any
     * @return one command copied from the legal-move list
     */
    @Override
    public synchronized String nextCommand(Solitaire solitaire, String moves, String feedback) {
        List<String> legalMoves = new ArrayList<>(LegalMovesHelper.listLegalMoves(solitaire));
        if (!legalMoves.contains("quit")) {
            legalMoves.add("quit");
        }

        Map<String, Object> currentBoard = observedBoard(solitaire);
        Map<String, Object> historicalObservation = historicalObservation(solitaire);
        Map<String, String> optionCommands = optionCommands(legalMoves);
        String payload = serializeRequest(
                buildRequest(solitaire, currentBoard, feedback, optionCommands));
        long startedNanos = System.nanoTime();
        TypeSafeHttpResponse response = sendWithRetry(payload);
        long latencyMillis = Duration.ofNanos(System.nanoTime() - startedNanos).toMillis();
        TypeSafeDecision decision = parseDecision(response.body(), optionCommands);

        String selectedCommand = optionCommands.get(decision.option());
        if (selectedCommand == null || !legalMoves.contains(selectedCommand)) {
            throw new IllegalStateException("TypeSafe returned an option outside the legal-move list");
        }

        turnHistory.add(new TurnObservation(
                turnHistory.size() + 1,
                historicalObservation,
                normalizeFeedback(feedback),
                selectedCommand,
                makesMeasurableProgress(solitaire, selectedCommand)));
        totalInputTokens += decision.inputTokens();
        totalOutputTokens += decision.outputTokens();
        totalLatencyMillis += latencyMillis;
        lastDecisionMetadata = decisionMetadata(
                decision, selectedCommand, latencyMillis, promptSet);

        if (log.isDebugEnabled()) {
            log.debug(
                    "TypeSafe decision turn={} model={} command={} confidence={} latencyMillis={} inputTokens={}",
                    turnHistory.size(),
                    decision.model(),
                    selectedCommand,
                    decision.confidence(),
                    latencyMillis,
                    decision.inputTokens());
        }
        return selectedCommand;
    }

    /** Builds stable opaque option ids so Jev cannot invent an engine command. */
    static Map<String, String> optionCommands(List<String> legalMoves) {
        Map<String, String> options = new LinkedHashMap<>();
        for (int index = 0; index < legalMoves.size(); index++) {
            options.put("move_" + index, legalMoves.get(index));
        }
        return options;
    }

    /** Builds the documented System One request with structured state and one Choice question. */
    Map<String, Object> buildRequest(
            Solitaire solitaire,
            Map<String, Object> currentBoard,
            String feedback,
            Map<String, String> optionCommands) {
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("state", buildState(currentBoard, feedback));
        request.put("model", modelName);

        Map<String, Object> criteria = new LinkedHashMap<>();
        List<String> requiredProgressCommands = requiredProgressCommands(solitaire, optionCommands.values());
        optionCommands.forEach((option, command) -> criteria.put(
                option,
                commandDescription(command, solitaire, requiredProgressCommands.contains(command))));

        Map<String, Object> question = new LinkedHashMap<>();
        question.put("type", "choice");
        question.put("instructions", promptSet.moveSelectionInstructions());
        question.put("criteria", criteria);
        request.put("questions", Map.of("next_move", question));
        return request;
    }

    /** Builds temporal state containing everything shown to the player on earlier turns. */
    private Map<String, Object> buildState(
            Map<String, Object> currentBoard, String feedback) {
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("game", "Klondike Solitaire");
        state.put("variant", "draw three with unlimited stock passes");
        state.put("observation_format", observationFormat());
        if (promptSet.profile().equals("detailed")) {
            state.put("rules", promptSet.rules());
            state.put("decision_priorities", promptSet.decisionPriorities());
            state.put("progress_summary", progressSummary());
        }
        state.put("previous_turns", turnHistory.stream()
                .map(TurnObservation::asState)
                .toList());

        Map<String, Object> currentTurn = new LinkedHashMap<>();
        currentTurn.put("turn", turnHistory.size() + 1);
        currentTurn.put("observed_board", currentBoard);
        String normalizedFeedback = normalizeFeedback(feedback);
        if (!normalizedFeedback.isBlank()) {
            currentTurn.put("engine_feedback", normalizedFeedback);
        }
        state.put("current_turn", currentTurn);
        return state;
    }

    /** Summarizes whether the game is advancing or cycling before the next P1 decision. */
    private Map<String, Object> progressSummary() {
        Map<String, Object> progress = new LinkedHashMap<>();
        progress.put("consecutive_stock_turns", consecutiveStockTurns());
        progress.put("turns_since_foundation_or_hidden_card_progress", turnsSinceMeasurableProgress());
        progress.put("instruction", GamePrompts.STALL_INSTRUCTION);
        return progress;
    }

    /** Captures exactly the board information visible in the human console. */
    private static Map<String, Object> observedBoard(Solitaire solitaire) {
        Map<String, Object> board = new LinkedHashMap<>();
        board.put("foundations", foundationState(solitaire));
        board.put("tableau", tableauState(solitaire));
        board.put("stock", Map.of("face_down_count", solitaire.getStockpile().size()));

        Map<String, Object> waste = new LinkedHashMap<>();
        waste.put("card_count", solitaire.getTalon().size());
        waste.put(
                "visible_top",
                solitaire.getTalon().isEmpty()
                        ? null
                        : solitaire.getTalon().getLast().shortName());
        board.put("waste", waste);
        return board;
    }

    /** Compresses a completed turn to the visible facts useful for future card tracking. */
    private static Map<String, Object> historicalObservation(Solitaire solitaire) {
        Map<String, Object> observation = new LinkedHashMap<>();
        observation.put("foundation_tops", solitaire.getFoundation().stream()
                .map(cards -> cards.isEmpty() ? null : cards.getLast().shortName())
                .toList());
        observation.put("tableau_tops", solitaire.getVisibleTableau().stream()
                .map(cards -> cards.isEmpty() ? null : cards.getLast().shortName())
                .toList());
        observation.put("tableau_face_down_counts", solitaire.getTableauFaceDownCounts());
        observation.put("stock_count", solitaire.getStockpile().size());
        observation.put("waste_count", solitaire.getTalon().size());
        observation.put(
                "waste_visible_top",
                solitaire.getTalon().isEmpty()
                        ? null
                        : solitaire.getTalon().getLast().shortName());
        return observation;
    }

    /** Represents each tableau pile using its command name, hidden count, and visible suffix. */
    private static Map<String, Object> tableauState(Solitaire solitaire) {
        List<List<Card>> visible = solitaire.getVisibleTableau();
        List<Integer> hidden = solitaire.getTableauFaceDownCounts();
        Map<String, Object> piles = new LinkedHashMap<>();
        for (int index = 0; index < visible.size(); index++) {
            Map<String, Object> pile = new LinkedHashMap<>();
            pile.put("face_down_count", hidden.get(index));
            pile.put("face_up_bottom_to_top", cardNames(visible.get(index)));
            piles.put("T" + (index + 1), pile);
        }
        return piles;
    }

    /** Represents each foundation by the top card visible on the console. */
    private static Map<String, Object> foundationState(Solitaire solitaire) {
        Map<String, Object> foundations = new LinkedHashMap<>();
        List<List<Card>> cards = solitaire.getFoundation();
        for (int index = 0; index < cards.size(); index++) {
            List<Card> foundation = cards.get(index);
            Map<String, Object> pile = new LinkedHashMap<>();
            pile.put("top", foundation.isEmpty() ? null : foundation.getLast().shortName());
            foundations.put("F" + (index + 1), pile);
        }
        return foundations;
    }

    /** Explains the board notation once so every historical snapshot stays compact. */
    private static Map<String, Object> observationFormat() {
        Map<String, Object> format = new LinkedHashMap<>();
        format.put("card_format", "Rank followed by suit symbol, for example Q♠ or 10♦.");
        format.put(
                "tableau_order",
                "face_up_bottom_to_top is ordered from the bottom of the visible stack to its exposed top card.");
        format.put("hidden_cards", "Only face_down_count is observable; identities remain unknown.");
        format.put("waste_visibility", "Only visible_top is playable and currently visible.");
        format.put("empty_pile", "An empty visible stack is [] and a missing top card is null.");
        format.put(
                "previous_turns",
                "Compact completed turns use F for foundation tops in F1-F4 order, T for tableau tops "
                        + "in T1-T7 order, hidden for tableau face-down counts, stock for face-down stock "
                        + "count, and waste as [card count, visible top].");
        return format;
    }

    /** Normalizes optional feedback before it becomes durable game history. */
    private static String normalizeFeedback(String feedback) {
        return feedback == null ? "" : feedback.trim();
    }

    /** Converts cards to their stable uncoloured short names. */
    private static List<String> cardNames(List<Card> cards) {
        return cards.stream().map(Card::shortName).toList();
    }

    /** Gives each opaque option command semantics and, for P1, simulated strategic consequences. */
    private Object commandDescription(
            String command, Solitaire solitaire, boolean requiredByProgressRule) {
        Map<String, Object> description = new LinkedHashMap<>();
        description.put("command", command);
        description.put("effect", GamePrompts.commandEffect(command));
        if (promptSet.profile().equals("detailed")) {
            Map<String, Object> assessment = strategicAssessment(command, solitaire);
            assessment.put("required_by_progress_rule", requiredByProgressRule);
            description.put("strategic_assessment", assessment);
        }
        return description;
    }

    /** Simulates a legal command and exposes objective progress and repetition signals to Jev. */
    private Map<String, Object> strategicAssessment(String command, Solitaire solitaire) {
        Map<String, Object> assessment = new LinkedHashMap<>();
        if (command.equals("quit")) {
            assessment.put("ends_game_without_winning", true);
            assessment.put("recommendation", "Avoid while any non-quit option can change the position.");
            return assessment;
        }

        int hiddenBefore = faceDownCount(solitaire);
        int foundationBefore = foundationCount(solitaire);
        Solitaire resultingState = solitaire.copy();
        applyCommand(resultingState, command);
        int hiddenDelta = hiddenBefore - faceDownCount(resultingState);
        int foundationDelta = foundationCount(resultingState) - foundationBefore;
        Map<String, Object> resultingObservation = historicalObservation(resultingState);
        long priorOccurrences = turnHistory.stream()
                .filter(turn -> turn.observedBoard().equals(resultingObservation))
                .count();
        boolean returnsToCurrent = historicalObservation(solitaire).equals(resultingObservation);

        assessment.put("face_down_cards_revealed", hiddenDelta);
        assessment.put("foundation_card_delta", foundationDelta);
        assessment.put("resulting_position_seen_count", priorOccurrences);
        assessment.put("already_seen", priorOccurrences > 0 || returnsToCurrent);
        assessment.put("moves_waste_card", command.startsWith("move W"));
        assessment.put("moves_tableau_cards", command.startsWith("move T"));
        assessment.put("turns_stock", command.equals("turn"));
        assessment.put("consecutive_stock_turns_before_move", consecutiveStockTurns());
        assessment.put("turns_since_measurable_progress", turnsSinceMeasurableProgress());
        assessment.put(
                "recommendation",
                GamePrompts.moveRecommendation(
                        command,
                        hiddenDelta,
                        foundationDelta,
                        priorOccurrences,
                        returnsToCurrent,
                        consecutiveStockTurns(),
                        turnsSinceMeasurableProgress()));
        return assessment;
    }

    /** Selects all highest-priority immediate-progress commands without hiding other legal options. */
    private static List<String> requiredProgressCommands(
            Solitaire solitaire, Collection<String> commands) {
        Map<String, SimulatedProgress> progressByCommand = new LinkedHashMap<>();
        for (String command : commands) {
            if (!command.equals("quit")) {
                progressByCommand.put(command, simulateProgress(solitaire, command));
            }
        }

        int greatestReveal = progressByCommand.values().stream()
                .mapToInt(SimulatedProgress::hiddenCardsRevealed)
                .max()
                .orElse(0);
        if (greatestReveal > 0) {
            return progressByCommand.entrySet().stream()
                    .filter(entry -> entry.getValue().hiddenCardsRevealed() == greatestReveal)
                    .map(Map.Entry::getKey)
                    .toList();
        }

        int greatestFoundationGain = progressByCommand.values().stream()
                .mapToInt(SimulatedProgress::foundationCardDelta)
                .max()
                .orElse(0);
        if (greatestFoundationGain > 0) {
            return progressByCommand.entrySet().stream()
                    .filter(entry -> entry.getValue().foundationCardDelta() == greatestFoundationGain)
                    .map(Map.Entry::getKey)
                    .toList();
        }
        return List.of();
    }

    /** Applies one engine-generated command to a lookahead copy. */
    private static void applyCommand(Solitaire solitaire, String command) {
        if (command.equals("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = command.split(" ");
        if (parts.length == 4) {
            solitaire.moveCard(parts[1], parts[2], parts[3]);
        } else if (parts.length == 3) {
            solitaire.moveCard(parts[1], null, parts[2]);
        }
    }

    /** Counts hidden tableau cards for one-step progress evaluation. */
    private static int faceDownCount(Solitaire solitaire) {
        return solitaire.getTableauFaceDownCounts().stream().mapToInt(Integer::intValue).sum();
    }

    /** Counts foundation cards for one-step progress evaluation. */
    private static int foundationCount(Solitaire solitaire) {
        return solitaire.getFoundation().stream().mapToInt(List::size).sum();
    }

    /** Determines whether a selected command reveals a hidden card or grows a foundation. */
    private static boolean makesMeasurableProgress(Solitaire solitaire, String command) {
        if (command.equals("quit")) {
            return false;
        }
        SimulatedProgress progress = simulateProgress(solitaire, command);
        return progress.hiddenCardsRevealed() > 0 || progress.foundationCardDelta() > 0;
    }

    /** Measures the two forms of immediate progress enforced by P1.5. */
    private static SimulatedProgress simulateProgress(Solitaire solitaire, String command) {
        int hiddenBefore = faceDownCount(solitaire);
        int foundationBefore = foundationCount(solitaire);
        Solitaire resultingState = solitaire.copy();
        applyCommand(resultingState, command);
        return new SimulatedProgress(
                hiddenBefore - faceDownCount(resultingState),
                foundationCount(resultingState) - foundationBefore);
    }

    /** Counts consecutive stock turns at the end of the completed history. */
    private int consecutiveStockTurns() {
        int count = 0;
        for (int index = turnHistory.size() - 1; index >= 0; index--) {
            if (!turnHistory.get(index).selectedCommand().equals("turn")) {
                break;
            }
            count++;
        }
        return count;
    }

    /** Counts decisions since a hidden-card reveal or positive foundation move. */
    private int turnsSinceMeasurableProgress() {
        int count = 0;
        for (int index = turnHistory.size() - 1; index >= 0; index--) {
            if (turnHistory.get(index).measurableProgress()) {
                break;
            }
            count++;
        }
        return count;
    }

    /** Serializes the request without ever including the API key. */
    private static String serializeRequest(Map<String, Object> request) {
        try {
            return JSON.writeValueAsString(request);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Could not serialize TypeSafe request", exception);
        }
    }

    /** Sends a request and retries rate limits, server failures, and transport interruptions. */
    private TypeSafeHttpResponse sendWithRetry(String payload) {
        RuntimeException lastFailure = null;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                TypeSafeHttpResponse response = transport.send(payload);
                if (response.statusCode() >= 200 && response.statusCode() < 300) {
                    return response;
                }
                if (!isTransientStatus(response.statusCode()) || attempt == maxAttempts) {
                    throw new IllegalStateException(
                            "TypeSafe API returned HTTP " + response.statusCode() + ": " + response.body());
                }
                lastFailure = new IllegalStateException("TypeSafe API returned HTTP " + response.statusCode());
                sleepBeforeRetry(attempt, response.retryAfterMillis());
            } catch (IOException exception) {
                lastFailure = new IllegalStateException("TypeSafe API transport failed", exception);
                if (attempt == maxAttempts) {
                    throw lastFailure;
                }
                sleepBeforeRetry(attempt, 0L);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted while calling TypeSafe API", exception);
            }
        }
        throw lastFailure == null ? new IllegalStateException("TypeSafe API request failed") : lastFailure;
    }

    /** Sleeps using Retry-After when present, otherwise exponential backoff. */
    private void sleepBeforeRetry(int attempt, long retryAfterMillis) {
        long delay = retryAfterMillis > 0
                ? retryAfterMillis
                : initialRetryMillis * (1L << Math.min(attempt - 1, 10));
        log.warn("Transient TypeSafe API failure on attempt {}/{}; retrying in {} ms", attempt, maxAttempts, delay);
        try {
            Thread.sleep(delay);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting to retry TypeSafe API", exception);
        }
    }

    /** Identifies response codes that may succeed without changing the request. */
    static boolean isTransientStatus(int statusCode) {
        return statusCode == 429 || statusCode >= 500;
    }

    /** Parses and validates Jev's typed Choice response. */
    static TypeSafeDecision parseDecision(String body, Map<String, String> optionCommands) {
        try {
            JsonNode root = JSON.readTree(body);
            JsonNode answer = root.path("answers").path("next_move");
            String option = answer.path("choice").asText("");
            if (!optionCommands.containsKey(option)) {
                throw new IllegalStateException("TypeSafe response selected an unknown option: " + option);
            }

            Map<String, Double> probabilities = new LinkedHashMap<>();
            answer.path("probabilities").properties().forEach(
                    entry -> probabilities.put(entry.getKey(), entry.getValue().asDouble()));
            return new TypeSafeDecision(
                    root.path("model").asText("unknown"),
                    option,
                    answer.path("confidence").asDouble(),
                    probabilities,
                    root.path("usage").path("input_tokens").asLong(),
                    root.path("usage").path("output_tokens").asLong());
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Could not parse TypeSafe response: " + body, exception);
        }
    }

    /** Captures the model decision in the corresponding episode step. */
    private static Map<String, Object> decisionMetadata(
            TypeSafeDecision decision,
            String selectedCommand,
            long latencyMillis,
            GamePrompts.PromptSet promptSet) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("provider", "TypeSafe AI");
        metadata.put("model", decision.model());
        metadata.put("prompt_profile", promptSet.profile());
        metadata.put("prompt_version", promptSet.version());
        metadata.put("policy_version", promptSet.version());
        metadata.put("option", decision.option());
        metadata.put("command", selectedCommand);
        metadata.put("confidence", decision.confidence());
        metadata.put("probabilities", decision.probabilities());
        metadata.put("latency_millis", latencyMillis);
        metadata.put("input_tokens", decision.inputTokens());
        metadata.put("output_tokens", decision.outputTokens());
        return metadata;
    }

    /** Resolves the key while preserving the existing property-over-environment convention. */
    private static String resolveApiKey(String propertyValue) {
        if (propertyValue != null && !propertyValue.isBlank()) {
            return propertyValue.trim();
        }
        String environmentValue = System.getenv("TYPESAFE_API_KEY");
        if (environmentValue != null && !environmentValue.isBlank()) {
            return environmentValue.trim();
        }
        throw new IllegalStateException(
                "TypeSafe API key must be set via 'typesafe.apiKey' or TYPESAFE_API_KEY");
    }

    /** Returns immutable metadata for the command most recently selected. */
    @Override
    public synchronized Map<String, Object> getLastDecisionMetadata() {
        return lastDecisionMetadata;
    }

    /** Describes the game-scoped Jev configuration and cumulative API usage. */
    @Override
    public synchronized Map<String, Object> getExperimentMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("provider", "TypeSafe AI");
        metadata.put("model", modelName);
        metadata.put("decision_api", "System One Choice");
        metadata.put("prompt_profile", promptSet.profile());
        metadata.put("prompt_version", promptSet.version());
        metadata.put("policy_version", promptSet.version());
        metadata.put("stateful", true);
        metadata.put("state_transport", "compact observed turn history in each request");
        metadata.put("observed_turns", turnHistory.size());
        metadata.put("total_input_tokens", totalInputTokens);
        metadata.put("total_output_tokens", totalOutputTokens);
        metadata.put("total_latency_millis", totalLatencyMillis);
        return metadata;
    }

    /** Returns cumulative input usage for result reporting. */
    public synchronized long getTotalInputTokens() {
        return totalInputTokens;
    }

    /** Returns cumulative output usage for result reporting. */
    public synchronized long getTotalOutputTokens() {
        return totalOutputTokens;
    }

    /** Returns cumulative HTTP latency for result reporting. */
    public synchronized long getTotalLatencyMillis() {
        return totalLatencyMillis;
    }

    /** Minimal transport abstraction used to keep API parsing tests in-process. */
    interface TypeSafeTransport {
        TypeSafeHttpResponse send(String payload) throws IOException, InterruptedException;
    }

    /** HTTP result with the retry delay already normalized to milliseconds. */
    record TypeSafeHttpResponse(int statusCode, String body, long retryAfterMillis) {}

    /** Immediate hidden-card and foundation effects of one legal lookahead command. */
    private record SimulatedProgress(int hiddenCardsRevealed, int foundationCardDelta) {}

    /** Human-visible state and selected action retained for one completed turn. */
    private record TurnObservation(
            int turn,
            Map<String, Object> observedBoard,
            String engineFeedback,
            String selectedCommand,
            boolean measurableProgress) {

        /** Converts the record to stable API field names and omits empty feedback. */
        private Map<String, Object> asState() {
            Map<String, Object> state = new LinkedHashMap<>();
            state.put("turn", turn);
            state.put("F", observedBoard.get("foundation_tops"));
            state.put("T", observedBoard.get("tableau_tops"));
            state.put("hidden", observedBoard.get("tableau_face_down_counts"));
            state.put("stock", observedBoard.get("stock_count"));
            state.put(
                    "waste",
                    List.of(
                            observedBoard.get("waste_count"),
                            observedBoard.get("waste_visible_top") == null
                                    ? "--"
                                    : observedBoard.get("waste_visible_top")));
            if (!engineFeedback.isBlank()) {
                state.put("feedback", engineFeedback);
            }
            state.put("command", selectedCommand);
            if (measurableProgress) {
                state.put("progress", true);
            }
            return state;
        }
    }

    /** Parsed typed decision returned by the System One endpoint. */
    record TypeSafeDecision(
            String model,
            String option,
            double confidence,
            Map<String, Double> probabilities,
            long inputTokens,
            long outputTokens) {}

    /** JDK HTTP implementation so the integration adds no new runtime dependency. */
    private static final class JdkTypeSafeTransport implements TypeSafeTransport {
        private final URI endpoint;
        private final String apiKey;
        private final Duration timeout;
        private final HttpClient client;

        private JdkTypeSafeTransport(String baseUrl, String apiKey, Duration timeout) {
            this.endpoint = URI.create(baseUrl);
            this.apiKey = apiKey;
            this.timeout = timeout;
            this.client = HttpClient.newBuilder().connectTimeout(timeout).build();
        }

        /** Executes one authenticated JSON request and extracts an optional Retry-After value. */
        @Override
        public TypeSafeHttpResponse send(String payload) throws IOException, InterruptedException {
            HttpRequest request = HttpRequest.newBuilder(endpoint)
                    .timeout(timeout)
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload))
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            long retryAfterMillis = response.headers()
                    .firstValue("retry-after")
                    .map(JdkTypeSafeTransport::parseRetryAfterMillis)
                    .orElse(0L);
            return new TypeSafeHttpResponse(response.statusCode(), response.body(), retryAfterMillis);
        }

        /** Parses the API's seconds-based Retry-After header. */
        private static long parseRetryAfterMillis(String value) {
            try {
                return (long) Math.ceil(Double.parseDouble(value) * 1000.0);
            } catch (NumberFormatException exception) {
                return 0L;
            }
        }
    }
}

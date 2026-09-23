package ai.games.player.ai;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Optional;
import java.util.Set;

/**
 * Metadata for OpenAI models that we benchmark through the API or Codex CLI.
 */
public enum OpenAIModelInfo {

    GPT_6_ASTRA("gpt-6-astra", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "low"),
    GPT_6_SOL("gpt-6-sol", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "medium"),
    GPT_6_LUNA("gpt-6-luna", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "low"),
    GPT_5_6_SOL("gpt-5.6-sol", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "medium"),
    GPT_5_6_TERRA("gpt-5.6-terra", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "medium"),
    GPT_5_6_LUNA("gpt-5.6-luna", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "low"),
    GPT_5_5("gpt-5.5", EnumSet.of(AccessPath.API, AccessPath.CODEX_CLI), "medium"),

    GPT_5_1("gpt-5.1", AccessPath.API),
    GPT_5_MINI("gpt-5-mini", AccessPath.API),
    GPT_5_NANO("gpt-5-nano", AccessPath.API),

    GPT_4_1("gpt-4.1", AccessPath.API),
    GPT_4_1_MINI("gpt-4.1-mini", AccessPath.API),
    GPT_4_1_NANO("gpt-4.1-nano", AccessPath.API),

    O3("o3", AccessPath.API),
    O4_MINI("o4-mini", AccessPath.API),

    GPT_4O("gpt-4o", AccessPath.API),
    GPT_4O_REALTIME_PREVIEW("gpt-4o-realtime-preview", AccessPath.API);

    private final String modelName;
    private final Set<AccessPath> accessPaths;
    private final String recommendedCodexReasoningEffort;

    OpenAIModelInfo(String modelName, AccessPath accessPath) {
        this(modelName, EnumSet.of(accessPath), null);
    }

    OpenAIModelInfo(String modelName, AccessPath accessPath, String recommendedCodexReasoningEffort) {
        this(modelName, EnumSet.of(accessPath), recommendedCodexReasoningEffort);
    }

    OpenAIModelInfo(
            String modelName, Set<AccessPath> accessPaths, String recommendedCodexReasoningEffort) {
        this.modelName = modelName;
        this.accessPaths = Set.copyOf(accessPaths);
        this.recommendedCodexReasoningEffort = recommendedCodexReasoningEffort;
    }

    public String getModelName() {
        return modelName;
    }

    public boolean supports(AccessPath accessPath) {
        return accessPaths.contains(accessPath);
    }

    public Optional<String> getRecommendedCodexReasoningEffort() {
        return Optional.ofNullable(recommendedCodexReasoningEffort);
    }

    public static Optional<OpenAIModelInfo> byModelName(String modelName) {
        return Arrays.stream(values())
                .filter(info -> info.modelName.equals(modelName))
                .findFirst();
    }

    public enum AccessPath {
        API,
        CODEX_CLI
    }
}

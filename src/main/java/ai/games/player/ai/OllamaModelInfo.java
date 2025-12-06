package ai.games.player.ai;

import java.util.Arrays;
import java.util.Optional;

/**
 * Metadata for Ollama-backed models that we benchmark.
 *
 * Centralising this information lets tests and docs consistently refer
 * to provider, player label, model name, and reference URL. Additional
 * fields like size and context are informational only and may be empty
 * where not known or not applicable.
 */
public enum OllamaModelInfo {

    GPT_OSS_120B(
            "OpenAIPlayer",
            "OpenAI",
            "gpt-oss:120b",
            "https://ollama.com/library/gpt-oss",
            "65GB",
            "128K"),

    LLAMA4_SCOUT(
            "MetaPlayer",
            "Meta",
            "llama4:scout",
            "https://ollama.com/library/llama4",
            null,
            null),

    GEMMA3_27B(
            "GooglePlayer",
            "Google",
            "gemma3:27b",
            "https://ollama.com/library/gemma3",
            "17GB",
            "128K"),

    QWEN3_30B(
            "AlibabaPlayer",
            "Alibaba",
            "qwen3-coder:30b",
            "https://ollama.com/library/qwen3-coder",
            null,
            null),

    MISTRAL_LARGE_123B(
            "MistralPlayer",
            "Mistral",
            "mistral-large:123b",
            "https://ollama.com/library/mistral-large",
            null,
            null),

    DEEPSEEK_R1_70B(
            "DeepSeekPlayer",
            "DeepSeek",
            "deepseek-r1:70b",
            "https://ollama.com/library/deepseek-r1:70b",
            "43GB",
            "128K");

    private final String playerName;
    private final String provider;
    private final String modelName;
    private final String url;
    private final String size;
    private final String context;

    OllamaModelInfo(String playerName, String provider, String modelName, String url, String size, String context) {
        this.playerName = playerName;
        this.provider = provider;
        this.modelName = modelName;
        this.url = url;
        this.size = size;
        this.context = context;
    }

    public String getPlayerName() {
        return playerName;
    }

    public String getProvider() {
        return provider;
    }

    public String getModelName() {
        return modelName;
    }

    public String getUrl() {
        return url;
    }

    /**
     * Approximate model size as reported by Ollama (e.g., "65GB").
     */
    public String getSize() {
        return size;
    }

    /**
     * Context window as reported by Ollama (e.g., "128K").
     */
    public String getContext() {
        return context;
    }

    /**
     * User-facing label for the "Algorithm" column in results tables.
     */
    public String algorithmLabel() {
        return playerName + " (" + modelName + ")";
    }

    /**
     * Exact lookup by fully-qualified model name (e.g. {@code gpt-oss:120b}).
     */
    public static Optional<OllamaModelInfo> byModelName(String modelName) {
        return Arrays.stream(values())
                .filter(info -> info.modelName.equals(modelName))
                .findFirst();
    }

    /**
     * Best-effort lookup that allows for model variants that share a common base
     * name (e.g. {@code qwen3}, {@code qwen3:30b}, {@code qwen3-coder:30b}).
     *
     * <p>If no exact match exists, this method compares the base name segment
     * (substring before the first {@code ':'}) in a case-insensitive way.</p>
     */
    public static Optional<OllamaModelInfo> byModelNameOrBase(String modelName) {
        if (modelName == null || modelName.isBlank()) {
            return Optional.empty();
        }
        Optional<OllamaModelInfo> exact = byModelName(modelName);
        if (exact.isPresent()) {
            return exact;
        }
        String requestedBase = baseName(modelName);
        return Arrays.stream(values())
                .filter(info -> baseName(info.modelName).equalsIgnoreCase(requestedBase))
                .findFirst();
    }

    /**
     * Infer a provider label for an arbitrary Ollama model name. If the model
     * is not explicitly listed in this enum, this still attempts to match the
     * base name to one of the known providers (e.g. {@code gpt-oss}, {@code llama4},
     * {@code gemma3}, {@code qwen3}, {@code mistral}, {@code deepseek-r1}).
     *
     * <p>Falls back to the generic label {@code "Ollama"} if no match is found.</p>
     */
    public static String inferProvider(String modelName) {
        return byModelNameOrBase(modelName)
                .map(OllamaModelInfo::getProvider)
                .orElse("Ollama");
    }

    private static String baseName(String modelName) {
        int idx = modelName.indexOf(':');
        return idx >= 0 ? modelName.substring(0, idx) : modelName;
    }
}

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

    MISTRAL_LATEST(
            "MistralPlayer",
            "Mistral",
            "mistral:latest",
            "https://ollama.com/library/mistral",
            "4.4GB",
            "32K"),

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

    public static Optional<OllamaModelInfo> byModelName(String modelName) {
        return Arrays.stream(values())
                .filter(info -> info.modelName.equals(modelName))
                .findFirst();
    }
}

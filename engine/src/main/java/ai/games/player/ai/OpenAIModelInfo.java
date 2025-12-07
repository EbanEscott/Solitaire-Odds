package ai.games.player.ai;

import java.util.Arrays;
import java.util.Optional;

/**
 * Metadata for OpenAI chat models that we benchmark.
 *
 * Encodes the model identifier and approximate rate limits so that tests and
 * docs can reason about throughput and contention.
 */
public enum OpenAIModelInfo {

    GPT_5_1("gpt-5.1", 500_000, 500),
    GPT_5_MINI("gpt-5-mini", 500_000, 500),
    GPT_5_NANO("gpt-5-nano", 200_000, 500),

    GPT_4_1("gpt-4.1", 30_000, 500),
    GPT_4_1_MINI("gpt-4.1-mini", 200_000, 500),
    GPT_4_1_NANO("gpt-4.1-nano", 200_000, 500),

    O3("o3", 30_000, 500),
    O4_MINI("o4-mini", 200_000, 500),

    GPT_4O("gpt-4o", 30_000, 500),
    GPT_4O_REALTIME_PREVIEW("gpt-4o-realtime-preview", 40_000, 200);

    private final String modelName;
    private final int tokensPerMinute;
    private final int requestsPerMinute;

    OpenAIModelInfo(String modelName, int tokensPerMinute, int requestsPerMinute) {
        this.modelName = modelName;
        this.tokensPerMinute = tokensPerMinute;
        this.requestsPerMinute = requestsPerMinute;
    }

    public String getModelName() {
        return modelName;
    }

    public int getTokensPerMinute() {
        return tokensPerMinute;
    }

    public int getRequestsPerMinute() {
        return requestsPerMinute;
    }

    public static Optional<OpenAIModelInfo> byModelName(String modelName) {
        return Arrays.stream(values())
                .filter(info -> info.modelName.equals(modelName))
                .findFirst();
    }
}


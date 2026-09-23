package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.player.ai.OpenAIModelInfo.AccessPath;
import org.junit.jupiter.api.Test;

class OpenAIModelInfoTest {

    @Test
    void identifiesCodexSubscriptionModels() {
        assertTrue(OpenAIModelInfo.GPT_6_SOL.supports(AccessPath.CODEX_CLI));
        assertTrue(OpenAIModelInfo.GPT_6_ASTRA.supports(AccessPath.CODEX_CLI));
        assertTrue(OpenAIModelInfo.GPT_6_SOL.supports(AccessPath.API));
    }

    @Test
    void keepsApiOnlyModelsSeparate() {
        assertTrue(OpenAIModelInfo.GPT_5_MINI.supports(AccessPath.API));
        assertFalse(OpenAIModelInfo.GPT_5_MINI.supports(AccessPath.CODEX_CLI));
    }

    @Test
    void exposesRecommendedCodexReasoningEffort() {
        assertEquals("low", OpenAIModelInfo.GPT_6_ASTRA.getRecommendedCodexReasoningEffort().orElseThrow());
    }
}

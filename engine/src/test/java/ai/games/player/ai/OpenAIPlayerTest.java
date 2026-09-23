package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class OpenAIPlayerTest {

    @AfterEach
    void clearModelProperty() {
        System.clearProperty("openai.model");
    }

    @Test
    void usesDefaultModelWhenPropertyIsAbsent() {
        System.clearProperty("openai.model");

        assertEquals("gpt-4o", OpenAIPlayer.configuredModelName());
    }

    @Test
    void usesConfiguredModelFromSystemProperty() {
        System.setProperty("openai.model", "gpt-5-mini");

        assertEquals("gpt-5-mini", OpenAIPlayer.configuredModelName());
    }
}

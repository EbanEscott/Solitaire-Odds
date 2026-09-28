package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class GamePromptsTest {

    @AfterEach
    void clearPromptProfile() {
        System.clearProperty(GamePrompts.PROMPT_PROFILE_PROPERTY);
        System.clearProperty(GamePrompts.LEGACY_PROMPT_PROFILE_PROPERTY);
    }

    @Test
    void defaultsToFrozenP0Profile() {
        GamePrompts.PromptSet promptSet = GamePrompts.configuredPromptSet();

        assertEquals("p0", promptSet.profile());
        assertEquals("P0", promptSet.version());
        assertTrue(promptSet.strategyPrompt().contains("strategy you already know"));
        assertFalse(promptSet.strategyPrompt().contains("decision order"));
    }

    @Test
    void resolvesDetailedProfileByNameOrVersion() {
        System.setProperty(GamePrompts.PROMPT_PROFILE_PROPERTY, "detailed");
        GamePrompts.PromptSet named = GamePrompts.promptSet("detailed");
        GamePrompts.PromptSet versioned = GamePrompts.promptSet("P1.5");

        assertEquals(named, GamePrompts.configuredPromptSet());
        assertEquals(named, versioned);
        assertEquals("detailed", named.profile());
        assertEquals("P1.5", named.version());
        assertTrue(named.strategyPrompt().contains("unlimited passes"));
        assertTrue(named.strategyPrompt().contains("decision order"));
        assertFalse(named.strategyPrompt().contains("turn impossible"));
        assertTrue(named.gameInterface().contains("exhaustive"));
        assertEquals("Build four same-suit foundations from Ace through King.",
                named.rules().get("objective"));
        assertEquals(8, named.decisionPriorities().size());
        assertTrue(named.moveSelectionInstructions().contains("required_by_progress_rule"));
    }

    @Test
    void acceptsLegacyLlmPropertyForExistingExperimentCommands() {
        System.setProperty(GamePrompts.LEGACY_PROMPT_PROFILE_PROPERTY, "detailed");

        assertEquals("P1.5", GamePrompts.configuredPromptSet().version());
    }

    @Test
    void usesTheInterfaceFromTheSelectedProfile() {
        Solitaire solitaire = new Solitaire(new Deck());
        String prompt = GamePrompts.buildTurnPrompt(
                solitaire,
                "",
                List.of("turn"),
                GamePrompts.promptSet("detailed"),
                true);

        assertTrue(prompt.contains("engine-validated legal-move list"));
        assertTrue(prompt.contains("# Current board"));
    }

    @Test
    void rejectsUnknownPromptProfiles() {
        assertThrows(
                IllegalArgumentException.class,
                () -> GamePrompts.promptSet("experimental"));
    }
}

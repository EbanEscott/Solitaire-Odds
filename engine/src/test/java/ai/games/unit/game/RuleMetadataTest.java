package ai.games.unit.game;

import ai.games.EpisodeLogger;
import ai.games.Game;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.MockedStatic;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Verifies that benchmark and episode records retain the actual game's rule configuration. */
class RuleMetadataTest {
    /**
     * Supplies every independent rules combination.
     *
     * @return all twenty-four supported configurations
     */
    static Stream<GameRules> rules() {
        return Stream.of(1, 3).flatMap(draw -> Stream.of(GameRules.StockVisibility.values())
                .flatMap(visibility -> Stream.of(1, 3, null)
                        .flatMap(passes -> Stream.of(false, true)
                                .map(thoughtful -> new GameRules(draw, visibility, passes, thoughtful)))));
    }

    /**
     * Checks startup logging without episodes, then the structured episode payloads.
     *
     * @param rules the explicit rules, independent of JVM defaults
     * @throws Exception if a logged JSON payload cannot be parsed
     */
    @ParameterizedTest
    @MethodSource("rules")
    void recordsActualRules(GameRules rules) throws Exception {
        Logger gameLogger = (Logger) LoggerFactory.getLogger(Game.class);
        Logger episodeLogger = (Logger) LoggerFactory.getLogger(EpisodeLogger.class);
        ListAppender<ILoggingEvent> events = new ListAppender<>();
        events.start();
        gameLogger.addAppender(events);
        episodeLogger.addAppender(events);
        try (MockedStatic<EpisodeLogger> mocked = mockStatic(EpisodeLogger.class, CALLS_REAL_METHODS)) {
            Solitaire state = new Solitaire(new Deck(), rules);
            Player player = mock(Player.class);
            when(player.nextCommand(any(), anyString(), anyString())).thenReturn("turn", "quit");
            Game game = new Game(player);
            game.setGuidanceEnabled(false);
            mocked.when(EpisodeLogger::isEnabled).thenReturn(false);
            game.play(state);
            String startup = events.list.stream().map(ILoggingEvent::getFormattedMessage)
                    .filter(line -> line.startsWith("GAME_RULES ")).findFirst().orElseThrow();
            assertTrue(startup.contains("thoughtful=" + rules.thoughtful()));
            assertTrue(startup.contains("drawCount=" + rules.drawCount()));
            assertTrue(startup.contains("stockVisibility=" + rules.stockVisibility()));
            assertTrue(startup.contains("maxStockPasses=" + rules.maxStockPasses()));
            assertTrue(events.list.stream().noneMatch(event -> event.getFormattedMessage().startsWith("EPISODE_")));

            events.list.clear();
            when(player.nextCommand(any(), anyString(), anyString())).thenReturn("turn", "quit");
            mocked.when(EpisodeLogger::isEnabled).thenReturn(true);
            game.play(new Solitaire(new Deck(), rules));
            ObjectMapper json = new ObjectMapper();
            for (String prefix : List.of("EPISODE_GAME ", "EPISODE_STEP ", "EPISODE_SUMMARY ")) {
                String line = events.list.stream().map(ILoggingEvent::getFormattedMessage)
                        .filter(message -> message.startsWith(prefix)).findFirst().orElseThrow();
                JsonNode record = json.readTree(line.substring(prefix.length()));
                JsonNode payload = prefix.equals("EPISODE_GAME ") ? record.get("metadata") : record;
                assertEquals(json.valueToTree(rules), payload.get("game_rules"));
                assertEquals(1, payload.get("stock_pass").asInt());
            }

            // A summary of a later pass must retain its actual pass counter.
            while (!state.getStockpile().isEmpty()) {
                state.turnStock();
            }
            state.turnStock();
            events.list.clear();
            EpisodeLogger.logStep(state, state, "test", 1, List.of(), "quit", Map.of());
            EpisodeLogger.logSummary(state, "test", 1, 0, false, 0, "quit", "quit");
            for (ILoggingEvent event : events.list) {
                String line = event.getFormattedMessage();
                JsonNode record = json.readTree(line.substring(line.indexOf(' ') + 1));
                assertEquals(state.getStockPass(), record.get("stock_pass").asInt());
                if (record.get("type").asText().equals("step")) {
                    Solitaire observation = state.planningCopy();
                    assertEquals(json.valueToTree(observation.getStockpile().reversed().stream()
                            .map(ai.games.game.Card::shortName).toList()), record.get("known_stock_next_draw_first"));
                    assertEquals(json.valueToTree(observation.getTableau().stream()
                            .map(pile -> pile.stream().map(ai.games.game.Card::shortName).toList()).toList()),
                            record.get("known_tableau_bottom_to_top"));
                }
            }
        } finally {
            gameLogger.detachAppender(events);
            episodeLogger.detachAppender(events);
            events.stop();
        }
    }
}

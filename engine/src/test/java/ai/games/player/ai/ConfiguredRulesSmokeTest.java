package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import ai.games.player.ai.astar.AStarPlayer;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

/** Bounded local smoke runs for every search player and rules configuration; no win-rate claims. */
class ConfiguredRulesSmokeTest {
    /** Twelve decisions exercise repeated observation and prediction validation without a result sweep. */
    private static final int DECISIONS = 12;

    /**
     * Enumerates the six local players against all twenty-four independent rules combinations.
     *
     * @return named player/configuration pairs for independently reported smoke tests
     */
    static Stream<Arguments> cases() {
        List<Arguments> cases = new ArrayList<>();
        for (String player : List.of("rule", "greedy", "hill", "beam", "monteCarlo", "astar")) {
            for (int draw : new int[]{1, 3}) {
                for (var visibility : GameRules.StockVisibility.values()) {
                    for (Integer passes : new Integer[]{1, 3, null}) {
                        for (boolean thoughtful : new boolean[]{false, true}) {
                            cases.add(Arguments.of(player, new GameRules(draw, visibility, passes, thoughtful)));
                        }
                    }
                }
            }
        }
        return cases.stream();
    }

    @ParameterizedTest(name = "{0}: {1}")
    @MethodSource("cases")
    void decisionsStayLegalAndLeaveRealStateUntouched(String name, GameRules rules) {
        GreedySearchPlayer.resetForNewGame();
        Player player = switch (name) {
            case "rule" -> new RuleBasedHeuristicsPlayer();
            case "greedy" -> new GreedySearchPlayer();
            case "hill" -> new HillClimberPlayer(42);
            case "beam" -> new BeamSearchPlayer(3, 8, 42);
            case "monteCarlo" -> new MonteCarloPlayer();
            case "astar" -> new AStarPlayer();
            default -> throw new IllegalArgumentException(name);
        };
        List<Card> deck = new ArrayList<>(SolitaireBuilder.fullDeckInStableOrder());
        Collections.shuffle(deck, new Random(20261017));
        Solitaire game = new Solitaire(new Deck(deck), rules);
        // Begin after a complete traversal to exercise waste memory and the recycle boundary.
        while (!game.getStockpile().isEmpty()) assertTrue(game.turnStock());
        for (int decision = 0; decision < DECISIONS; decision++) {
            long key = game.getStateKey();
            String observation = game.toString();
            int history = game.getMoveHistorySize();
            long pass = game.getStockPass();
            List<String> legal = LegalMovesHelper.listLegalMoves(game);
            String command = player.nextCommand(game, "", "");
            assertEquals(key, game.getStateKey(), "Search changed the true board");
            assertEquals(observation, game.toString(), "Search changed actual knowledge");
            assertEquals(history, game.getMoveHistorySize(), "Search changed actual history");
            assertEquals(pass, game.getStockPass(), "Search consumed a real stock pass");
            assertNotNull(command);
            if (command.equals("quit")) break;
            if (command.equals("turn")) {
                assertTrue(legal.contains(command), "Stock turn was not listed as legal");
                assertTrue(game.turnStock());
            } else {
                // Players use both explicit-card commands and the engine's top-card shorthand.
                String[] parts = command.split(" ");
                assertEquals("move", parts[0]);
                assertTrue(parts.length == 3 || parts.length == 4, "Malformed command: " + command);
                assertTrue(game.moveCard(parts[1], parts.length == 4 ? parts[2] : null,
                        parts[parts.length - 1]), "Rejected player move: " + command);
            }
            SolitaireBuilder.assertValidGameState(game);
            if (rules.maxStockPasses() != null) assertTrue(game.getStockPass() <= rules.maxStockPasses());
        }
        GreedySearchPlayer.resetForNewGame();
    }
}

package ai.games.results;

import ai.games.Game;
import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import ai.games.player.ai.*;
import ai.games.player.ai.astar.AStarPlayer;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Opt-in search-only benchmark using identical recorded deals for every selected player. */
@EnabledIfSystemProperty(named = "search.sweep", matches = "true")
class SearchSweepTest {
    /** Supported local players; no external services or model-backed players are selected. */
    private static final Set<String> PLAYERS = Set.of("rule", "greedy", "hill", "beam", "monteCarlo", "astar");

    /**
     * Runs the configured matched-deal sweep and flushes each completed game to disk.
     *
     * @throws Exception if configuration, output, or a player command is invalid
     */
    @Test
    void matchedDeals() throws Exception {
        int games = Integer.getInteger("test.games", 100);
        int cap = Integer.getInteger("test.max.moves.per.game", 1000);
        long seedStart = Long.getLong("search.seedStart", 0);
        long playerSeed = Long.getLong("search.playerSeed", 20261017);
        List<String> players = List.of(System.getProperty("search.players", "rule,greedy,hill,beam,monteCarlo,astar").split(","));
        assertTrue(games > 0 && cap > 0);
        assertEquals(players.size(), new HashSet<>(players).size(), "Duplicate player");
        assertTrue(PLAYERS.containsAll(players), "Unknown player");
        Math.addExact(seedStart, games - 1L);
        GameRules rules = GameRules.configured();
        Path output = Path.of(System.getProperty("search.output"));
        Files.createDirectories(output);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("rules", rules);
        metadata.put("gamesPerPlayer", games);
        metadata.put("players", players);
        metadata.put("seedStart", seedStart);
        metadata.put("playerSeed", playerSeed);
        metadata.put("maxIterations", cap);
        metadata.put("guidance", false);
        metadata.put("training", false);
        metadata.put("javaVersion", System.getProperty("java.version"));
        metadata.put("beamDepth", 2);
        metadata.put("beamWidth", 8);
        metadata.put("monteCarloSamplesPerCandidate", 64);
        metadata.put("monteCarloSearchVisits", Integer.getInteger("montecarlo.search.visits", 128));
        metadata.put("monteCarloEvaluator", System.getProperty("montecarlo.evaluator", "dfs"));
        metadata.put("astarNodeBudget", Integer.getInteger("astar.nodeBudget", 1024));
        metadata.put("astarMaxSearchNodes", Integer.getInteger("astar.maxSearchNodes", 5000));
        metadata.put("astarKnownFirstBudget", Integer.getInteger("astar.knownFirstBudget",
                Integer.getInteger("astar.nodeBudget", 1024) / 2));
        metadata.put("hillMaxSteps", 5000);
        metadata.put("hillMaxSidewaysSteps", 8);
        metadata.put("hillPerturbation", "one random eligible neighbor after sideways preference");
        metadata.put("commandValidationIncludedInTiming", true);
        Map<String, String> overrides = new TreeMap<>();
        System.getProperties().stringPropertyNames().stream()
                .filter(key -> key.startsWith("astar.") || key.startsWith("montecarlo."))
                .forEach(key -> overrides.put(key, System.getProperty(key)));
        metadata.put("searchOverrides", overrides);
        Files.writeString(output.resolve("engine-metadata.json"),
                new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(metadata), StandardOpenOption.CREATE_NEW);
        String previousCap = System.getProperty("max.moves.per.game");
        String previousIndex = System.getProperty("game.index");
        String previousTotal = System.getProperty("game.total");
        System.setProperty("max.moves.per.game", Integer.toString(cap));
        System.setProperty("game.total", Integer.toString(games));
        try (var rows = Files.newBufferedWriter(output.resolve("games.csv"), StandardOpenOption.CREATE_NEW);
             var deals = Files.newBufferedWriter(output.resolve("deals.csv"), StandardOpenOption.CREATE_NEW)) {
            rows.write("player,game,deal_seed,player_seed,won,moves,termination,foundation,seconds\n");
            deals.write("game,seed,deck_bottom_to_top\n");
            for (int index = 0; index < games; index++) {
                long seed = seedStart + index;
                List<Card> cards = new ArrayList<>(SolitaireBuilder.fullDeckInStableOrder());
                Collections.shuffle(cards, new Random(seed));
                deals.write((index + 1) + "," + seed + "," + String.join(" ", cards.stream().map(Card::shortName).toList()) + "\n");
                deals.flush();
                System.setProperty("game.index", Integer.toString(index + 1));
                // Rotate player order to distribute JVM warm-up effects across the batch.
                for (int offset = 0; offset < players.size(); offset++) {
                    String name = players.get((offset + index) % players.size());
                    GreedySearchPlayer.resetForNewGame();
                    long randomSeed = Math.addExact(playerSeed, index);
                    Player player = player(name, randomSeed);
                    Solitaire board = new Solitaire(new Deck(cards), rules);
                    System.out.printf("SWEEP_START player=%s class=%s game=%d deal_seed=%d player_seed=%d%n",
                            name, player.getClass().getName(), index + 1, seed, randomSeed);
                    Game game = new Game(new CheckedPlayer(player));
                    game.setGuidanceEnabled(false);
                    Game.GameResult result = game.play(board);
                    SolitaireBuilder.assertValidGameState(board);
                    assertTrue(result.getMoves() <= cap);
                    rows.write(String.format(Locale.ROOT, "%s,%d,%d,%d,%s,%d,%s,%d,%.9f%n",
                            name, index + 1, seed, randomSeed, result.isWon(), result.getMoves(),
                            result.getTerminationReason(), board.getFoundation().stream().mapToInt(List::size).sum(),
                            result.getDurationNanos() / 1e9));
                    rows.flush();
                    System.out.printf(Locale.ROOT, "SWEEP %s %d/%d %s moves=%d seconds=%.3f%n",
                            name, index + 1, games, result.getTerminationReason(), result.getMoves(), result.getDurationNanos() / 1e9);
                }
            }
        } finally {
            restore("max.moves.per.game", previousCap);
            restore("game.index", previousIndex);
            restore("game.total", previousTotal);
        }
    }

    /**
     * Creates a fresh local player with reproducible sampling where supported.
     *
     * @param name the selected player identifier
     * @param seed randomness independent of the deal shuffle
     * @return the player for one game
     */
    private static Player player(String name, long seed) {
        return switch (name) {
            case "rule" -> new RuleBasedHeuristicsPlayer();
            case "greedy" -> new GreedySearchPlayer();
            case "hill" -> new HillClimberPlayer(seed);
            case "beam" -> new BeamSearchPlayer(2, 8, seed);
            case "monteCarlo" -> new MonteCarloPlayer(new Random(seed));
            case "astar" -> new AStarPlayer();
            default -> throw new IllegalArgumentException(name);
        };
    }

    /**
     * Restores a JVM property after the sweep.
     *
     * @param key the property name
     * @param value the previous value, or null when absent
     */
    private static void restore(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }

    /** Fails the sweep on an illegal command instead of silently consuming the iteration cap. */
    private static final class CheckedPlayer extends AIPlayer {
        /** The actual search player, kept separate for each deal. */
        private final Player delegate;

        /**
         * Wraps a local player for command validation.
         *
         * @param delegate the player whose decisions are checked
         */
        private CheckedPlayer(Player delegate) {
            this.delegate = delegate;
        }

        @Override
        public String nextCommand(Solitaire board, String moves, String feedback) {
            String command = delegate.nextCommand(board, moves, feedback);
            assertNotNull(command, "Search player closed input");
            if (command.equals("quit")) return command;
            Solitaire validation = board.copy();
            if (command.equals("turn")) assertTrue(validation.turnStock(), "Illegal stock turn");
            else {
                String[] parts = command.split("\\s+");
                assertTrue(parts.length == 3 || parts.length == 4, command);
                assertEquals("move", parts[0]);
                assertTrue(validation.moveCard(parts[1], parts.length == 4 ? parts[2] : null,
                        parts[parts.length - 1]), command);
            }
            return command;
        }
    }
}

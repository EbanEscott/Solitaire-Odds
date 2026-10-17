package ai.games.experiments;

import ai.games.game.*;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Opt-in matched-deal Monte Carlo research runner with saved decks and seeded sampling.
 * Results are development artifacts, not updates to the canonical player table.
 */
@EnabledIfSystemProperty(named = "montecarlo.screen", matches = "true")
class MonteCarloScreenTest {
    @Test
    void screen() throws Exception {
        Path output = Path.of(System.getProperty("montecarlo.output"));
        Files.createDirectories(output);
        Path manifest = Path.of(System.getProperty("montecarlo.deals", output.resolve("deals.csv").toString()));
        int games = Integer.getInteger("montecarlo.games", 20);
        int firstSeed = Integer.getInteger("montecarlo.seedStart", 0);
        long samplingSeed = Long.getLong("montecarlo.samplingSeed", 4242L);
        int cap = Integer.getInteger("montecarlo.moveCap", 1000);
        String[] variants = System.getProperty("montecarlo.variants", "baseline,order,visits,transpositions").split(",");
        Map<Integer, List<Card>> deals = loadDeals(manifest, firstSeed, games);
        String[] properties = {"montecarlo.evaluator", "montecarlo.search.visits",
                "montecarlo.search.revealFirst", "montecarlo.search.transpositions", "montecarlo.search.reuseKnownSolution", "montecarlo.search.safeFoundations",
                "montecarlo.search.cutoffReward"};
        Map<String, String> saved = new HashMap<>();
        for (String property : properties) saved.put(property, System.getProperty(property));
        try (var rows = Files.newBufferedWriter(output.resolve("games.csv"))) {
            rows.write("variant,seed,sampling_seed,won,moves,foundation,termination,seconds\n"); rows.flush();
            for (int seed = firstSeed; seed < firstSeed + games; seed++) {
                List<Card> cards = Objects.requireNonNull(deals.get(seed), "Missing saved deal " + seed);
                for (int offset = 0; offset < variants.length; offset++) {
                    String variant = variants[(offset + seed - firstSeed) % variants.length];
                    configure(variant);
                    Solitaire game = new Solitaire(new Deck(cards));
                    MonteCarloPlayer player = new MonteCarloPlayer(new Random(samplingSeed));
                    long start = System.nanoTime();
                    int moves = 0;
                    String termination = "cap";
                    Path trace = output.resolve(variant + "-" + seed + ".commands.txt");
                    System.out.println("MC start: " + variant + " seed=" + seed);
                    try (var commands = Files.newBufferedWriter(trace)) {
                        while (moves < cap) {
                            if (foundations(game) == 52) { termination = "win"; break; }
                            String command = player.nextCommand(game, "", "");
                            commands.write(command + "\n"); commands.flush();
                            if (command.equals("quit")) { termination = "quit"; break; }
                            assertTrue(LegalMovesHelper.listLegalMoves(game).contains(command), command);
                            Move move = Objects.requireNonNull(Move.tryParse(command));
                            if (move.isTurn()) game.turnThree();
                            else assertTrue(game.moveCard(move.from().toCode(),
                                    move.card() == null ? null : move.card().shortName(), move.to().toCode()), command);
                            moves++;
                        }
                    }
                    boolean won = foundations(game) == 52;
                    if (won) termination = "win";
                    double seconds = (System.nanoTime() - start) / 1e9;
                    rows.write(String.format(Locale.ROOT, "%s,%d,%d,%s,%d,%d,%s,%.6f%n",
                            variant, seed, samplingSeed, won, moves, foundations(game), termination, seconds));
                    rows.flush();
                    System.out.printf(Locale.ROOT, "MC result: %s seed=%d won=%s moves=%d seconds=%.3f%n",
                            variant, seed, won, moves, seconds);
                }
            }
        } finally {
            for (String property : properties) {
                if (saved.get(property) == null) System.clearProperty(property);
                else System.setProperty(property, saved.get(property));
            }
        }
    }

    private static void configure(String variant) {
        if (!Set.of("baseline", "order", "visits", "transpositions", "order-visits",
                "order-transpositions", "visits-transpositions", "order-visits-transpositions",
                "solution", "order-solution", "visits-solution", "transpositions-solution",
                "dominance", "dominance-order", "dominance-visits", "dominance-transpositions", "dominance-solution",
                "order-cutoff0", "order-cutoff25").contains(variant))
            throw new IllegalArgumentException("Unknown variant: " + variant);
        System.setProperty("montecarlo.search.cutoffReward",
                variant.contains("cutoff0") ? "0.0" : variant.contains("cutoff25") ? "0.25" : "0.5");
        System.setProperty("montecarlo.search.safeFoundations", Boolean.toString(variant.contains("dominance")));
        System.setProperty("montecarlo.search.reuseKnownSolution", Boolean.toString(variant.contains("solution")));
        System.setProperty("montecarlo.evaluator", "dfs");
        System.setProperty("montecarlo.search.visits", variant.contains("visits") ? "1000" : "128");
        System.setProperty("montecarlo.search.revealFirst", Boolean.toString(variant.contains("order")));
        System.setProperty("montecarlo.search.transpositions", Boolean.toString(variant.contains("transpositions")));
    }

    private static Map<Integer, List<Card>> loadDeals(Path path, int first, int count) throws Exception {
        List<Card> canonical = new ArrayList<>();
        for (Suit suit : Suit.values()) if (suit != Suit.UNKNOWN)
            for (Rank rank : Rank.values()) if (rank != Rank.UNKNOWN) canonical.add(new Card(rank, suit));
        if (!Files.exists(path)) {
            Files.createDirectories(path.toAbsolutePath().getParent());
            try (var out = Files.newBufferedWriter(path, StandardOpenOption.CREATE_NEW)) {
                out.write("seed,deck_bottom_to_top\n");
                for (int seed = first; seed < first + count; seed++) {
                    List<Card> cards = new ArrayList<>(canonical);
                    Collections.shuffle(cards, new Random(seed));
                    out.write(seed + "," + String.join(" ", cards.stream().map(Card::shortName).toList()) + "\n");
                }
            }
        }
        Map<String, Card> byName = new HashMap<>();
        for (Card card : canonical) byName.put(card.shortName(), card);
        Map<Integer, List<Card>> result = new LinkedHashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (String line : lines.subList(1, lines.size())) {
            String[] row = line.split(",", 2);
            List<Card> cards = Arrays.stream(row[1].split(" ")).map(byName::get).toList();
            assertEquals(52, cards.size()); assertFalse(cards.contains(null));
            assertEquals(52, new HashSet<>(cards).size());
            assertNull(result.put(Integer.parseInt(row[0]), cards), "Duplicate deal identifier");
        }
        return result;
    }

    private static int foundations(Solitaire board) {
        return board.getFoundation().stream().mapToInt(List::size).sum();
    }
}

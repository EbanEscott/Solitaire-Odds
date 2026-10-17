package ai.games.experiments;

import ai.games.game.*;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.astar.AStarPlayer;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Opt-in matched-deal development screen; never publishes canonical results. */
@EnabledIfSystemProperty(named = "astar.screen", matches = "true")
class AStarScreenTest {
    @Test
    void screen() throws Exception {
        Path output = Path.of(System.getProperty("astar.output"));
        Files.createDirectories(output);
        int games = Integer.getInteger("astar.games", 20);
        int firstSeed = Integer.getInteger("astar.seedStart", 0);
        String[] variants = System.getProperty("astar.variants", "baseline,foundation,stock,reveal").split(",");
        Set<String> supported = Set.of("balanced", "reuse", "baseline", "foundation", "stock", "reveal", "lowcost", "deeper",
                "finish", "hidden4", "hidden8", "noempty", "transpositions", "failure200", "failure400", "hill", "undeployed", "transpositions4x", "scan");
        assertTrue(games > 0, "astar.games must be positive");
        assertEquals(variants.length, new HashSet<>(Arrays.asList(variants)).size(), "Duplicate variants");
        for (String variant : variants) assertTrue(supported.contains(variant), "Unknown variant: " + variant);
        try (var rows = Files.newBufferedWriter(output.resolve("games.csv"), StandardOpenOption.CREATE_NEW);
             var deals = Files.newBufferedWriter(output.resolve("deals.csv"), StandardOpenOption.CREATE_NEW)) {
            rows.write("variant,seed,won,moves,foundation,termination,seconds\n");
            deals.write("seed,deck_bottom_to_top\n");
            for (int seed = firstSeed; seed < firstSeed + games; seed++) {
                List<Card> cards = new ArrayList<>();
                for (Suit suit : Suit.values()) if (suit != Suit.UNKNOWN)
                    for (Rank rank : Rank.values()) if (rank != Rank.UNKNOWN)
                        cards.add(new Card(rank, suit));
                Collections.shuffle(cards, new Random(seed));
                deals.write(seed + "," + String.join(" ", cards.stream().map(Card::shortName).toList()) + "\n");
                // Rotate execution order to reduce systematic warm-up/order effects.
                for (int i = 0; i < variants.length; i++) {
                    String variant = variants[(i + seed - firstSeed) % variants.length];
                    System.setProperty("astar.foundationWeight", variant.equals("hill") ? "3.0" : variant.equals("foundation") ? "2.0" : "1.0");
                    System.setProperty("astar.stockWeight", variant.equals("hill") ? "1.2" : variant.equals("undeployed") ? "2.0" : variant.equals("stock") ? "0.0" : "0.5");
                    System.setProperty("astar.knownFirstBudget", variant.equals("reveal") ? "0" : (variant.equals("deeper") || variant.equals("transpositions4x")) ? "2048" : "512");
                    System.setProperty("astar.scanStock", Boolean.toString(variant.equals("scan")));
                    System.setProperty("astar.talonWeight", variant.equals("hill") ? "1.0" : variant.equals("undeployed") ? "2.0" : "0.0");
                    System.setProperty("astar.failureCost", variant.equals("failure200") ? "200.0" : variant.equals("failure400") ? "400.0" : "1000.0");
                    System.setProperty("astar.balancedSearch", Boolean.toString(variant.equals("balanced")));
                    System.setProperty("astar.reuseDuplicates", Boolean.toString(variant.equals("reuse")));
                    System.setProperty("astar.transpositions", Boolean.toString(variant.equals("balanced") || variant.equals("reuse") || variant.equals("transpositions") || variant.equals("transpositions4x") || variant.equals("scan")));
                    System.setProperty("astar.hiddenWeight", variant.equals("hill") ? "1.6" : variant.equals("hidden4") ? "4.0" : variant.equals("hidden8") ? "8.0" : "2.0");
                    System.setProperty("astar.emptyWeight", variant.equals("noempty") || variant.equals("hill") ? "0.0" : "3.0");
                    System.setProperty("astar.finishKnownWin", Boolean.toString(variant.equals("finish")));
                    System.setProperty("astar.moveCost", variant.equals("lowcost") ? "0.1" : "1.0");
                    System.setProperty("astar.nodeBudget", (variant.equals("deeper") || variant.equals("transpositions4x")) ? "4096" : "1024");
                    System.setProperty("astar.maxSearchNodes", (variant.equals("deeper") || variant.equals("transpositions4x")) ? "20000" : "5000");
                    Solitaire game = new Solitaire(new Deck(cards));
                    AStarPlayer player = new AStarPlayer();
                    List<String> trace = new ArrayList<>();
                    long start = System.nanoTime();
                    int moves = 0;
                    String termination = "cap";
                    while (moves < 1000) {
                        if (foundations(game) == 52) { termination = "win"; break; }
                        if (Boolean.getBoolean("astar.trace")) trace.add("STEP " + moves + "\n" + game.toString());
                        String command = player.nextCommand(game, "", "");
                        if (Boolean.getBoolean("astar.trace")) trace.add(player.getDecisionTrace() + "\nSELECT " + command);
                        if (command.equals("quit")) { termination = "quit"; break; }
                        assertTrue(LegalMovesHelper.listLegalMoves(game).contains(command), command);
                        if (command.equals("turn")) game.turnThree();
                        else {
                            String[] parts = command.split("\\s+");
                            assertTrue(game.moveCard(parts[1], parts.length == 4 ? parts[2] : null,
                                    parts[parts.length - 1]), command);
                        }
                        moves++;
                    }
                    if (Boolean.getBoolean("astar.trace")) Files.write(output.resolve(variant + "-" + seed + ".trace"), trace);
                    boolean won = foundations(game) == 52;
                    if (won) termination = "win";
                    rows.write(String.format(Locale.ROOT, "%s,%d,%s,%d,%d,%s,%.6f%n", variant,
                            seed, won, moves, foundations(game), termination, (System.nanoTime() - start) / 1e9));
                    rows.flush();
                }
                deals.flush();
                System.out.println("A* matched screen: completed seed " + seed);
            }
        } finally {
            System.clearProperty("astar.scanStock");
            System.clearProperty("astar.talonWeight");
            System.clearProperty("astar.failureCost");
            System.clearProperty("astar.balancedSearch");
            System.clearProperty("astar.reuseDuplicates");
            System.clearProperty("astar.transpositions");
            System.clearProperty("astar.hiddenWeight");
            System.clearProperty("astar.emptyWeight");
            System.clearProperty("astar.finishKnownWin");
            System.clearProperty("astar.moveCost");
            System.clearProperty("astar.nodeBudget");
            System.clearProperty("astar.maxSearchNodes");
            System.clearProperty("astar.foundationWeight");
            System.clearProperty("astar.stockWeight");
            System.clearProperty("astar.knownFirstBudget");
        }
    }
    private static int foundations(Solitaire game) {
        return game.getFoundation().stream().mapToInt(List::size).sum();
    }
}

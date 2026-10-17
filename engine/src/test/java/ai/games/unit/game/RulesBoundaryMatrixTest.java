package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.GameRules;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

/** Cross-checks rule boundaries using a complete deck with an exact five-card stock. */
class RulesBoundaryMatrixTest {
    /**
     * Enumerates independent draw, observation, and pass settings.
     *
     * @return every supported rules configuration
     */
    static Stream<GameRules> rules() {
        List<GameRules> rules = new ArrayList<>();
        for (int draw : new int[]{1, 3}) {
            for (var visibility : GameRules.StockVisibility.values()) {
                for (Integer passes : new Integer[]{1, 3, null}) {
                    for (boolean thoughtful : new boolean[]{false, true}) {
                        rules.add(new GameRules(draw, visibility, passes, thoughtful));
                    }
                }
            }
        }
        return rules.stream();
    }

    @ParameterizedTest
    @MethodSource("rules")
    void partialDrawsPassBoundariesCopiesAndWorryBack(GameRules rules) {
        Set<String> specified = Set.of("A♣", "2♣", "3♣", "4♣", "5♣", "2♥");
        String[] remaining = SolitaireBuilder.fullDeckInStableOrder().stream().map(Card::shortName)
                .filter(card -> !specified.contains(card)).toArray(String[]::new);
        Solitaire game = SolitaireBuilder.newGame().stock("A♣", "2♣", "3♣", "4♣", "5♣")
                .tableau("T1", "2♥").tableau("T7", 1, remaining).build(rules);
        List<Card> expected = List.copyOf(game.getStockpile().reversed());
        int completedPasses = rules.maxStockPasses() == null ? 4 : rules.maxStockPasses();
        for (int pass = 1; pass <= completedPasses; pass++) {
            int drawn = 0;
            do {
                int count = Math.min(rules.drawCount(), expected.size() - drawn);
                assertTrue(game.turnStock());
                drawn += count;
                assertEquals(pass, game.getStockPass());
                assertEquals(expected.subList(0, drawn), game.getTalon());
                assertEquals(expected.get(drawn - 1), game.planningCopy().getTalon().getLast());
            } while (!game.getStockpile().isEmpty());
        }
        Solitaire plan = game.planningCopy();
        long unknownWaste = plan.getTalon().stream().filter(c -> c.getRank() == Rank.UNKNOWN).count();
        assertEquals(!rules.thoughtful() && rules.drawCount() == 3 && rules.stockVisibility() == GameRules.StockVisibility.topOnly
                ? 3 : 0, unknownWaste);
        for (Solitaire copy : List.of(game.copy(), plan.copy())) {
            long before = copy.getStateKey();
            if (rules.maxStockPasses() != null) {
                assertFalse(copy.turnStock());
                assertFalse(LegalMovesHelper.listLegalMoves(copy).contains("turn"));
                assertEquals(0, copy.getMoveHistorySize());
            }
            // Pass exhaustion does not end play, and worry-back remains legal.
            assertTrue(copy.moveCard("W", "F1"));
            if (copy.nextUnresolvedCard() != null) {
                copy.assumeCard(copy.nextUnresolvedCard(), new Card(Rank.TWO, Suit.CLUBS));
            }
            assertTrue(copy.moveCard("F1", "T1"));
            copy.rollbackTo(0);
            assertEquals(before, copy.getStateKey());
            assertEquals(completedPasses, copy.getStockPass());
        }
        assertEquals(expected, game.getTalon());
        SolitaireBuilder.assertValidGameState(game);
    }
}

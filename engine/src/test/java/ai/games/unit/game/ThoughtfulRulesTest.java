package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.TypeSafePlayer;
import ai.games.player.ai.alpha.AlphaSolitaireRequest;
import ai.games.unit.helpers.SolitaireBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

/** Verifies full knowledge independently of face-up status and legal access. */
class ThoughtfulRulesTest {
    /**
     * Supplies the complete configuration matrix.
     *
     * @return all twenty-four combinations
     */
    static Stream<GameRules> rules() {
        return RulesBoundaryMatrixTest.rules();
    }

    @Test
    void validatesBooleanWithoutChangingLegacyDefaults() {
        Properties properties = new Properties();
        assertFalse(GameRules.fromProperties(properties).thoughtful());
        assertEquals(GameRules.DEFAULT, new GameRules(3, GameRules.StockVisibility.topOnly, null));
        properties.setProperty("game.thoughtful", "true");
        assertTrue(GameRules.fromProperties(properties).thoughtful());
        properties.setProperty("game.thoughtful", "false");
        assertEquals(GameRules.DEFAULT, GameRules.fromProperties(properties));
        for (String invalid : List.of("", "null", "1", "yes", "TRUE", " false")) {
            properties.setProperty("game.thoughtful", invalid);
            assertThrows(IllegalArgumentException.class, () -> GameRules.fromProperties(properties));
        }
    }

    /**
     * Checks legal access, full knowledge, and rollback on the same physical deal.
     *
     * @param rules the configuration under test
     */
    @ParameterizedTest
    @MethodSource("rules")
    void knowledgeDoesNotChangeLegalAccessAndSurvivesRollback(GameRules rules) {
        List<Card> deck = SolitaireBuilder.fullDeckInStableOrder();
        Solitaire game = new Solitaire(new Deck(deck), rules);
        Solitaire counterpart = new Solitaire(new Deck(deck), new GameRules(rules.drawCount(),
                rules.stockVisibility(), rules.maxStockPasses(), !rules.thoughtful()));
        assertEquals(game.getTableau(), counterpart.getTableau());
        assertEquals(game.getTableauFaceDownCounts(), counterpart.getTableauFaceDownCounts());
        assertNotEquals(game.getStateKey(), counterpart.getStateKey());
        assertFalse(game.matchesObservation(counterpart));
        assertEquals(rules.thoughtful() ? 0 : 45, game.getUnknownCards().size());
        Solitaire plan = game.planningCopy();
        assertEquals(game.getRules(), plan.getRules());
        assertEquals(game.getTableauFaceDownCounts(), plan.getTableauFaceDownCounts());
        if (rules.thoughtful()) {
            assertFullKnowledge(game, plan);
            long key = plan.getStateKey();
            plan.sampleUnknownCards(new Random(42));
            assertEquals(key, plan.getStateKey());
            assertEquals(0, plan.getMoveHistorySize());
        } else {
            assertTrue(plan.getStockpile().stream().allMatch(card -> card.getRank() == Rank.UNKNOWN));
        }
        // Known face-down and stock cards still cannot be moved directly.
        assertFalse(game.moveCard("T7", game.getTableau().get(6).getFirst().shortName(), "F1"));
        assertFalse(game.moveCard("S", game.getStockpile().getLast().shortName(), "F1"));
        int draws = 24 / rules.drawCount();
        for (int action = 0; action < draws * 3 + 1; action++) {
            assertEquals(LegalMovesHelper.listLegalMoves(game), LegalMovesHelper.listLegalMoves(counterpart));
            assertEquals(LegalMovesHelper.listLegalMoves(game), LegalMovesHelper.listLegalMoves(game.planningCopy()));
            if (!game.canTurnStock()) break;
            assertTrue(game.turnStock());
            assertTrue(counterpart.turnStock());
            if (rules.thoughtful()) assertFullKnowledge(game, game.planningCopy());
        }
        for (Solitaire copy : List.of(game.copy(), game.planningCopy().copy())) {
            assertEquals(rules, copy.getRules());
            long key = copy.getStateKey();
            copy.turnStock();
            copy.rollbackTo(0);
            assertEquals(key, copy.getStateKey());
            if (rules.thoughtful()) assertFullKnowledge(game, copy);
        }
        assertTrue(game.undoTrainingMove());
        if (rules.thoughtful()) assertFullKnowledge(game, game.planningCopy());
        game.rollbackTo(0);
        if (rules.thoughtful()) assertFullKnowledge(game, game.planningCopy());
        assertEquals(1, game.getStockPass());
    }

    @Test
    void exposingAndUndoingTableauRetainsCompleteKnowledge() {
        Solitaire game = SolitaireBuilder.newGame().tableau("T1", 1, "K♣", "A♥")
                .build(new GameRules(3, GameRules.StockVisibility.topOnly, null, true));
        long key = game.getStateKey();
        assertTrue(game.moveCard("T1", "F1"));
        assertFullKnowledge(game, game.planningCopy());
        assertTrue(game.undoLastMove());
        assertEquals(key, game.getStateKey());
        assertEquals(1, game.getTableauFaceDownCounts().getFirst());
        assertTrue(game.moveCard("T1", "F1"));
        assertTrue(game.undoTrainingMove());
        assertEquals(key, game.getStateKey());
        assertFullKnowledge(game, game.planningCopy());
    }

    /**
     * Checks the actual model and neural payloads without contacting a service.
     *
     * @param rules the configuration under test
     * @throws Exception if reflection or JSON conversion fails
     */
    @ParameterizedTest
    @MethodSource("rules")
    void observationsCarryKnownPositionsAndRules(GameRules rules) throws Exception {
        Solitaire game = new Solitaire(new Deck(SolitaireBuilder.fullDeckInStableOrder()), rules);
        Solitaire plan = game.planningCopy();
        AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(game);
        ObjectMapper json = new ObjectMapper();
        assertEquals(rules.thoughtful(), json.valueToTree(request).get("rules").get("thoughtful").asBoolean());
        assertEquals(plan.getUnknownCards().stream().map(Card::shortName).toList(), request.getUnknownCards());
        List<String> stock = plan.getStockpile().reversed().stream().map(Card::shortName).toList();
        List<List<String>> tableau = plan.getTableau().stream()
                .map(pile -> pile.stream().map(Card::shortName).toList()).toList();
        assertEquals(stock, request.getKnownStock());
        assertEquals(tableau, request.getKnownTableau());
        var observedBoard = TypeSafePlayer.class.getDeclaredMethod("observedBoard", Solitaire.class);
        observedBoard.setAccessible(true);
        var board = json.valueToTree(observedBoard.invoke(null, game));
        assertEquals(rules.thoughtful(), board.get("game_rules").get("thoughtful").asBoolean());
        assertEquals(json.valueToTree(stock), board.get("known_stock_next_draw_first"));
        assertEquals(json.valueToTree(tableau), board.get("known_tableau_bottom_to_top"));
        assertTrue(game.toString().contains("thoughtful=" + rules.thoughtful()));
        assertTrue(game.toString().contains("Known stock (next draw first): " + stock));
        assertTrue(game.toString().contains("Known tableau (bottom to top; face-down cards remain unplayable): " + tableau));
    }

    /**
     * Compares exact positions while ensuring knowledge does not alter physical exposure.
     *
     * @param actual the true board
     * @param observation the planning view or copy
     */
    private static void assertFullKnowledge(Solitaire actual, Solitaire observation) {
        assertTrue(actual.getUnknownCards().isEmpty());
        assertTrue(observation.getUnknownCards().isEmpty());
        assertNull(observation.nextUnresolvedCard());
        assertEquals(0, observation.unresolvedObservationCount());
        assertEquals(actual.getTableau(), observation.getTableau());
        assertEquals(actual.getStockpile(), observation.getStockpile());
        assertEquals(actual.getTalon(), observation.getTalon());
        assertEquals(actual.getTableauFaceDownCounts(), observation.getTableauFaceDownCounts());
    }
}

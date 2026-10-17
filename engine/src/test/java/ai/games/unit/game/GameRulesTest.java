package ai.games.unit.game;

import ai.games.game.*;
import ai.games.player.LegalMovesHelper;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GameRulesTest {
    private List<Card> cards() {
        List<Card> cards = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) continue;
            for (Rank rank : Rank.values()) {
                if (rank != Rank.UNKNOWN) cards.add(new Card(rank, suit));
            }
        }
        return cards;
    }

    private Solitaire game(int draw, GameRules.StockVisibility visibility, Integer passes) {
        return new Solitaire(new Deck(cards()), new GameRules(draw, visibility, passes));
    }

    @Test void validatesConfigurationAndDefaults() {
        assertEquals(GameRules.DEFAULT, GameRules.fromProperties(new Properties()));
        Properties properties = new Properties();
        properties.setProperty("game.drawCount", "1");
        properties.setProperty("game.stockVisibility", "allDrawn");
        properties.setProperty("game.maxStockPasses", "3");
        assertEquals(new GameRules(1, GameRules.StockVisibility.allDrawn, 3),
                GameRules.fromProperties(properties));
        for (int invalid : new int[]{0, 2, 4, -1}) {
            assertThrows(IllegalArgumentException.class,
                    () -> new GameRules(invalid, GameRules.StockVisibility.topOnly, null));
            assertThrows(IllegalArgumentException.class,
                    () -> new GameRules(3, GameRules.StockVisibility.topOnly, invalid));
        }
        assertThrows(NullPointerException.class, () -> new GameRules(3, null, null));
        properties.setProperty("game.stockVisibility", "unknown");
        assertThrows(IllegalArgumentException.class, () -> GameRules.fromProperties(properties));
    }

    @Test void drawOrderAndLimitsAcrossAllTwelveConfigurations() {
        for (int draw : new int[]{1, 3}) {
            for (var visibility : GameRules.StockVisibility.values()) {
                for (Integer limit : new Integer[]{1, 3, null}) {
                    Solitaire game = game(draw, visibility, limit);
                    List<Card> order = new ArrayList<>(game.getStockpile());
                    Collections.reverse(order);
                    int passes = limit == null ? 5 : limit;
                    for (int pass = 1; pass <= passes; pass++) {
                        for (int drawn = 0; drawn < 24; drawn += draw) {
                            assertTrue(game.canTurnStock());
                            assertTrue(game.turnStock());
                            assertEquals(pass, game.getStockPass());
                            assertEquals(order.subList(0, drawn + draw), game.getTalon());
                        }
                    }
                    if (limit != null) {
                        long key = game.getStateKey();
                        int depth = game.getMoveHistorySize();
                        assertFalse(game.turnStock());
                        game.turnThree();
                        assertEquals(key, game.getStateKey());
                        assertEquals(depth, game.getMoveHistorySize());
                        for (Solitaire board : List.of(game, game.planningCopy())) {
                            assertFalse(board.canTurnStock());
                            assertFalse(LegalMovesHelper.listLegalMoves(board).contains("turn"));
                            assertFalse(LegalMovesHelper.listMoves(board).contains(Move.turn()));
                        }
                    }
                }
            }
        }
    }

    @Test void partialDrawAndEmptyStock() {
        Solitaire game = new Solitaire(new Deck(cards().subList(0, 30)), GameRules.DEFAULT);
        assertTrue(game.turnStock());
        assertEquals(2, game.getTalon().size());
        assertTrue(game.getStockpile().isEmpty());
        Solitaire empty = new Solitaire(new Deck(cards().subList(0, 28)), GameRules.DEFAULT);
        assertFalse(empty.turnStock());
        assertEquals(0, empty.getMoveHistorySize());
        assertEquals(1, empty.getStockPass());
    }

    @Test void finalPassStillAllowsWasteMovesAndRemovedCardsDoNotRecycle() {
        List<Card> deck = cards();
        int ace = deck.indexOf(new Card(Rank.ACE, Suit.CLUBS));
        Collections.swap(deck, ace, 23); // Last card drawn from the stock is an ace.
        for (Integer limit : new Integer[]{1, 3}) {
            Solitaire game = new Solitaire(new Deck(deck),
                    new GameRules(3, GameRules.StockVisibility.topOnly, limit));
            for (int i = 0; i < 8; i++) game.turnStock();
            List<Card> waste = new ArrayList<>(game.getTalon());
            assertTrue(game.moveCard("W", "F1"));
            waste.removeLast();
            if (limit == 1) {
                assertFalse(game.canTurnStock());
                assertEquals(1, game.getFoundation().getFirst().size());
            } else {
                for (int i = 0; i < 8; i++) assertTrue(game.turnStock());
                assertEquals(waste, game.getTalon());
                assertEquals(2, game.getStockPass());
            }
        }
    }

    @Test void copiesAndRollbackRetainPassAllowance() {
        Solitaire game = game(3, GameRules.StockVisibility.topOnly, 3);
        for (int i = 0; i < 8; i++) game.turnStock();
        List<Card> waste = List.copyOf(game.getTalon());
        long key = game.getStateKey();
        int depth = game.getMoveHistorySize();
        game.turnStock();
        assertEquals(2, game.getStockPass());
        for (Solitaire copy : List.of(game.copy(), game.planningCopy())) {
            assertEquals(game.getRules(), copy.getRules());
            assertEquals(2, copy.getStockPass());
            assertEquals(0, copy.getMoveHistorySize());
            copy.turnStock();
            copy.undoLastMove();
            assertEquals(2, copy.getStockPass());
        }
        game.rollbackTo(depth);
        assertEquals(1, game.getStockPass());
        assertEquals(waste, game.getTalon());
        assertTrue(game.getStockpile().isEmpty());
        assertEquals(key, game.getStateKey());
    }

    @Test void keysAndComparisonsIncludeRulesAndFinitePasses() {
        for (Integer limit : new Integer[]{3, null}) {
            Solitaire game = game(3, GameRules.StockVisibility.topOnly, limit);
            for (int i = 0; i < 8; i++) game.turnStock();
            Solitaire previous = game.copy();
            for (int i = 0; i < 8; i++) game.turnStock();
            assertEquals(previous.getTalon(), game.getTalon());
            assertEquals(limit == null, previous.getStateKey() == game.getStateKey());
            assertEquals(limit == null, previous.matchesObservation(game));
        }
        Solitaire baseline = game(3, GameRules.StockVisibility.topOnly, null);
        for (GameRules rules : List.of(new GameRules(1, GameRules.StockVisibility.topOnly, null),
                new GameRules(3, GameRules.StockVisibility.allDrawn, null),
                new GameRules(3, GameRules.StockVisibility.topOnly, 1))) {
            Solitaire other = new Solitaire(new Deck(cards()), rules);
            assertFalse(baseline.matchesObservation(other));
            assertNotEquals(baseline.getStateKey(), other.getStateKey());
        }
    }
}

package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.BoardFormatter;
import ai.games.game.Deck;
import ai.games.game.GameRules;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import ai.games.player.ai.BeamSearchPlayer;
import ai.games.player.ai.GreedySearchPlayer;
import ai.games.player.ai.HillClimberPlayer;
import ai.games.player.ai.RuleBasedHeuristicsPlayer;
import ai.games.player.ai.TypeSafePlayer;
import ai.games.player.ai.alpha.AlphaSolitaireRequest;
import ai.games.unit.helpers.SolitaireBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Verifies observation boundaries independently of the true hidden arrangement. */
class KnowledgeRulesTest {
    private List<Card> deck() {
        List<Card> cards = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) continue;
            for (Rank rank : Rank.values()) {
                if (rank != Rank.UNKNOWN) cards.add(new Card(rank, suit));
            }
        }
        Collections.shuffle(cards, new Random(9));
        return cards;
    }

    @Test
    void observationCountsRespectAllTwelveConfigurationsAndSurviveRecycling() {
        for (int draw : new int[]{1, 3}) {
            for (var visibility : GameRules.StockVisibility.values()) {
                for (Integer limit : new Integer[]{1, 3, null}) {
                    Solitaire real = new Solitaire(new Deck(deck()), new GameRules(draw, visibility, limit));
                    int seenPerDraw = visibility == GameRules.StockVisibility.allDrawn ? draw : 1;
                    real.turnStock();
                    Solitaire plan = real.planningCopy();
                    assertEquals(draw - seenPerDraw,
                            plan.getTalon().stream().filter(c -> c.getRank() == Rank.UNKNOWN).count());
                    assertEquals(45 - seenPerDraw, plan.getUnknownCards().size());
                    assertEquals(real.getTalon().getLast(), plan.getTalon().getLast());
                    while (!real.getStockpile().isEmpty()) real.turnStock();
                    int unseen = 45 - (24 / draw) * seenPerDraw;
                    assertEquals(unseen, real.getUnknownCards().size());
                    assertEquals(unseen, real.planningCopy().getUnknownCards().size());
                    if (limit == null || limit > 1) {
                        List<Card> observed = real.planningCopy().getTalon().stream()
                                .filter(c -> c.getRank() != Rank.UNKNOWN).toList();
                        real.turnStock();
                        List<Card> known = new ArrayList<>(real.planningCopy().getStockpile());
                        known.addAll(real.planningCopy().getTalon());
                        assertTrue(known.containsAll(observed));
                        assertEquals(unseen, real.getUnknownCards().size());
                    }
                }
            }
        }
    }

    @Test
    void coveredWasteIsNotResolvedUntilExposedAndRollbackRestoresIt() {
        Solitaire real = SolitaireBuilder.newGame().waste("2♣", "A♣").build();
        Solitaire plan = real.planningCopy();
        Card hidden = plan.getTalon().getFirst();
        assertEquals(Rank.UNKNOWN, hidden.getRank());
        assertNull(plan.nextUnresolvedCard());
        assertEquals(0, plan.unresolvedObservationCount());
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(hidden, new Card(Rank.TWO, Suit.CLUBS)));
        long before = plan.getStateKey();
        assertTrue(plan.moveCard("W", "F1"));
        assertSame(hidden, plan.nextUnresolvedCard());
        assertEquals(1, plan.unresolvedObservationCount());
        plan.assumeCard(hidden, new Card(Rank.TWO, Suit.CLUBS));
        assertTrue(plan.moveCard("W", "F1"));
        plan.rollbackTo(0);
        assertEquals(before, plan.getStateKey());
        assertSame(hidden, plan.getTalon().getFirst());
        assertNull(plan.nextUnresolvedCard());
        assertTrue(real.moveCard("W", "F1"));
        assertEquals(new Card(Rank.TWO, Suit.CLUBS), real.planningCopy().getTalon().getLast());
    }

    @Test
    void trainingUndoRemembersWasteAndTableauButSearchRollbackDoesNot() {
        Solitaire real = SolitaireBuilder.newGame().tableau("T1", 1, "K♥", "A♣")
                .waste("3♦", "2♣").build();
        Solitaire untouched = real.copy();
        assertTrue(real.moveCard("T1", "F1"));
        assertTrue(real.moveCard("W", "F1"));
        assertTrue(real.undoTrainingMove());
        assertTrue(real.undoTrainingMove());
        assertEquals(untouched.getTableau(), real.getTableau());
        assertEquals(untouched.getTalon(), real.getTalon());
        Solitaire remembered = real.planningCopy();
        assertEquals(new Card(Rank.KING, Suit.HEARTS), remembered.getTableau().getFirst().getFirst());
        assertEquals(new Card(Rank.THREE, Suit.DIAMONDS), remembered.getTalon().getFirst());
        assertNotEquals(untouched.getStateKey(), real.getStateKey());
        assertFalse(untouched.matchesObservation(real));
        assertFalse(real.matchesObservation(untouched));
        assertEquals(LegalMovesHelper.listLegalMoves(untouched), LegalMovesHelper.listLegalMoves(real));

        long key = untouched.getStateKey();
        untouched.moveCard("T1", "F1");
        untouched.moveCard("W", "F1");
        untouched.rollbackTo(0);
        assertEquals(key, untouched.getStateKey());
        assertEquals(Rank.UNKNOWN, untouched.planningCopy().getTalon().getFirst().getRank());
        assertThrows(IllegalStateException.class, () -> remembered.undoTrainingMove());
    }

    @Test
    void hiddenArrangementsDoNotChangeRenderedOrNeuralObservationsOrSimpleSearchDecisions() throws Exception {
        List<Card> cards = deck();
        Solitaire first = new Solitaire(new Deck(cards), GameRules.DEFAULT);
        Collections.swap(cards, 0, 1); // Covered members of the first draw-three group.
        Collections.swap(cards, 4, 5); // Still in the stock after that draw.
        Solitaire second = new Solitaire(new Deck(cards), GameRules.DEFAULT);
        first.turnStock();
        second.turnStock();
        assertNotEquals(first.getTalon(), second.getTalon());
        assertTrue(first.matchesObservation(second));
        assertTrue(second.matchesObservation(first));
        assertEquals(first.planningCopy().getStateKey(), second.planningCopy().getStateKey());
        assertEquals(first.toString(), second.toString());
        ObjectMapper json = new ObjectMapper();
        assertEquals(json.valueToTree(AlphaSolitaireRequest.fromSolitaire(first)),
                json.valueToTree(AlphaSolitaireRequest.fromSolitaire(second)));
        var observation = TypeSafePlayer.class.getDeclaredMethod("observedBoard", Solitaire.class);
        observation.setAccessible(true);
        assertEquals(observation.invoke(null, first), observation.invoke(null, second));
        List<Supplier<Player>> players = List.of(RuleBasedHeuristicsPlayer::new,
                GreedySearchPlayer::new, () -> new HillClimberPlayer(42),
                () -> new BeamSearchPlayer(3, 8, 42));
        for (Supplier<Player> factory : players) {
            Player left = factory.get();
            Player right = factory.get();
            GreedySearchPlayer.resetForNewGame();
            String a = left.nextCommand(first, "", "");
            GreedySearchPlayer.resetForNewGame();
            String b = right.nextCommand(second, "", "");
            assertEquals(a, b, left.getClass().getSimpleName());
            assertTrue(LegalMovesHelper.listLegalMoves(first).contains(a));
        }
    }

    @Test
    void onlyExposedSlotsAreResolvedUnderEachPolicy() {
        for (var visibility : GameRules.StockVisibility.values()) {
            Solitaire plan = new Solitaire(new Deck(deck()), new GameRules(3, visibility, null)).planningCopy();
            int size = plan.getUnknownCards().size();
            long key = plan.getStateKey();
            plan.turnStock();
            int count = visibility == GameRules.StockVisibility.allDrawn ? 3 : 1;
            assertEquals(count, plan.unresolvedObservationCount());
            for (int i = 0; i < count; i++) {
                plan.assumeCard(plan.nextUnresolvedCard(), plan.getUnknownCards().getFirst());
            }
            assertNull(plan.nextUnresolvedCard());
            assertEquals(size - count, plan.getUnknownCards().size());
            plan.rollbackTo(0);
            assertEquals(key, plan.getStateKey());
            assertEquals(size, plan.getUnknownCards().size());
        }
    }

    @Test
    void allDrawnObservationsExposeCoveredCardsButDoNotMakeThemPlayable() {
        Solitaire real = new Solitaire(new Deck(deck()),
                new GameRules(3, GameRules.StockVisibility.allDrawn, 3));
        real.turnStock();
        AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(real);
        assertEquals(real.getTalon().stream().map(Card::shortName).toList(), request.getTalon());
        assertEquals(real.getRules(), request.getRules());
        assertEquals(1, request.getStockPass());
        assertEquals(42, request.getUnknownCards().size());
        for (Card card : real.getTalon()) assertTrue(real.toString().contains(card.shortName()));
        Card covered = real.getTalon().getFirst();
        long before = real.getStateKey();
        assertFalse(real.moveCard("W", covered.shortName(), "F1"));
        assertEquals(before, real.getStateKey());
    }

    @Test
    void trainingUndoOfARecycleRestoresPassAllowanceWithoutForgetting() {
        Solitaire real = new Solitaire(new Deck(deck()),
                new GameRules(3, GameRules.StockVisibility.topOnly, 3));
        while (!real.getStockpile().isEmpty()) real.turnStock();
        List<Card> before = List.copyOf(real.getTalon());
        List<Card> unseen = real.getUnknownCards();
        real.turnStock();
        assertEquals(2, real.getStockPass());
        assertTrue(real.undoTrainingMove());
        assertEquals(1, real.getStockPass());
        assertEquals(before, real.getTalon());
        assertEquals(unseen, real.getUnknownCards());
        assertTrue(real.canTurnStock());
    }

    @Test
    void reusedFormatterReadsFreshObservations() {
        Solitaire real = new Solitaire(new Deck(deck()), GameRules.DEFAULT);
        BoardFormatter formatter = new BoardFormatter(real);
        String before = formatter.format();
        real.turnStock();
        assertNotEquals(before, formatter.format());
        assertEquals(real.toString(), formatter.format());
    }
}

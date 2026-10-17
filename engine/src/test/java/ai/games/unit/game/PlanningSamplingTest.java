package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PlanningSamplingTest {
    @Test
    void samplingPreservesKnownCardsAndAssignmentsAndUsesTheExistingUndoHistory() {
        Solitaire real = SolitaireBuilder.newGame().tableau("T1", 1, "5♣", "A♠")
                .tableau("T2", 1, "K♦", "2♥").build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null));
        real.turnThree();
        long realKey = real.getStateKey();
        List<Card> knownWaste = List.copyOf(real.getTalon());
        Solitaire plan = real.planningCopy();
        assertTrue(plan.moveCard("T1", "F1"));
        Card assumed = new Card(Rank.FIVE, Suit.CLUBS);
        plan.assumeCard(plan.nextUnresolvedCard(), assumed);
        plan.turnThree(); // Unresolved waste as well as hidden tableau and stock slots.
        List<Card> before = cards(plan);
        List<Card> pool = plan.getUnknownCards();
        List<Integer> visibility = plan.getTableauFaceUpCounts();
        var guesses = Map.copyOf(plan.getUnknownCardGuesses());
        int checkpoint = plan.getMoveHistorySize();
        long key = plan.getStateKey();

        plan.sampleUnknownCards(new Random(42));

        List<Card> sampled = cards(plan);
        assertEquals(52, new HashSet<>(sampled).size());
        assertTrue(sampled.stream().noneMatch(card -> card.getRank() == Rank.UNKNOWN));
        assertTrue(plan.getUnknownCards().isEmpty());
        assertEquals(checkpoint + pool.size(), plan.getMoveHistorySize());
        assertEquals(visibility, plan.getTableauFaceUpCounts());
        assertEquals(assumed, plan.getTableau().get(0).get(0));
        assertEquals(knownWaste, plan.getTalon().subList(0, knownWaste.size()));
        for (int i = 0; i < before.size(); i++) {
            if (before.get(i).getRank() != Rank.UNKNOWN) assertSame(before.get(i), sampled.get(i));
        }

        int sampledCheckpoint = plan.getMoveHistorySize();
        plan.turnThree();
        plan.rollbackTo(sampledCheckpoint);
        assertEquals(sampled, cards(plan));
        assertTrue(plan.getUnknownCards().isEmpty());

        plan.rollbackTo(checkpoint);
        List<Card> restored = cards(plan);
        for (int i = 0; i < before.size(); i++) assertSame(before.get(i), restored.get(i));
        assertSame(pool, plan.getUnknownCards());
        assertEquals(guesses, plan.getUnknownCardGuesses());
        assertEquals(key, plan.getStateKey());
        assertEquals(realKey, real.getStateKey());
    }

    @Test
    void equalObservationsAndRandomSeedsProduceEqualWorldsDespiteDifferentRealHiddenCards() {
        Solitaire first = SolitaireBuilder.newGame().tableau("T1", 1, "5♣", "A♠").build();
        Solitaire second = SolitaireBuilder.newGame().tableau("T1", 1, "7♦", "A♠").build();
        assertNotEquals(first.getStateKey(), second.getStateKey());
        Solitaire a = first.planningCopy();
        Solitaire b = second.planningCopy();
        a.sampleUnknownCards(new Random(19));
        b.sampleUnknownCards(new Random(19));
        assertEquals(cards(a), cards(b));
        assertEquals(a.getStateKey(), b.getStateKey());
    }

    @Test
    void aSamplingFailureRestoresEveryPartialAssignment() {
        Solitaire plan = SolitaireFactory.stockOnly().planningCopy();
        plan.turnThree();
        int checkpoint = plan.getMoveHistorySize();
        long key = plan.getStateKey();
        List<Card> pool = plan.getUnknownCards();
        List<Card> before = cards(plan);
        RuntimeException failure = new IllegalStateException("Interrupted sampling");
        Random random = new Random() {
            private int draws;
            @Override public int nextInt(int bound) {
                if (draws++ == 3) throw failure;
                return 0;
            }
        };

        assertSame(failure, assertThrows(IllegalStateException.class, () -> plan.sampleUnknownCards(random)));
        assertEquals(before, cards(plan));
        assertSame(pool, plan.getUnknownCards());
        assertTrue(plan.getUnknownCardGuesses().isEmpty());
        assertEquals(checkpoint, plan.getMoveHistorySize());
        assertEquals(key, plan.getStateKey());
    }

    @Test
    void ordinaryAssumptionsStillRequireRevealedSlots() {
        Solitaire plan = SolitaireBuilder.newGame().tableau("T1", 1, "5♣", "A♠")
                .build().planningCopy();
        Card identity = plan.getUnknownCards().get(0);
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(plan.getTableau().get(0).get(0), identity));
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(plan.getStockpile().get(0), identity));
        assertEquals(0, plan.getMoveHistorySize());
    }

    @Test
    void samplingCannotMutateTheRealGame() {
        Solitaire real = SolitaireFactory.stockOnly();
        long key = real.getStateKey();
        assertThrows(IllegalStateException.class, () -> real.sampleUnknownCards(new Random(1)));
        assertEquals(key, real.getStateKey());
        assertEquals(0, real.getMoveHistorySize());
    }

    private static List<Card> cards(Solitaire board) {
        List<Card> cards = new ArrayList<>();
        board.getTableau().forEach(cards::addAll);
        board.getFoundation().forEach(cards::addAll);
        cards.addAll(board.getStockpile());
        cards.addAll(board.getTalon());
        return cards;
    }
}

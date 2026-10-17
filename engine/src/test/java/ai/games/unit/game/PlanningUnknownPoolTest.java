package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardTracker;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PlanningUnknownPoolTest {
    @Test
    void assignmentReplacesAnImmutableSnapshotAndUndoRestoresItsOrder() {
        Solitaire plan = SolitaireBuilder.newGame().build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null)).planningCopy();
        List<Card> before = plan.getUnknownCards();
        assertSame(before, plan.getUnknownCards());
        assertThrows(UnsupportedOperationException.class, () -> before.remove(0));
        plan.turnThree();
        assertSame(before, plan.getUnknownCards());
        Card identity = before.get(17);
        assertEquals(1.0 / 52, plan.assumeCard(plan.nextUnresolvedCard(), identity));
        List<Card> after = plan.getUnknownCards();
        List<Card> expected = new ArrayList<>(before);
        expected.remove(identity);
        assertEquals(expected, after);
        assertEquals(52, before.size());
        assertSame(after, plan.getUnknownCards());
        assertThrows(UnsupportedOperationException.class, () -> after.add(identity));
        assertTrue(plan.undoLastMove());
        assertSame(before, plan.getUnknownCards());
        assertEquals(expected, after); // Earlier snapshots remain usable after rollback too.
    }

    @Test
    void copiesCanAssignAndUndoIndependentlyWhileSharingImmutableSnapshots() {
        Solitaire plan = SolitaireBuilder.newGame().build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null)).planningCopy();
        plan.turnThree();
        List<Card> before = plan.getUnknownCards();
        Solitaire sibling = plan.copy();
        assertSame(before, sibling.getUnknownCards());
        plan.assumeCard(plan.nextUnresolvedCard(), before.get(0));
        List<Card> assigned = plan.getUnknownCards();
        Solitaire descendant = plan.planningCopy();
        assertSame(assigned, descendant.getUnknownCards());
        sibling.assumeCard(sibling.nextUnresolvedCard(), before.get(1));
        assertTrue(sibling.getUnknownCards().contains(before.get(0)));
        assertTrue(plan.getUnknownCards().contains(before.get(1)));
        assertTrue(plan.undoLastMove());
        assertSame(before, plan.getUnknownCards());
        assertSame(assigned, descendant.getUnknownCards());
        assertFalse(descendant.getUnknownCards().contains(before.get(0)));
        assertFalse(descendant.canUndo());
        sibling.rollbackTo(0);
        assertSame(before, sibling.getUnknownCards());
    }

    @Test
    void completeStockAssignmentRecycleAndRollbackAgreeWithRebuildingThePool() {
        Solitaire plan = SolitaireBuilder.newGame().build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null)).planningCopy();
        List<Card> initial = plan.getUnknownCards();
        long key = plan.getStateKey();
        while (!plan.getStockpile().isEmpty()) {
            plan.turnThree();
            Card slot;
            while ((slot = plan.nextUnresolvedCard()) != null) {
                List<Card> pool = plan.getUnknownCards();
                Card chosen = pool.get(pool.size() / 2);
                assertEquals(1.0 / pool.size(), plan.assumeCard(slot, chosen));
                assertEquals(new UnknownCardTracker().getUnknownCards(plan), plan.getUnknownCards());
            }
        }
        List<Card> empty = plan.getUnknownCards();
        assertTrue(empty.isEmpty());
        plan.turnThree(); // Recycle assigned identities; none become available again.
        assertSame(empty, plan.getUnknownCards());
        assertNull(plan.nextUnresolvedCard());
        while (plan.canUndo()) {
            plan.undoLastMove();
            assertEquals(new UnknownCardTracker().getUnknownCards(plan), plan.getUnknownCards());
        }
        assertSame(initial, plan.getUnknownCards());
        assertEquals(key, plan.getStateKey());
        assertEquals(52, initial.size());
    }

    @Test
    void movingAssignedCardsAndUndoingMovesDoNotReleaseTheirIdentities() {
        Solitaire plan = SolitaireBuilder.newGame().tableau("T1", 1, "5♣", "A♠")
                .tableau("T2", "6♥").build().planningCopy();
        List<Card> before = plan.getUnknownCards();
        assertTrue(plan.moveCard("T1", "F1"));
        assertSame(before, plan.getUnknownCards());
        plan.assumeCard(plan.nextUnresolvedCard(), new Card(Rank.FIVE, Suit.CLUBS));
        List<Card> assigned = plan.getUnknownCards();
        assertTrue(plan.moveCard("T1", "T2"));
        assertSame(assigned, plan.getUnknownCards());
        assertTrue(plan.undoLastMove());
        assertSame(assigned, plan.getUnknownCards());
        assertTrue(plan.undoLastMove());
        assertSame(before, plan.getUnknownCards());
    }

    @Test
    void invalidAssignmentsDoNotChangeThePoolOrHistory() {
        Solitaire plan = SolitaireBuilder.newGame().build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null)).planningCopy();
        plan.turnThree();
        List<Card> before = plan.getUnknownCards();
        int depth = plan.getMoveHistorySize();
        assertThrows(IllegalArgumentException.class, () -> plan.assumeCard(
                new Card(Rank.UNKNOWN, Suit.UNKNOWN), before.get(0)));
        assertSame(before, plan.getUnknownCards());
        assertEquals(depth, plan.getMoveHistorySize());
        plan.assumeCard(plan.nextUnresolvedCard(), before.get(0));
        List<Card> assigned = plan.getUnknownCards();
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(plan.nextUnresolvedCard(), before.get(0)));
        assertSame(assigned, plan.getUnknownCards());
        assertEquals(depth + 1, plan.getMoveHistorySize());
    }
}

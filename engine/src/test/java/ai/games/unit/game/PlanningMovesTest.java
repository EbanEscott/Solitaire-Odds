package ai.games.unit.game;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PlanningMovesTest {
    @Test
    void planningCopyKeepsTheRealGameAndItsCurrentlyLegalMovesUnchanged() {
        Solitaire real = threeUnseenCards();
        Solitaire plan = real.planningCopy();
        assertFalse(real.isInPlanMode());
        assertTrue(plan.isInPlanMode());
        assertEquals(LegalMovesHelper.listLegalMoves(real), LegalMovesHelper.listLegalMoves(plan));
        assertTrue(plan.copy().isInPlanMode());
    }

    @Test
    void threeUnseenCardsHaveConditionalProbabilitiesOneThirdOneHalfAndOne() {
        Solitaire real = threeUnseenCards();
        Solitaire plan = real.planningCopy();
        long initialKey = plan.getStateKey();
        plan.turnThree();
        long turnedKey = plan.getStateKey();
        assertNotEquals(initialKey, turnedKey);
        List<Card> unknownSlots = List.copyOf(plan.getTalon());
        List<Card> unseen = plan.getUnknownCards();
        assertEquals(3, unseen.size());
        int depth = plan.getMoveHistorySize();
        double probability = plan.assumeCard(plan.nextUnresolvedCard(), unseen.get(0));
        assertEquals(1.0 / 3, probability);
        assertEquals(2, plan.getUnknownCards().size());
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(plan.nextUnresolvedCard(), unseen.get(0)));
        probability *= plan.assumeCard(plan.nextUnresolvedCard(), unseen.get(1));
        assertEquals(1.0 / 6, probability);
        probability *= plan.assumeCard(plan.nextUnresolvedCard(), unseen.get(2));
        assertEquals(1.0 / 6, probability);
        assertNull(plan.nextUnresolvedCard());
        assertTrue(plan.getUnknownCards().isEmpty());
        assertEquals(3, plan.getUnknownCardGuesses().size());

        // Recycling does not draw new identities for slots that are already assigned.
        plan.turnThree();
        assertNull(plan.nextUnresolvedCard());
        assertTrue(plan.getUnknownCards().isEmpty());
        plan.rollbackTo(depth);
        assertEquals(turnedKey, plan.getStateKey());
        assertEquals(unseen, plan.getUnknownCards());
        assertTrue(plan.getUnknownCardGuesses().isEmpty());
        for (int i = 0; i < 3; i++) assertSame(unknownSlots.get(i), plan.getTalon().get(i));
        plan.rollbackTo(0);
        assertEquals(initialKey, plan.getStateKey());
        assertTrue(real.getTalon().isEmpty());
    }

    @Test
    void siblingsKeepIndependentAssignmentsAndCopiesRetainCommittedIdentities() {
        Solitaire parent = threeUnseenCards().planningCopy();
        parent.turnThree();
        Card unknown = parent.nextUnresolvedCard();
        List<Card> unseen = parent.getUnknownCards();
        Solitaire a = parent.copy();
        Solitaire b = parent.copy();
        a.assumeCard(unknown, unseen.get(0));
        b.assumeCard(unknown, unseen.get(1));
        assertNotEquals(a.getStateKey(), b.getStateKey());
        assertEquals(3, parent.getUnknownCards().size());
        assertTrue(parent.getUnknownCardGuesses().isEmpty());
        Solitaire copy = a.copy();
        assertEquals(a.getTalon(), copy.getTalon());
        assertEquals(a.getUnknownCards(), copy.getUnknownCards());
        assertFalse(copy.canUndo());
        a.undoLastMove();
        assertEquals(3, a.getUnknownCards().size());
        assertEquals(2, copy.getUnknownCards().size());
    }

    @Test
    void unresolvedCardsCannotActAsWildcardsAndMoveListingDoesNotCommitGuesses() {
        Solitaire plan = SolitaireBuilder.newGame()
                .tableau("T1", 1, "5♣", "A♠").tableau("T2", "4♥").build().planningCopy();
        assertThrows(IllegalArgumentException.class,
                () -> plan.assumeCard(plan.getTableau().get(0).get(0), new Card(Rank.FIVE, Suit.CLUBS)));
        assertTrue(plan.moveCard("T1", "F1"));
        assertFalse(plan.moveCard("T1", "F2"));
        assertFalse(plan.moveCard("T2", "T1"));
        assertFalse(LegalMovesHelper.listLegalMoves(plan).contains("move T2 4♥ T1"));
        assertTrue(plan.getUnknownCardGuesses().isEmpty());
        plan.assumeCard(plan.nextUnresolvedCard(), new Card(Rank.FIVE, Suit.CLUBS));
        assertTrue(LegalMovesHelper.listLegalMoves(plan).contains("move T2 4♥ T1"));
        assertTrue(plan.moveCard("T2", "T1"));
        plan.rollbackTo(0);
        assertEquals(Rank.UNKNOWN, plan.getTableau().get(0).get(0).getRank());
        assertTrue(plan.getUnknownCardGuesses().isEmpty());
    }

    private static Solitaire threeUnseenCards() {
        SolitaireBuilder builder = SolitaireBuilder.newGame();
        int pile = 1;
        for (Suit suit : List.of(Suit.CLUBS, Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS)) {
            int maximum = suit == Suit.DIAMONDS ? 10 : 13;
            String[] cards = Arrays.stream(Rank.values()).filter(rank -> rank != Rank.UNKNOWN && rank.getValue() <= maximum)
                    .map(rank -> new Card(rank, suit).shortName()).toArray(String[]::new);
            builder.foundation("F" + pile++, cards);
        }
        return builder.build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null));
    }
}

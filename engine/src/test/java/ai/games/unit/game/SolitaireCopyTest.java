package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardGuess;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.List;
import org.junit.jupiter.api.Test;

class SolitaireCopyTest {
    @Test
    void gameCopyPreservesTheEntireBoardAndLegalMoves() {
        Solitaire original = position();

        Solitaire copy = original.copy();

        assertEquals(snapshot(original), snapshot(copy));
        assertEquals(original.getStateKey(), copy.getStateKey());
        assertEquals(LegalMovesHelper.listLegalMoves(original), LegalMovesHelper.listLegalMoves(copy));
        assertSame(original.getTableau().get(0).get(0), copy.getTableau().get(0).get(0));
        assertSame(original.getFoundation().get(0).get(0), copy.getFoundation().get(0).get(0));
        assertSame(original.getStockpile().get(0), copy.getStockpile().get(0));
        assertSame(original.getTalon().get(0), copy.getTalon().get(0));
        SolitaireBuilder.assertValidGameState(copy);
    }

    @Test
    void movesAndStockTurnsDoNotLeakBetweenCopies() {
        Solitaire original = position();
        Board before = snapshot(original);
        Solitaire copy = original.copy();
        Solitaire sibling = original.copy();

        // Exercise tableau transfers, revealing a hidden card, foundation growth, and stock/talon changes.
        assertTrue(copy.moveCard("T1", "T2"));
        assertEquals(0, copy.getTableauFaceDownCounts().get(0));
        assertTrue(copy.moveCard("W", "F1"));
        copy.turnThree();

        assertEquals(before, snapshot(original));
        assertEquals(before, snapshot(sibling));
        SolitaireBuilder.assertValidGameState(copy);

        Board copiedPosition = snapshot(copy);
        original.turnThree();
        assertEquals(copiedPosition, snapshot(copy));
        assertEquals(before, snapshot(sibling));
    }

    @Test
    void planCopiesMaskHiddenCardsAndKeepGuessesIndependent() {
        Solitaire original = position();
        Card hidden = original.getTableau().get(0).get(0);

        Solitaire plan = original.planningCopy();
        Card unknown = plan.getTableau().get(0).get(0);
        assertEquals(Solitaire.GameMode.PLAN, plan.getMode());
        assertEquals(Rank.UNKNOWN, unknown.getRank());
        assertEquals(Suit.UNKNOWN, unknown.getSuit());
        assertEquals(Rank.KING, hidden.getRank());
        assertSame(original.getVisibleTableau().get(0).get(0), plan.getVisibleTableau().get(0).get(0));
        assertEquals(original.getTableauFaceUpCounts(), plan.getTableauFaceUpCounts());
        assertEquals(original.getTableauFaceDownCounts(), plan.getTableauFaceDownCounts());

        UnknownCardGuess guess = new UnknownCardGuess(unknown, hidden);
        plan.getUnknownCardGuesses().put(unknown, guess);
        Solitaire copy = plan.copy();
        Card copiedUnknown = copy.getTableau().get(0).get(0);
        UnknownCardGuess copiedGuess = copy.getUnknownCardGuesses().get(copiedUnknown);

        assertEquals(snapshot(plan), snapshot(copy));
        assertSame(unknown, copiedUnknown);
        assertNotNull(copiedGuess);
        assertSame(guess, copiedGuess);
        assertSame(copiedUnknown, copiedGuess.getUnknownCard());
        assertEquals(guess.getPossibilities(), copiedGuess.getPossibilities());
        copy.getUnknownCardGuesses().clear();
        assertEquals(1, plan.getUnknownCardGuesses().size());
    }

    @Test
    void planningMasksUnseenStockAndLeavesTheRealBoardUntouched() {
        Solitaire real = position();
        Board before = snapshot(real);
        long realKey = real.getStateKey();

        Solitaire plan = real.planningCopy();

        assertEquals(before, snapshot(real));
        assertEquals(realKey, real.getStateKey());
        assertFalse(real.isInPlanMode());
        assertTrue(plan.getStockpile().stream().allMatch(card -> card.getRank() == Rank.UNKNOWN));
        assertEquals(real.getStockpile().size(), plan.getStockpile().size());
        assertEquals(real.getTalon(), plan.getTalon());
        assertEquals(real.getFoundation(), plan.getFoundation());
        plan.turnThree();
        assertEquals(Rank.UNKNOWN, plan.getTalon().get(plan.getTalon().size() - 1).getRank());
        assertTrue(plan.undoLastMove());
        assertEquals(snapshot(real.planningCopy()), snapshot(plan));
        assertEquals(before, snapshot(real));
    }

    @Test
    void bothKindsOfCopyPreservePreviouslySeenStockAndIndependentUndo() {
        Solitaire real = new Solitaire(new Deck());
        real.turnThree();
        List<Card> seen = List.copyOf(real.getTalon());
        Solitaire realCopy = real.copy();
        assertTrue(real.undoLastMove());
        assertEquals(45, real.getUnknownCards().size());
        assertEquals(42, realCopy.getUnknownCards().size());
        assertEquals(seen, realCopy.planningCopy().getTalon());

        // A complete stock pass followed by recycling retains the observed stock order.
        for (int i = 0; i < 9; i++) real.turnThree();
        Solitaire plan = real.planningCopy();
        assertEquals(real.getStockpile(), plan.getStockpile());
        assertEquals(21, plan.getUnknownCards().size());
        for (Solitaire copy : List.of(real.copy(), plan.copy(), plan.planningCopy())) {
            assertEquals(real.getStockpile(), copy.getStockpile());
            assertEquals(21, copy.getUnknownCards().size());
            assertFalse(copy.canUndo());
            Board before = snapshot(copy);
            copy.turnThree();
            copy.rollbackTo(0);
            assertEquals(before, snapshot(copy));
            assertEquals(21, copy.getUnknownCards().size());
        }
    }

    @Test
    void hiddenDealsDoNotChangePlanningPositionsOrPossibleUnseenCards() {
        Solitaire a = SolitaireBuilder.newGame().tableau("T1", 1, "K♦", "5♠").build();
        Solitaire b = SolitaireBuilder.newGame().tableau("T1", 1, "Q♦", "5♠").build();
        Solitaire planA = a.planningCopy();
        Solitaire planB = b.planningCopy();
        assertNotEquals(a.getTableau(), b.getTableau());
        assertEquals(snapshot(planA), snapshot(planB));
        assertEquals(planA.getStateKey(), planB.getStateKey());
        assertEquals(planA.getUnknownCards(), planB.getUnknownCards());
        assertEquals(51, planA.getUnknownCards().size());
        assertFalse(planA.getUnknownCards().contains(new Card(Rank.FIVE, Suit.SPADES)));
    }

    @Test
    void copiesRetainSeparateGuessesForHiddenAndExposedUnknownPositions() {
        Solitaire plan = SolitaireBuilder.newGame()
                .tableau("T1", 1, "K♦", "A♣").tableau("T2", 1, "Q♣", "4♥")
                .build().planningCopy();
        Card first = plan.getTableau().get(0).get(0);
        Card second = plan.getTableau().get(1).get(0);
        UnknownCardGuess a = new UnknownCardGuess(first, new Card(Rank.KING, Suit.DIAMONDS));
        UnknownCardGuess b = new UnknownCardGuess(second, new Card(Rank.QUEEN, Suit.CLUBS));
        plan.getUnknownCardGuesses().put(first, a);
        plan.getUnknownCardGuesses().put(second, b);
        assertEquals(first, second); // Value equality must not merge different hidden positions.
        assertNotSame(first, second);
        assertTrue(plan.moveCard("T1", "F1"));
        long key = plan.getStateKey();
        for (Solitaire copy : List.of(plan.copy(), plan.planningCopy())) {
            assertEquals(2, copy.getUnknownCardGuesses().size());
            assertSame(a, copy.getUnknownCardGuesses().get(first));
            assertSame(b, copy.getUnknownCardGuesses().get(second));
            assertEquals(key, copy.getStateKey());
            assertEquals(snapshot(plan), snapshot(copy));
            copy.getUnknownCardGuesses().remove(first);
            assertEquals(2, plan.getUnknownCardGuesses().size());
        }
    }

    private static Solitaire position() {
        return SolitaireBuilder.newGame()
                .tableau("T1", 1, "K♦", "5♠")
                .tableau("T2", "6♥")
                .foundation("F1", "A♣")
                .waste("2♣")
                .build();
    }

    private static Board snapshot(Solitaire game) {
        return new Board(
                game.getMode(),
                game.getTableau().stream().map(List::copyOf).toList(),
                game.getFoundation().stream().map(List::copyOf).toList(),
                List.copyOf(game.getTableauFaceUpCounts()),
                List.copyOf(game.getStockpile()),
                List.copyOf(game.getTalon()));
    }

    private record Board(Solitaire.GameMode mode, List<List<Card>> tableau,
                         List<List<Card>> foundation, List<Integer> faceUp,
                         List<Card> stock, List<Card> talon) {}
}

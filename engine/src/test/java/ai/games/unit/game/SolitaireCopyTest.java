package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
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
        original.setMode(Solitaire.GameMode.PLAN);
        Card hidden = original.getTableau().get(0).get(0);

        Solitaire plan = original.copy();
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
        assertNotSame(unknown, copiedUnknown);
        assertNotNull(copiedGuess);
        assertNotSame(guess, copiedGuess);
        assertSame(copiedUnknown, copiedGuess.getUnknownCard());
        assertEquals(guess.getPossibilities(), copiedGuess.getPossibilities());
        copy.getUnknownCardGuesses().clear();
        assertEquals(1, plan.getUnknownCardGuesses().size());
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

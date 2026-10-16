package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.Game;
import ai.games.config.GuidanceModeProperties;
import ai.games.config.TrainingModeProperties;
import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardGuess;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

/** Exact undo shared by training commands and simulation boards. */
class TrainingModeTest {
    @Test
    void failedMovesAndEmptyTurnsDoNotCreateHistory() {
        Solitaire state = new Solitaire(new Deck(List.of()));
        Board before = snapshot(state);
        assertFalse(state.moveCard("W", "T1"));
        state.turnThree();
        assertFalse(state.canUndo());
        assertFalse(state.undoLastMove());
        state.rollbackTo(0);
        assertEquals(before, snapshot(state));
    }

    @ParameterizedTest
    @CsvSource({"T1,T2", "T3,F3", "W,F1", "W,T4", "F2,T5"})
    void undoRestoresEveryCardMoveCategory(String from, String to) {
        Solitaire state = position();
        Board before = snapshot(state);
        assertTrue(state.moveCard(from, to));
        assertTrue(state.canUndo());
        assertEquals(1, state.getMoveHistorySize());
        if (from.equals("T1")) {
            assertEquals(0, state.getTableauFaceDownCounts().get(0));
        }
        assertTrue(state.undoLastMove());
        assertEquals(before, snapshot(state));
        assertFalse(state.undoLastMove());
    }

    @Test
    void undoRestoresAStackInOrderAndTurnsTheExposedCardFaceDown() {
        Solitaire state = SolitaireBuilder.newGame()
                .tableau("T1", 2, "3♣", "Q♥", "J♠").tableau("T2", "K♣").build();
        Board before = snapshot(state);
        assertTrue(state.moveCard("T1", "Q♥", "T2"));
        assertEquals(List.of("K♣", "Q♥", "J♠"),
                state.getVisibleTableau().get(1).stream().map(Card::shortName).toList());
        assertEquals(0, state.getTableauFaceDownCounts().get(0));
        assertTrue(state.undoLastMove());
        assertEquals(before, snapshot(state));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 5, 24})
    void turnsAndRecyclesRestoreExactOrderAndKnowledge(int stockSize) {
        Solitaire state = gameWithStockSize(stockSize);
        List<Board> positions = new ArrayList<>();
        // Two complete stock passes exercise partial draws and already-known cards.
        int turns = 2 * ((stockSize + 2) / 3);
        for (int i = 0; i < turns; i++) {
            positions.add(snapshot(state));
            state.turnThree();
            assertEquals(i + 1, state.getMoveHistorySize());
        }
        for (int i = positions.size() - 1; i >= 0; i--) {
            assertTrue(state.undoLastMove());
            assertEquals(positions.get(i), snapshot(state), "Undo turn " + (i + 1));
        }
        assertFalse(state.canUndo());
    }

    @Test
    void recycleUndoRestoresPreviouslyUnseenWasteCardsOnACopy() {
        Solitaire state = gameWithStockSize(5);
        state.turnThree();
        state.turnThree();
        Solitaire copy = state.copy(); // Fresh revealed tracking; stock is empty.
        Board before = snapshot(copy);
        copy.turnThree(); // Recycle and reveal three cards.
        assertTrue(copy.undoLastMove());
        assertEquals(before, snapshot(copy));
        // Drawing the same cards again must give the same knowledge as the first time.
        copy.turnThree();
        Board after = snapshot(copy);
        copy.undoLastMove();
        copy.turnThree();
        assertEquals(after, snapshot(copy));
    }

    @Test
    void mixedActionsCanRollbackAndExploreAnotherBranch() {
        Solitaire state = position();
        Board start = snapshot(state);
        assertTrue(state.moveCard("T1", "T2"));
        int checkpoint = state.getMoveHistorySize();
        Board candidate = snapshot(state);
        for (int i = 0; i < 3; i++) {
            assertTrue(state.moveCard("W", "F1"));
            state.turnThree();
            assertTrue(state.moveCard("F2", "T5"));
            state.rollbackTo(checkpoint);
            assertEquals(candidate, snapshot(state));
        }
        assertTrue(state.moveCard("T3", "F3"));
        state.rollbackTo(checkpoint);
        assertEquals(candidate, snapshot(state));
        state.rollbackTo(0);
        assertEquals(start, snapshot(state));
    }

    @Test
    void rollbackRejectsInvalidDepthBeforeChangingTheBoard() {
        Solitaire state = position();
        state.turnThree();
        Board before = snapshot(state);
        assertThrows(IllegalArgumentException.class, () -> state.rollbackTo(-1));
        assertThrows(IllegalArgumentException.class, () -> state.rollbackTo(2));
        state.rollbackTo(1);
        assertEquals(before, snapshot(state));
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void copiesHaveIndependentUndoBoundaries(Solitaire.GameMode mode) {
        Solitaire source = position();
        if (mode == Solitaire.GameMode.PLAN) source = source.planningCopy();
        source.turnThree();
        Board original = snapshot(source);
        Solitaire copy = source.copy();
        Board copied = snapshot(copy);
        assertFalse(copy.canUndo());
        assertFalse(copy.undoLastMove());
        assertTrue(copy.moveCard("T1", "T2"));
        copy.turnThree();
        copy.rollbackTo(0);
        assertEquals(copied, snapshot(copy));
        assertEquals(original, snapshot(source));
        source.undoLastMove();
        assertEquals(copied, snapshot(copy));
    }

    @Test
    void planUndoPreservesUnknownCardIdentityAndGuesses() {
        Solitaire source = position();
        source = source.planningCopy();
        Solitaire plan = source.copy();
        Card unknown = plan.getTableau().get(0).get(0);
        UnknownCardGuess guess = new UnknownCardGuess(unknown, new Card(Rank.KING, Suit.DIAMONDS));
        plan.getUnknownCardGuesses().put(unknown, guess);
        Board before = snapshot(plan);
        assertTrue(plan.moveCard("T1", "T2")); // Expose the UNKNOWN card.
        plan.assumeCard(unknown, new Card(Rank.KING, Suit.DIAMONDS));
        assertTrue(plan.moveCard("T1", "T6")); // Move the assigned king using normal rules.
        plan.rollbackTo(0);
        assertEquals(before, snapshot(plan));
        assertSame(unknown, plan.getTableau().get(0).get(0));
        assertSame(guess, plan.getUnknownCardGuesses().get(unknown));
    }

    @Test
    void undoRestoresAnAlreadyCalculatedKeyWithoutRescanningCards() {
        Card ace = spy(new Card(Rank.ACE, Suit.CLUBS));
        Solitaire state = new Solitaire(new Deck(List.of(ace)));
        long key = state.getStateKey();
        assertTrue(state.moveCard("T1", "F1"));
        state.getStateKey();
        clearInvocations(ace);
        assertTrue(state.undoLastMove());
        assertEquals(key, state.getStateKey());
        verifyNoInteractions(ace);
    }

    @Test
    void undoRestoresAnUncomputedKeyAsInvalid() {
        Card ace = spy(new Card(Rank.ACE, Suit.CLUBS));
        Solitaire state = new Solitaire(new Deck(List.of(ace)));
        assertTrue(state.moveCard("T1", "F1"));
        state.getStateKey();
        assertTrue(state.undoLastMove());
        clearInvocations(ace);
        long restored = state.getStateKey();
        verify(ace).getRank();
        assertEquals(new Solitaire(new Deck(List.of(new Card(Rank.ACE, Suit.CLUBS)))).getStateKey(), restored);
    }

    @Test
    void seededLegalPathUndoesToEveryPreviousPosition() {
        Solitaire state = gameWithStockSize(24).copy();
        Random random = new Random(719);
        List<Board> positions = new ArrayList<>();
        for (int i = 0; i < 150; i++) {
            List<Move> moves = LegalMovesHelper.listMoves(state);
            moves.removeIf(Move::isQuit);
            if (moves.isEmpty()) break;
            positions.add(snapshot(state));
            Move move = moves.get(random.nextInt(moves.size()));
            if (move.isTurn()) {
                state.turnThree();
            } else {
                assertTrue(state.moveCard(move.from().toCode(),
                        move.card() == null ? null : move.card().shortName(), move.to().toCode()));
            }
            assertEquals(i + 1, state.getMoveHistorySize());
        }
        assertTrue(positions.size() >= 20, "Exercise a substantial mixed path");
        for (int i = positions.size() - 1; i >= 0; i--) {
            assertTrue(state.undoLastMove());
            assertEquals(positions.get(i), snapshot(state), "Undo path step " + i);
        }
    }

    @Test
    void trainingCommandUsesTheSameHistoryAndRecordsTurnsOnlyOnce() {
        Solitaire state = position();
        Board before = snapshot(state);
        Player player = mock(Player.class);
        when(player.nextCommand(any(), anyString(), anyString()))
                .thenReturn("turn", "undo", "quit");
        TrainingModeProperties training = new TrainingModeProperties();
        training.setMode(true);
        GuidanceModeProperties guidance = new GuidanceModeProperties();
        guidance.setMode(false);
        new Game(player, training, guidance).play(state);
        assertEquals(before, snapshot(state));
        verify(player, times(3)).nextCommand(any(), anyString(), anyString());
    }

    private static Solitaire position() {
        return SolitaireBuilder.newGame().tableau("T1", 1, "K♦", "5♠")
                .tableau("T2", "6♥").tableau("T3", "A♠")
                .tableau("T4", "3♥").tableau("T5", "4♣")
                .foundation("F1", "A♣").foundation("F2", "A♦", "2♦", "3♦")
                .waste("2♣").build();
    }

    private static Solitaire gameWithStockSize(int stockSize) {
        List<Card> cards = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) continue;
            for (Rank rank : Rank.values()) {
                if (rank != Rank.UNKNOWN) cards.add(new Card(rank, suit));
            }
        }
        return new Solitaire(new Deck(cards.subList(0, 28 + stockSize)));
    }

    private static Board snapshot(Solitaire state) {
        return new Board(state.getTableau().stream().map(List::copyOf).toList(),
                state.getFoundation().stream().map(List::copyOf).toList(),
                List.copyOf(state.getTableauFaceUpCounts()), List.copyOf(state.getStockpile()),
                List.copyOf(state.getTalon()), new HashSet<>(state.getUnknownCards()),
                Map.copyOf(state.getUnknownCardGuesses()), state.getMode(), state.getStateKey(),
                state.getMoveHistorySize());
    }

    private record Board(List<List<Card>> tableau, List<List<Card>> foundations,
                         List<Integer> faceUp, List<Card> stock, List<Card> talon,
                         Set<Card> unknown, Map<Card, UnknownCardGuess> guesses,
                         Solitaire.GameMode mode, long key, int historySize) {}
}

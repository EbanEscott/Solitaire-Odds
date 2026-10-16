package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class SolitaireStateKeyCacheTest {
    @Test
    void repeatedReadsAndGameCopiesReuseTheCalculatedKey() {
        Card ace = spy(new Card(Rank.ACE, Suit.CLUBS));
        Solitaire state = new Solitaire(new Deck(List.of(ace)));
        clearInvocations(ace);

        long key = state.getStateKey();
        verify(ace).getRank();
        clearInvocations(ace);

        assertEquals(key, state.getStateKey());
        Solitaire copy = state.copy();
        assertEquals(key, copy.getStateKey());
        assertEquals(key, copy.getStateKey());
        verifyNoInteractions(ace);
    }

    @Test
    void failedMovesAndEmptyStockTurnsKeepTheCachedKey() {
        Card ace = spy(new Card(Rank.ACE, Suit.CLUBS));
        Solitaire state = new Solitaire(new Deck(List.of(ace)));
        long key = state.getStateKey();

        assertFalse(state.moveCard("T1", "T2")); // An ace cannot fill an empty tableau.
        state.turnThree(); // Both stock and waste are empty.
        assertFalse(state.undoLastMove());
        clearInvocations(ace);

        assertEquals(key, state.getStateKey());
        verifyNoInteractions(ace);
    }

    @ParameterizedTest
    @CsvSource({"T1,T2", "T3,F3", "W,F1", "W,T4", "F2,T5"})
    void everyCardMoveCategoryInvalidatesTheKey(String from, String to) throws Exception {
        Solitaire state = position();
        long before = state.getStateKey();

        assertTrue(state.moveCard(from, to));

        long expected = fullHash(state);
        assertNotEquals(before, expected, "Fixture must change the existing hash");
        assertEquals(expected, state.getStateKey());
        assertEquals(expected, state.getStateKey());
        if (from.equals("T1")) {
            assertEquals(0, state.getTableauFaceDownCounts().get(0));
        }
    }

    @Test
    void stockDrawsAndRecyclingInvalidateTheKey() throws Exception {
        Solitaire state = orderedGame();
        for (int turn = 0; turn < 9; turn++) {
            state.getStateKey();
            if (turn == 8) {
                assertTrue(state.getStockpile().isEmpty());
                assertFalse(state.getTalon().isEmpty());
            }
            state.turnThree();
            assertEquals(fullHash(state), state.getStateKey(), "Turn " + turn);
        }
    }

    @Test
    void undoRestoresKeysAfterMultipleTurns() {
        Solitaire state = orderedGame();
        long initial = state.getStateKey();
        state.turnThree();
        long afterFirstTurn = state.getStateKey();
        state.turnThree();
        state.getStateKey();

        assertTrue(state.undoLastMove());
        assertEquals(afterFirstTurn, state.getStateKey());
        assertTrue(state.undoLastMove());
        assertEquals(initial, state.getStateKey());
    }

    @Test
    void undoCardMoveRestoresTheInitialKey() {
        Solitaire state = new Solitaire(new Deck(List.of(
                new Card(Rank.KING, Suit.SPADES), new Card(Rank.FIVE, Suit.HEARTS),
                new Card(Rank.ACE, Suit.CLUBS))));
        long initial = state.getStateKey();
        assertTrue(state.moveCard("T1", "F1"));
        assertNotEquals(initial, state.getStateKey());

        assertTrue(state.undoLastMove());
        assertEquals(initial, state.getStateKey());
    }

    @Test
    void sourceAndCopyInvalidateIndependently() throws Exception {
        Solitaire state = position();
        long originalKey = state.getStateKey();
        Solitaire copy = state.copy();
        assertTrue(copy.moveCard("T1", "T2"));
        long copyKey = copy.getStateKey();
        assertEquals(fullHash(copy), copyKey);
        assertEquals(originalKey, state.getStateKey());

        state.turnThree();
        assertEquals(fullHash(state), state.getStateKey());
        assertEquals(copyKey, copy.getStateKey());
    }

    @Test
    void copyingAnUnhashedBoardDoesNotTreatTheDefaultValueAsCached() throws Exception {
        Solitaire state = position();
        Solitaire copy = state.copy();
        long expected = fullHash(state);
        assertNotEquals(0L, expected);
        assertEquals(expected, copy.getStateKey());
        assertEquals(expected, state.getStateKey());
    }

    @Test
    void planCopyRehashesMaskedCardsRatherThanInheritingTheSourceKey() throws Exception {
        Solitaire state = position();
        long originalKey = state.getStateKey();
        state.setMode(Solitaire.GameMode.PLAN);
        // Mode alone does not change the hashed cards; masking happens in copy().
        assertEquals(originalKey, state.getStateKey());

        Solitaire copy = state.copy();
        assertEquals(Rank.UNKNOWN, copy.getTableau().get(0).get(0).getRank());
        long expected = fullHash(copy);
        assertNotEquals(originalKey, expected);
        assertEquals(expected, copy.getStateKey());
        assertEquals(originalKey, state.getStateKey());
    }

    @Test
    void cycleCheckSeesDirectBoardMutationsAfterTheNodeWasCreated() {
        Solitaire state = position();
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        node.setMove(Move.turn());

        Solitaire nextState = state.copy();
        nextState.turnThree();
        MonteCarloTreeNode earlier = new MonteCarloTreeNode(nextState);
        earlier.setParent(new MonteCarloTreeNode(nextState.copy()));
        node.setParent(earlier);
        assertFalse(node.isCycleDetected());

        // Mutate via Solitaire rather than TreeNode.applyMove(). Its own cache must refresh.
        state.turnThree();
        assertTrue(node.isCycleDetected());
    }

    @Test
    void zeroIsAValidKey() {
        Solitaire empty = new Solitaire(new Deck(List.of()));
        assertEquals(0L, empty.getStateKey());
        assertEquals(0L, empty.getStateKey());
        assertEquals(0L, empty.copy().getStateKey());
    }

    // Bypass the cache to compare against a complete hash of the current board.
    private long fullHash(Solitaire state) throws Exception {
        Method method = Solitaire.class.getDeclaredMethod("computeStateKey");
        method.setAccessible(true);
        return (long) method.invoke(state);
    }

    private Solitaire position() {
        return SolitaireBuilder.newGame().tableau("T1", 1, "K♦", "5♠")
                .tableau("T2", "6♥").tableau("T3", "A♠")
                .tableau("T4", "3♥").tableau("T5", "4♣")
                .foundation("F1", "A♣").foundation("F2", "A♦", "2♦", "3♦")
                .waste("2♣").build();
    }

    private Solitaire orderedGame() {
        List<Card> cards = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN) continue;
            for (Rank rank : Rank.values()) {
                if (rank != Rank.UNKNOWN) cards.add(new Card(rank, suit));
            }
        }
        return new Solitaire(new Deck(cards));
    }
}

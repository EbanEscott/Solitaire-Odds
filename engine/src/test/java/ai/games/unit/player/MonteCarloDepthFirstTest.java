package ai.games.unit.player;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class MonteCarloDepthFirstTest {
    @Test
    void backtracksFromADeadEndAndFindsAWinningSiblingOnTheLastVisit() {
        Solitaire board = spy(SolitaireFactory.oneMoveFromWin().planningCopy());
        long root = board.getStateKey();
        Move deadEnd = Move.tryParse("move F1 T2");
        Move win = Move.tryParse("move T1 K♥ F4");
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation ->
                    board.getStateKey() == root ? List.of(deadEnd, win) : List.of(Move.quit()));

            assertEquals(0.5, searchAndAssertRestored(board, 2));
            assertEquals(1.0, searchAndAssertRestored(board, 3));
        }
    }

    @Test
    void distinguishesAnAlreadyWonBoardExhaustedSearchAndAnUnresolvedCutoff() {
        assertEquals(1.0, searchAndAssertRestored(spy(SolitaireFactory.wonGame().planningCopy()), 1));
        Solitaire board = spy(SolitaireFactory.oneMoveFromWin().planningCopy());
        assertEquals(0.5, searchAndAssertRestored(board, 1));
        assertEquals(1.0, searchAndAssertRestored(board, 2));
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenReturn(List.of(Move.quit()));
            assertEquals(0.0, searchAndAssertRestored(board, 1));
        }
    }

    @Test
    void stockCyclesTerminateWithoutConsumingTheEntireBudget() {
        Solitaire board = spy(SolitaireFactory.stockOnly().planningCopy());
        AtomicInteger positions = new AtomicInteger();
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation -> {
                positions.incrementAndGet();
                return List.of(Move.turn());
            });
            assertEquals(0.0, searchAndAssertRestored(board, 100));
        }
        assertTrue(positions.get() > 1 && positions.get() < 100);
    }

    @Test
    void theSamePositionCanBeExploredFromDifferentBranches() {
        Solitaire board = spy(SolitaireBuilder.newGame().tableau("T1", "A♣")
                .tableau("T2", "A♥").build().planningCopy());
        Move clubs = Move.tryParse("move T1 A♣ F1");
        Move hearts = Move.tryParse("move T2 A♥ F2");
        AtomicInteger joinedVisits = new AtomicInteger();
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation -> {
                boolean hasClubs = !board.getFoundation().get(0).isEmpty();
                boolean hasHearts = !board.getFoundation().get(1).isEmpty();
                if (hasClubs && hasHearts) {
                    joinedVisits.incrementAndGet();
                    return List.of(Move.quit());
                }
                if (hasClubs) return List.of(hearts);
                if (hasHearts) return List.of(clubs);
                return List.of(clubs, hearts);
            });
            assertEquals(0.0, searchAndAssertRestored(board, 5));
        }
        assertEquals(2, joinedVisits.get());
    }

    @Test
    void hiddenArrangementStaysFixedAcrossBacktrackingAndIsResampledNextTime() {
        Solitaire board = spy(SolitaireBuilder.newGame().tableau("T1", "A♣")
                .tableau("T2", "A♥").build().planningCopy());
        List<Card> pool = board.getUnknownCards();
        AtomicInteger draws = new AtomicInteger();
        Random random = new Random(10) {
            @Override public int nextInt(int bound) {
                draws.incrementAndGet();
                return super.nextInt(bound);
            }
        };
        List<List<Card>> arrangements = new ArrayList<>();
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation -> {
                assertTrue(board.getUnknownCards().isEmpty());
                arrangements.add(List.copyOf(board.getStockpile()));
                return board.getFoundation().stream().allMatch(List::isEmpty)
                        ? List.of(Move.tryParse("move T1 A♣ F1"), Move.tryParse("move T2 A♥ F2"))
                        : List.of(Move.quit());
            });
            for (int i = 0; i < 2; i++) {
                assertEquals(0.0, search(new MonteCarloTreeNode(board), random, 3));
                assertSame(pool, board.getUnknownCards());
                assertEquals(0, board.getMoveHistorySize());
            }
        }
        assertEquals(2 * pool.size(), draws.get());
        assertEquals(6, arrangements.size());
        assertEquals(arrangements.get(0), arrangements.get(1));
        assertEquals(arrangements.get(0), arrangements.get(2));
        assertEquals(arrangements.get(3), arrangements.get(4));
        assertEquals(arrangements.get(3), arrangements.get(5));
        assertNotEquals(arrangements.get(0), arrangements.get(3));
        verify(board, never()).copy();
    }

    @Test
    void searchExceptionsRestoreAssignmentsAndPreserveTheCandidatesExistingMove() {
        Solitaire board = spy(SolitaireFactory.stockOnly().planningCopy());
        board.turnThree();
        int checkpoint = board.getMoveHistorySize();
        long key = board.getStateKey();
        List<Card> pool = board.getUnknownCards();
        MonteCarloTreeNode candidate = new MonteCarloTreeNode(board);
        candidate.setMove(Move.turn());
        RuntimeException failure = new IllegalStateException("Search interrupted");
        AtomicInteger visits = new AtomicInteger();
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation -> {
                if (visits.incrementAndGet() == 2) throw failure;
                return List.of(Move.turn());
            });
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> search(candidate, new Random(42), 10)));
        }
        assertSame(pool, board.getUnknownCards());
        assertTrue(board.getUnknownCardGuesses().isEmpty());
        assertEquals(checkpoint, board.getMoveHistorySize());
        assertEquals(key, board.getStateKey());
        assertEquals(key, candidate.getStateKey());
        assertEquals(Move.turn(), candidate.getMove());
        verify(board, never()).copy();
    }

    private static double searchAndAssertRestored(Solitaire board, int budget) {
        long key = board.getStateKey();
        int checkpoint = board.getMoveHistorySize();
        List<Card> pool = board.getUnknownCards();
        var guesses = Map.copyOf(board.getUnknownCardGuesses());
        doThrow(new AssertionError("DFS must not copy boards")).when(board).copy();
        double reward = search(new MonteCarloTreeNode(board), new Random(42), budget);
        assertEquals(key, board.getStateKey());
        assertEquals(checkpoint, board.getMoveHistorySize());
        assertSame(pool, board.getUnknownCards());
        assertEquals(guesses, board.getUnknownCardGuesses());
        verify(board, never()).copy();
        return reward;
    }

    private static double search(MonteCarloTreeNode node, RandomGenerator random, int budget) {
        try {
            Method method = MonteCarloPlayer.class.getDeclaredMethod("simulateDepthFirst",
                    MonteCarloTreeNode.class, RandomGenerator.class, int.class);
            method.setAccessible(true);
            return (double) method.invoke(new MonteCarloPlayer(), node, random, budget);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException failure) throw failure;
            if (e.getCause() instanceof Error failure) throw failure;
            throw new AssertionError(e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }
}

package ai.games.unit.player;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.game.Solitaire;
import ai.games.game.Card;
import ai.games.game.UnknownCardGuess;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class MonteCarloRolloutTest {
    @Test
    void rolloutGeneratesAndAppliesMovesWithoutParsingCommands() {
        MonteCarloTreeNode node = new MonteCarloTreeNode(twoAces());
        try (var moves = mockStatic(Move.class, CALLS_REAL_METHODS)) {
            Simulation result = simulateOneStep(node, new IndexRandom(0));
            assertEquals(1, result.attemptedPositions().size());
            moves.verify(() -> Move.tryParse(anyString()), never());
        }
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void aceSymmetryRetainsEachSourceAndTriesOnlyTheSelectedRepresentative(Solitaire.GameMode mode) {
        Solitaire position = twoAces();
        if (mode == Solitaire.GameMode.PLAN) position = position.planningCopy();
        List<String> representatives = List.of("move T1 A♣ F1", "move W F1", "turn");
        assertEquals(10, LegalMovesHelper.listLegalMoves(position).size());

        for (int index = 0; index < representatives.size(); index++) {
            MonteCarloTreeNode node = new MonteCarloTreeNode(position.copy());
            IndexRandom random = mode == Solitaire.GameMode.PLAN && index == 2
                    ? new IndexRandom(index, 0) : new IndexRandom(index);

            Simulation result = simulateOneStep(node, random);

            if (mode == Solitaire.GameMode.PLAN && index == 2) {
                assertEquals(List.of(3, 49), random.bounds);
            } else {
                assertEquals(List.of(3), random.bounds);
            }
            assertEquals(1, result.attemptedPositions().size());
            assertPositionAfterMove(position, representatives.get(index), result.attemptedPositions().get(0));
        }
    }

    @Test
    void kingPruningKeepsARevealingMoveAndRemovesUselessShufflesBeforeApplying() {
        Solitaire state = SolitaireBuilder.newGame()
                .tableau("T1", 1, "3♣", "K♦").tableau("T2", "K♥").build();
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(new MonteCarloTreeNode(state), random);

        assertEquals(List.of(2), random.bounds); // One revealing king destination and turn.
        assertEquals(1, result.attemptedPositions().size());
        assertPositionAfterMove(state, "move T1 K♦ T3", result.attemptedPositions().get(0));
    }

    @Test
    void firstRolloutStepRemembersThePreviousMoveForInversePruning() {
        Solitaire state = SolitaireBuilder.newGame()
                .foundation("F1", SolitaireFactory.foundationUpTo(Suit.CLUBS, Rank.QUEEN))
                .tableau("T1", "K♥").build();
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        node.applyMove("move F1 T1");
        assertTrue(LegalMovesHelper.listLegalMoves(state).contains("move T1 Q♣ F1"));
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(1), random.bounds); // Only turn survives inverse/king pruning.
        assertEquals(1, result.attemptedPositions().size());
        assertPositionAfterMove(state, "turn", result.attemptedPositions().get(0));
    }

    @Test
    void cyclicCandidateIsDiscardedAndNextCandidateIsTriedWithoutReplacement() {
        Solitaire state = twoAces();
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        addRepeatedResultToHistory(node, "move T1 A♣ F1");
        IndexRandom random = new IndexRandom(0, 0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(3, 2), random.bounds);
        assertEquals(2, result.attemptedPositions().size());
        assertPositionAfterMove(state, "move T1 A♣ F1", result.attemptedPositions().get(0));
        assertPositionAfterMove(state, "turn", result.attemptedPositions().get(1));
        assertEquals(result.attemptedPositions().get(1).score()
                / MonteCarloTreeNode.MAX_SCORE, result.reward());
    }

    @Test
    void exhaustedCyclicCandidatesEndTheRollout() {
        MonteCarloTreeNode node = new MonteCarloTreeNode(SolitaireFactory.stockOnly());
        addRepeatedResultToHistory(node, "turn");
        assertTrue(node.evaluate() > 0);
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(1), random.bounds);
        assertEquals(1, result.attemptedPositions().size());
        assertEquals(0.0, result.reward());
    }

    @Test
    void aSinglePreviousOccurrenceDoesNotRejectACandidate() {
        Solitaire state = SolitaireFactory.stockOnly();
        MonteCarloTreeNode previous = new MonteCarloTreeNode(state.copy());
        previous.applyMove("turn");
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        node.setParent(previous);
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(1), random.bounds);
        assertEquals(1, result.attemptedPositions().size());
        assertPositionAfterMove(state, "turn", result.attemptedPositions().get(0));
    }

    @Test
    void aWonBoardNeedsNoMovesOrRandomDraws() {
        Simulation result = simulateOneStep(new MonteCarloTreeNode(SolitaireFactory.wonGame()),
                new IndexRandom());
        assertTrue(result.attemptedPositions().isEmpty());
        assertEquals(1.0, result.reward());
    }

    @Test
    void aWinOnTheLastPermittedStepGetsTheFullRewardAndRollsBack() {
        Simulation result = simulateOneStep(new MonteCarloTreeNode(SolitaireFactory.oneMoveFromWin()),
                new IndexRandom(0));

        assertEquals(1, result.attemptedPositions().size());
        assertEquals(52, result.attemptedPositions().get(0).foundations().stream()
                .mapToInt(List::size).sum());
        assertEquals(1.0, result.reward());
    }

    @Test
    void aStepLimitKeepsTheProgressEstimateWithoutClaimingAWinOrLoss() {
        MonteCarloTreeNode node = new MonteCarloTreeNode(SolitaireFactory.oneMoveFromWin());

        double reward = simulate(node, new IndexRandom(), 0);

        assertEquals(node.evaluate() / MonteCarloTreeNode.MAX_SCORE, reward);
        assertTrue(reward > 0.0 && reward < 1.0);
    }

    @Test
    void sampledDeadEndsDoNotRejectTheCandidateOrStopFurtherRollouts() throws Exception {
        MonteCarloPlayer player = new MonteCarloPlayer();
        // Expose a root action, then exhaust each rollout's continuations. This isolates
        // the sampling contract from random hidden identities and move ordering.
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(any(Solitaire.class)))
                    .thenReturn(new ArrayList<>(List.of(Move.turn())), List.of(Move.quit()));

            assertEquals("turn", player.nextCommand(SolitaireFactory.stockOnly(), "", ""));
        }

        Field field = MonteCarloPlayer.class.getDeclaredField("current");
        field.setAccessible(true);
        MonteCarloTreeNode candidate = (MonteCarloTreeNode) field.get(player);
        assertEquals(64, candidate.getVisits());
        assertEquals(0.0, candidate.getMeanReward());
        assertTrue(candidate.getState().getUnknownCardGuesses().isEmpty());
    }

    @Test
    void playerStillFinishesANearlyWonGame() {
        Solitaire state = SolitaireFactory.oneMoveFromWin();
        String command = new MonteCarloPlayer().nextCommand(state, "", "");
        MonteCarloTreeNode result = new MonteCarloTreeNode(state);
        result.applyMove(command);
        assertTrue(result.isWon());
    }

    @Test
    void repeatedRolloutsMatchTheCopyingReferenceAndRestoreTheCandidate() {
        Solitaire board = spy(SolitaireBuilder.newGame()
                .tableau("T1", 1, "3♣", "K♦").tableau("T2", "Q♣")
                .tableau("T3", "A♥").waste("A♠").build());
        MonteCarloTreeNode node = new MonteCarloTreeNode(board);
        // Preserve a non-empty history boundary across every rollout.
        node.applyMove("move T3 F1");
        Board before = snapshot(board);
        int history = board.getMoveHistorySize();
        Move move = node.getMove();
        for (int seed = 0; seed < 8; seed++) {
            double expected = copyingReference(node, new Random(seed), 32);
            clearInvocations(board);
            assertEquals(expected, simulate(node, new Random(seed), 32), "Seed " + seed);
            verify(board, never()).copy();
            assertEquals(before, snapshot(board));
            assertEquals(history, board.getMoveHistorySize());
            assertSame(move, node.getMove());
            assertEquals(before.key(), node.getStateKey());
            assertTrue(node.getChildren().isEmpty());
        }
    }

    @Test
    void rollbackRunsEvenWhenARolloutThrowsAfterApplyingAMove() {
        Solitaire board = spy(twoAces());
        MonteCarloTreeNode node = new MonteCarloTreeNode(board);
        Board before = snapshot(board);
        RuntimeException failure = new IllegalStateException("Interrupted rollout");
        Random random = new Random() {
            private int draws;
            @Override public int nextInt(int bound) {
                if (draws++ == 0) return 0;
                throw failure;
            }
        };
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> simulate(node, random, 3)));
        assertEquals(before, snapshot(board));
        assertEquals(0, board.getMoveHistorySize());
        verify(board, never()).copy();
    }

    @Test
    void planGuessesDoNotLeakBetweenRolloutsOrAfterExceptions() {
        Solitaire source = SolitaireBuilder.newGame()
                .tableau("T1", 1, "K♦", "A♣").tableau("T2", "5♠").build();
        source = source.planningCopy();
        Solitaire board = spy(source.copy());
        assertTrue(board.moveCard("T1", "F1")); // Expose an UNKNOWN destination.
        assertTrue(board.getUnknownCardGuesses().isEmpty());
        MonteCarloTreeNode node = new MonteCarloTreeNode(board);
        Board before = snapshot(board);
        int history = board.getMoveHistorySize();
        Random inspectGuesses = new Random() {
            @Override public int nextInt(int bound) {
                return bound - 1;
            }
        };
        double reward = simulate(node, inspectGuesses, 1);
        assertTrue(board.getUnknownCardGuesses().isEmpty());
        assertEquals(reward, simulate(node, inspectGuesses, 1));
        RuntimeException failure = new IllegalStateException("Interrupted after guessing");
        Random throwing = new Random() {
            @Override public int nextInt(int bound) {
                if (board.getUnknownCardGuesses().isEmpty()) return 0;
                throw failure;
            }
        };
        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> simulate(node, throwing, 1)));
        assertTrue(board.getUnknownCardGuesses().isEmpty());
        assertEquals(before, snapshot(board));
        assertEquals(history, board.getMoveHistorySize());
        verify(board, never()).copy();
    }

    @Test
    void rollbackRetainsGuessesForSeparateUnknownPositions() {
        Solitaire board = SolitaireBuilder.newGame()
                .tableau("T1", 1, "K♦", "A♣").tableau("T2", 1, "Q♣", "5♠")
                .build().planningCopy();
        Card first = board.getTableau().get(0).get(0);
        Card second = board.getTableau().get(1).get(0);
        UnknownCardGuess a = new UnknownCardGuess(first, new Card(Rank.KING, Suit.DIAMONDS));
        UnknownCardGuess b = new UnknownCardGuess(second, new Card(Rank.QUEEN, Suit.CLUBS));
        board.getUnknownCardGuesses().put(first, a);
        board.getUnknownCardGuesses().put(second, b);

        simulate(new MonteCarloTreeNode(board), new Random(42), 1);

        assertEquals(2, board.getUnknownCardGuesses().size());
        assertSame(a, board.getUnknownCardGuesses().get(first));
        assertSame(b, board.getUnknownCardGuesses().get(second));
    }

    @Test
    void rejectedCandidateDoesNotUndoTheRootMoveOrExistingHistory() {
        Solitaire board = twoAces();
        board.turnThree();
        MonteCarloTreeNode node = new MonteCarloTreeNode(board);
        addRepeatedResultToHistory(node, "move T1 A♣ F1");
        int history = board.getMoveHistorySize();
        Board before = snapshot(board);
        Simulation result = simulateOneStep(node, new IndexRandom(0, 0));
        assertEquals(2, result.attemptedPositions().size());
        assertEquals(before, snapshot(board));
        assertEquals(history, board.getMoveHistorySize());
    }

    @Test
    void decisionCopiesOnlyTheRootAndEachFirstLegalMove() throws Exception {
        Solitaire initial = SolitaireFactory.twoMovesFromWin();
        Solitaire source = spy(initial);
        Board before = snapshot(source);
        int rootMoves = (int) LegalMovesHelper.listMoves(initial).stream()
                .filter(move -> !move.isQuit()).count();
        List<Solitaire> boards = new ArrayList<>();
        Solitaire planning = spy(initial.planningCopy());
        boards.add(planning);
        doReturn(planning).when(source).planningCopy();
        doAnswer(invocation -> {
            Solitaire copy = (Solitaire) invocation.callRealMethod();
            boards.add(copy);
            return copy;
        }).when(planning).copy();
        MonteCarloPlayer player = new MonteCarloPlayer();
        assertNotNull(player.nextCommand(source, "", ""));
        verify(source).planningCopy();
        verify(source, never()).copy();
        verify(planning, times(rootMoves)).copy();
        assertEquals(rootMoves + 1, boards.size());
        assertEquals(before, snapshot(source));
        assertEquals(0, source.getMoveHistorySize());

        Field field = MonteCarloPlayer.class.getDeclaredField("root");
        field.setAccessible(true);
        MonteCarloTreeNode root = (MonteCarloTreeNode) field.get(player);
        assertEquals(rootMoves, root.getChildren().size());
        int eligible = 0;
        int rolloutVisits = 0;
        for (var entry : root.getChildren().entrySet()) {
            MonteCarloTreeNode candidate = (MonteCarloTreeNode) entry.getValue();
            assertPositionAfterMove(planning, entry.getKey(), snapshot(candidate.getState()));
            assertEquals(1, candidate.getState().getMoveHistorySize());
            assertTrue(candidate.getChildren().isEmpty());
            if (candidate.getMeanReward() == -1.0) {
                assertEquals(1, candidate.getVisits());
            } else {
                assertTrue(candidate.getVisits() >= 1);
                eligible++;
                rolloutVisits += candidate.getVisits();
            }
        }
        assertEquals(64 * eligible, rolloutVisits);
        assertEquals(rolloutVisits, root.getVisits());
        Field currentField = MonteCarloPlayer.class.getDeclaredField("current");
        currentField.setAccessible(true);
        MonteCarloTreeNode chosen = (MonteCarloTreeNode) currentField.get(player);
        double bestMean = root.getChildren().values().stream()
                .mapToDouble(child -> ((MonteCarloTreeNode) child).getMeanReward()).max().orElseThrow();
        assertEquals(bestMean, chosen.getMeanReward());
    }

    /** Copying traversal oracle with the same terminal rewards as the mutable rollout. */
    private static double copyingReference(MonteCarloTreeNode node, RandomGenerator random, int maxSteps) {
        if (node.isQuit() || node.isUselessKingMove() || node.isCycleDetected()) return -1.0;
        MonteCarloTreeNode leaf = new MonteCarloTreeNode(node.getState().copy());
        leaf.setParent(node);
        leaf.setMove(node.getMove());
        for (int steps = 0; steps < maxSteps && !leaf.isWon(); steps++) {
            List<Move> candidates = new ArrayList<>();
            for (Move candidate : LegalMovesHelper.listMoves(leaf.getState())) {
                if (!candidate.isQuit() && !leaf.isInverseOfLastMove(candidate)
                        && !leaf.isUselessKingMove(candidate)
                        && !leaf.isSimilarSibling(candidate, candidates)) candidates.add(candidate);
            }
            MonteCarloTreeNode selected = null;
            while (!candidates.isEmpty()) {
                int index = random.nextInt(candidates.size());
                Move candidate = candidates.get(index);
                Move last = candidates.remove(candidates.size() - 1);
                if (index < candidates.size()) candidates.set(index, last);
                MonteCarloTreeNode child = new MonteCarloTreeNode(leaf.getState().copy());
                child.setParent(leaf);
                child.applyMove(candidate);
                if (child.isCycleDetected()) continue;
                selected = child;
                break;
            }
            if (selected == null) return 0.0;
            leaf.addChild(selected.getMove().toCommandString(), selected);
            leaf = selected;
        }
        if (leaf.isWon()) return 1.0;
        return Math.max(0.0, Math.min(Math.nextDown(1.0),
                leaf.evaluate() / MonteCarloTreeNode.MAX_SCORE));
    }

    private static Solitaire twoAces() {
        return SolitaireBuilder.newGame()
                .tableau("T1", "A♣").tableau("T2", "7♠").waste("A♥").build();
    }

    private static void addRepeatedResultToHistory(MonteCarloTreeNode node, String command) {
        MonteCarloTreeNode earlier = new MonteCarloTreeNode(node.getState().copy());
        earlier.applyMove(command);
        MonteCarloTreeNode later = new MonteCarloTreeNode(earlier.getState().copy());
        later.setParent(earlier);
        node.setParent(later);
    }

    private static void assertPositionAfterMove(Solitaire initial, String command, Board actual) {
        MonteCarloTreeNode expected = new MonteCarloTreeNode(initial.copy());
        expected.applyMove(command);
        assertEquals(snapshot(expected.getState()), actual);
    }

    /** Exercise the actual player while observing move results before they are rolled back. */
    private static Simulation simulateOneStep(MonteCarloTreeNode node, RandomGenerator random) {
        Solitaire board = spy(node.getState());
        node.setState(board);
        Board before = snapshot(board);
        int history = board.getMoveHistorySize();
        Map<Card, UnknownCardGuess> guesses = Map.copyOf(board.getUnknownCardGuesses());
        Move originalMove = node.getMove();
        long originalKey = node.getStateKey();
        doThrow(new AssertionError("Rollouts must not copy boards")).when(board).copy();
        List<Board> attempts = new ArrayList<>();
        doAnswer(invocation -> {
            Object result = invocation.callRealMethod();
            attempts.add(snapshot(board));
            return result;
        }).when(board).attemptMove(any(), any(), any());
        doAnswer(invocation -> {
            invocation.callRealMethod();
            attempts.add(snapshot(board));
            return null;
        }).when(board).turnThree();

        double reward = simulate(node, random, 1);
        verify(board, never()).copy();
        assertEquals(before, snapshot(board));
        assertEquals(history, board.getMoveHistorySize());
        assertEquals(guesses, board.getUnknownCardGuesses());
        assertSame(originalMove, node.getMove());
        assertEquals(originalKey, node.getStateKey());
        assertTrue(node.getChildren().isEmpty(), "Rollout paths must not attach to the retained tree");
        return new Simulation(reward, attempts);
    }

    private static double simulate(MonteCarloTreeNode node, RandomGenerator random, int maxSteps) {
        try {
            Method method = MonteCarloPlayer.class.getDeclaredMethod(
                    "simulateRandom", MonteCarloTreeNode.class, RandomGenerator.class, int.class);
            method.setAccessible(true);
            return (double) method.invoke(new MonteCarloPlayer(), node, random, maxSteps);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException failure) throw failure;
            if (e.getCause() instanceof Error failure) throw failure;
            throw new AssertionError(e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    private static Board snapshot(Solitaire state) {
        return new Board(state.getMode(), state.getTableau().stream().map(List::copyOf).toList(),
                state.getFoundation().stream().map(List::copyOf).toList(),
                List.copyOf(state.getTableauFaceUpCounts()), List.copyOf(state.getStockpile()),
                List.copyOf(state.getTalon()), new HashSet<>(state.getUnknownCards()),
                state.getStateKey(), new MonteCarloTreeNode(state).evaluate());
    }

    private record Board(Solitaire.GameMode mode, List<List<Card>> tableau,
                         List<List<Card>> foundations, List<Integer> faceUp, List<Card> stock,
                         List<Card> talon, Set<Card> unknown, long key, int score) {}

    private record Simulation(double reward, List<Board> attemptedPositions) {}

    private static class IndexRandom extends Random {
        private final int[] indices;
        private final List<Integer> bounds = new ArrayList<>();

        IndexRandom(int... indices) {
            this.indices = indices;
        }

        @Override
        public int nextInt(int bound) {
            int draw = bounds.size();
            assertTrue(draw < indices.length, "Unexpected extra random draw");
            int index = indices[draw];
            assertTrue(index >= 0 && index < bound, "Scripted draw is outside the candidate range");
            bounds.add(bound);
            return index;
        }
    }
}

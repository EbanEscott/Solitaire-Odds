package ai.games.unit.player;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.game.Solitaire;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
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
            assertEquals(1, result.candidateBoards().size());
            moves.verify(() -> Move.tryParse(anyString()), never());
        }
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void aceSymmetryRetainsEachSourceAndCopiesOnlyTheSelectedRepresentative(Solitaire.GameMode mode) {
        Solitaire position = twoAces();
        position.setMode(mode);
        List<String> representatives = List.of("move T1 A♣ F1", "move W F1", "turn");
        assertEquals(10, LegalMovesHelper.listLegalMoves(position).size());

        for (int index = 0; index < representatives.size(); index++) {
            MonteCarloTreeNode node = new MonteCarloTreeNode(position.copy());
            IndexRandom random = new IndexRandom(index);

            Simulation result = simulateOneStep(node, random);

            assertEquals(List.of(3), random.bounds);
            assertEquals(1, result.candidateBoards().size());
            assertPositionAfterMove(position, representatives.get(index), result.candidateBoards().get(0));
        }
    }

    @Test
    void kingPruningKeepsARevealingMoveAndRemovesUselessShufflesBeforeCopying() {
        Solitaire state = SolitaireBuilder.newGame()
                .tableau("T1", 1, "3♣", "K♦").tableau("T2", "K♥").build();
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(new MonteCarloTreeNode(state), random);

        assertEquals(List.of(2), random.bounds); // One revealing king destination and turn.
        assertEquals(1, result.candidateBoards().size());
        assertPositionAfterMove(state, "move T1 K♦ T3", result.candidateBoards().get(0));
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
        assertEquals(1, result.candidateBoards().size());
        assertPositionAfterMove(state, "turn", result.candidateBoards().get(0));
    }

    @Test
    void cyclicCandidateIsDiscardedAndNextCandidateIsTriedWithoutReplacement() {
        Solitaire state = twoAces();
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        addRepeatedResultToHistory(node, "move T1 A♣ F1");
        IndexRandom random = new IndexRandom(0, 0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(3, 2), random.bounds);
        assertEquals(2, result.candidateBoards().size());
        assertPositionAfterMove(state, "move T1 A♣ F1", result.candidateBoards().get(0));
        assertPositionAfterMove(state, "turn", result.candidateBoards().get(1));
        assertEquals(new MonteCarloTreeNode(result.candidateBoards().get(1)).evaluate()
                / MonteCarloTreeNode.MAX_SCORE, result.reward());
    }

    @Test
    void exhaustedCyclicCandidatesEndTheRollout() {
        MonteCarloTreeNode node = new MonteCarloTreeNode(SolitaireFactory.stockOnly());
        addRepeatedResultToHistory(node, "turn");
        double initialReward = node.evaluate() / MonteCarloTreeNode.MAX_SCORE;
        IndexRandom random = new IndexRandom(0);

        Simulation result = simulateOneStep(node, random);

        assertEquals(List.of(1), random.bounds);
        assertEquals(1, result.candidateBoards().size());
        assertEquals(initialReward, result.reward());
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
        assertEquals(1, result.candidateBoards().size());
        assertPositionAfterMove(state, "turn", result.candidateBoards().get(0));
    }

    @Test
    void aWonBoardNeedsNoCandidateCopiesOrRandomDraws() {
        Simulation result = simulateOneStep(new MonteCarloTreeNode(SolitaireFactory.wonGame()),
                new IndexRandom());
        assertTrue(result.candidateBoards().isEmpty());
    }

    @Test
    void playerStillFinishesANearlyWonGame() {
        Solitaire state = SolitaireFactory.oneMoveFromWin();
        String command = new MonteCarloPlayer().nextCommand(state, "", "");
        MonteCarloTreeNode result = new MonteCarloTreeNode(state);
        result.applyMove(command);
        assertTrue(result.isWon());
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

    private static void assertPositionAfterMove(Solitaire initial, String command, Solitaire actual) {
        MonteCarloTreeNode expected = new MonteCarloTreeNode(initial.copy());
        expected.applyMove(command);
        assertEquals(expected.getState().toString(), actual.toString());
        assertEquals(initial.getMode(), actual.getMode());
    }

    /** Exercise the player's actual simulation with a bounded budget and scripted random draws. */
    private static Simulation simulateOneStep(MonteCarloTreeNode node, RandomGenerator random) {
        Solitaire source = spy(node.getState());
        String before = source.toString();
        Solitaire rollout = spy(source.copy());
        String rolloutBefore = rollout.toString();
        node.setState(source);
        doReturn(rollout).when(source).copy();
        clearInvocations(source);
        List<Solitaire> copies = new ArrayList<>();
        doAnswer(invocation -> {
            Solitaire copy = (Solitaire) invocation.callRealMethod();
            copies.add(copy);
            return copy;
        }).when(rollout).copy();

        try {
            Method simulate = MonteCarloPlayer.class.getDeclaredMethod(
                    "simulate", MonteCarloTreeNode.class, RandomGenerator.class, int.class);
            simulate.setAccessible(true);
            double reward = (double) simulate.invoke(new MonteCarloPlayer(), node, random, 1);
            verify(source, times(1)).copy(); // Independent rollout root, separate from candidate copies.
            assertEquals(before, source.toString());
            assertEquals(rolloutBefore, rollout.toString());
            assertTrue(node.getChildren().isEmpty(), "The simulated path must not attach to the real tree");
            return new Simulation(reward, copies);
        } catch (InvocationTargetException e) {
            throw new AssertionError("Simulation failed", e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    private record Simulation(double reward, List<Solitaire> candidateBoards) {}

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

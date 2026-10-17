package ai.games.unit.player;

import ai.games.game.*;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.mcts.*;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class MonteCarloResearchTest {
    @ParameterizedTest
    @ValueSource(strings = {"0.0", "0.25", "0.5"})
    void cutoffRewardChangesOnlyUnresolvedSearchAndRestoresTheBoard(String reward) throws Exception {
        String saved = System.getProperty("montecarlo.search.cutoffReward");
        try {
            System.setProperty("montecarlo.search.cutoffReward", reward);
            MonteCarloPlayer player = player(1);
            Solitaire board = SolitaireFactory.twoMovesFromWin().planningCopy();
            long key = board.getStateKey();
            Method search = MonteCarloPlayer.class.getDeclaredMethod("simulateDepthFirst",
                    MonteCarloTreeNode.class, RandomGenerator.class, int.class);
            search.setAccessible(true);
            assertEquals(Double.parseDouble(reward), search.invoke(player,
                    new MonteCarloTreeNode(board), new Random(19), 1));
            assertEquals(key, board.getStateKey());
            assertEquals(0, board.getMoveHistorySize());
            assertEquals(1.0, search.invoke(player, new MonteCarloTreeNode(board), new Random(19), 64));
            assertEquals(key, board.getStateKey());
            assertEquals(0, board.getMoveHistorySize());
        } finally {
            restore("montecarlo.search.cutoffReward", saved);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"NaN", "Infinity", "-0.1", "1.1"})
    void rejectsInvalidCutoffRewards(String reward) {
        String saved = System.getProperty("montecarlo.search.cutoffReward");
        try {
            System.setProperty("montecarlo.search.cutoffReward", reward);
            assertThrows(IllegalArgumentException.class, () -> new MonteCarloPlayer(new Random(19)));
        } finally {
            restore("montecarlo.search.cutoffReward", saved);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"0.0", "0.25"})
    void lowerCutoffRewardsRemainBlindToRealHiddenCards(String reward) throws Exception {
        String saved = System.getProperty("montecarlo.search.cutoffReward");
        try {
            System.setProperty("montecarlo.search.cutoffReward", reward);
            equalObservationsAndSamplingSeedsGiveIdenticalDecisionsAndRewards(1);
        } finally {
            restore("montecarlo.search.cutoffReward", saved);
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
    void equalObservationsAndSamplingSeedsGiveIdenticalDecisionsAndRewards(int flags) throws Exception {
        Solitaire a = SolitaireBuilder.newGame().tableau("T1", 1, "5♣", "A♠").build();
        Solitaire b = SolitaireBuilder.newGame().tableau("T1", 1, "7♦", "A♠").build();
        long keyA = a.getStateKey(), keyB = b.getStateKey();
        assertNotEquals(keyA, keyB);
        MonteCarloPlayer first = player(flags), second = player(flags);
        assertEquals(first.nextCommand(a, "", ""), second.nextCommand(b, "", ""));
        assertEquals(rewards(first), rewards(second));
        assertEquals(keyA, a.getStateKey());
        assertEquals(keyB, b.getStateKey());
    }

    @Test
    void followsAKnownWinningContinuationWithoutAnotherSearch() throws Exception {
        Solitaire game = SolitaireFactory.twoMovesFromWin();
        MonteCarloPlayer player = player(4);
        String first = player.nextCommand(game, "", "");
        new MonteCarloTreeNode(game).applyMove(first);
        assertFalse(continuation(player).isEmpty());
        String second = player.nextCommand(game, "", "");
        new MonteCarloTreeNode(game).applyMove(second);
        assertTrue(new MonteCarloTreeNode(game).isWon());
        assertTrue(continuation(player).isEmpty());
        Field field = MonteCarloPlayer.class.getDeclaredField("current"); field.setAccessible(true);
        assertEquals(0, ((MonteCarloTreeNode) field.get(player)).getVisits());
    }

    @Test
    void aWinningSampleWithAnUnobservedCardCannotBecomeARetainedSolution() throws Exception {
        Solitaire game = SolitaireBuilder.newGame().tableau("T1", 1, "K♣", "K♥")
                .foundation("F1", SolitaireFactory.foundationUpTo(Suit.CLUBS, Rank.QUEEN))
                .foundation("F2", SolitaireFactory.foundationUpTo(Suit.DIAMONDS, Rank.KING))
                .foundation("F3", SolitaireFactory.foundationUpTo(Suit.SPADES, Rank.KING))
                .foundation("F4", SolitaireFactory.foundationUpTo(Suit.HEARTS, Rank.QUEEN))
                .stock().waste().build();
        MonteCarloPlayer player = player(4);
        assertEquals(1, game.planningCopy().getUnknownCards().size());
        assertEquals("move T1 K♥ F4", player.nextCommand(game, "", ""));
        assertTrue(continuation(player).isEmpty());
        Field field = MonteCarloPlayer.class.getDeclaredField("knownSolutions"); field.setAccessible(true);
        assertTrue(((Map<?, ?>) field.get(player)).isEmpty());
    }

    @Test
    void transpositionsRetainIncomingMoveContextAndResetBetweenSamples() throws Exception {
        Solitaire board = SolitaireBuilder.newGame().tableau("T1", "A♣")
                .tableau("T2", "A♥").tableau("T3", "A♠").build().planningCopy();
        long key = board.getStateKey();
        AtomicInteger joined = new AtomicInteger(), completed = new AtomicInteger();
        try (var moves = mockStatic(LegalMovesHelper.class)) {
            moves.when(() -> LegalMovesHelper.listMoves(board)).thenAnswer(invocation -> {
                boolean a = !board.getFoundation().get(0).isEmpty();
                boolean b = !board.getFoundation().get(1).isEmpty();
                boolean c = !board.getFoundation().get(2).isEmpty();
                if (c) { completed.incrementAndGet(); return List.of(); }
                if (a && b) { joined.incrementAndGet(); return List.of(Move.tryParse("move T3 A♠ F3")); }
                if (a) return List.of(Move.tryParse("move T2 A♥ F2"));
                if (b) return List.of(Move.tryParse("move T1 A♣ F1"));
                return List.of(Move.tryParse("move T1 A♣ F1"), Move.tryParse("move T2 A♥ F2"));
            });
            assertEquals(0.0, search(player(0), board));
            assertEquals(2, completed.get());
            MonteCarloPlayer cached = player(2);
            for (int i = 0; i < 2; i++) {
                joined.set(0); completed.set(0);
                assertEquals(0.0, search(cached, board));
                assertEquals(2, joined.get(), "Different incoming moves retain different pruning contexts");
                assertEquals(1, completed.get(), "Same board and incoming move is searched once per sample");
                assertEquals(key, board.getStateKey());
                assertEquals(0, board.getMoveHistorySize());
                assertTrue(board.getUnknownCardGuesses().isEmpty());
            }
        }
    }

    @Test
    void revealOrderingRetainsEveryMoveAndPrioritisesAHiddenCardReveal() throws Exception {
        Solitaire board = SolitaireBuilder.newGame().tableau("T1", 1, "3♣", "Q♥")
                .tableau("T2", "K♠").tableau("T3", "A♣")
                .tableau("T4", "6♣").tableau("T5", "7♥").build().planningCopy();
        List<Move> ordinary = frameMoves(player(0), board);
        List<Move> ordered = frameMoves(player(1), board);
        assertEquals(new HashSet<>(ordinary), new HashSet<>(ordered));
        assertEquals(ordinary.size(), ordered.size());
        assertEquals("move T1 Q♥ T2", ordered.getFirst().toCommandString());
        assertTrue(ordered.indexOf(Move.turn()) < ordered.indexOf(Move.tryParse("move T4 6♣ T5")));
    }

    @Test
    void safeTableauPlacementsAreForcedButWastePlacementsAreNot() throws Exception {
        Solitaire tableauAce = SolitaireBuilder.newGame().tableau("T1", "A♣").build().planningCopy();
        assertEquals(List.of(Move.tryParse("move T1 A♣ F1")), frameMoves(player(8), tableauAce));
        Solitaire wasteAce = SolitaireBuilder.newGame().waste("A♣")
                .tableau("T1", "6♣").tableau("T2", "7♥").build().planningCopy();
        assertEquals(frameMoves(player(0), wasteAce), frameMoves(player(8), wasteAce));
    }

    @Test
    void safeFoundationWithdrawalsAreExcludedInsideDfs() throws Exception {
        Solitaire board = SolitaireBuilder.newGame().foundation("F1", "A♣", "2♣", "3♣")
                .foundation("F2", "A♦", "2♦").foundation("F3", "A♥", "2♥")
                .foundation("F4", "A♠").tableau("T1", "4♦").build().planningCopy();
        Move withdrawal = Move.tryParse("move F1 T1");
        assertTrue(frameMoves(player(0), board).contains(withdrawal));
        assertFalse(frameMoves(player(8), board).contains(withdrawal));
    }

    @Test
    void higherRanksNeedBothOppositeSuitsAndTheOtherSameColourSuit() throws Exception {
        Method method = MonteCarloPlayer.class.getDeclaredMethod("isSafeFoundation", Card.class, int[].class);
        method.setAccessible(true);
        MonteCarloPlayer player = player(8);
        int[] heights = new int[Suit.values().length];
        heights[Suit.CLUBS.ordinal()] = 4;
        heights[Suit.SPADES.ordinal()] = 4;
        heights[Suit.DIAMONDS.ordinal()] = 3;
        Card five = new Card(Rank.FIVE, Suit.HEARTS);
        assertEquals(true, method.invoke(player, five, heights));
        heights[Suit.SPADES.ordinal()] = 3;
        assertEquals(false, method.invoke(player, five, heights));
        heights[Suit.SPADES.ordinal()] = 4;
        heights[Suit.DIAMONDS.ordinal()] = 2;
        assertEquals(false, method.invoke(player, five, heights));
    }

    private static MonteCarloPlayer player(int flags) {
        String order = System.getProperty("montecarlo.search.revealFirst");
        String transpositions = System.getProperty("montecarlo.search.transpositions");
        String solution = System.getProperty("montecarlo.search.reuseKnownSolution");
        String safe = System.getProperty("montecarlo.search.safeFoundations");
        try {
            System.setProperty("montecarlo.search.revealFirst", Boolean.toString((flags & 1) != 0));
            System.setProperty("montecarlo.search.transpositions", Boolean.toString((flags & 2) != 0));
            System.setProperty("montecarlo.search.reuseKnownSolution", Boolean.toString((flags & 4) != 0));
            System.setProperty("montecarlo.search.safeFoundations", Boolean.toString((flags & 8) != 0));
            return new MonteCarloPlayer(new Random(4242));
        } finally {
            restore("montecarlo.search.revealFirst", order);
            restore("montecarlo.search.transpositions", transpositions);
            restore("montecarlo.search.reuseKnownSolution", solution);
            restore("montecarlo.search.safeFoundations", safe);
        }
    }

    private static Deque<?> continuation(MonteCarloPlayer player) throws Exception {
        Field field = MonteCarloPlayer.class.getDeclaredField("knownContinuation"); field.setAccessible(true);
        return (Deque<?>) field.get(player);
    }

    private static void restore(String key, String value) {
        if (value == null) System.clearProperty(key); else System.setProperty(key, value);
    }

    private static Map<String, Double> rewards(MonteCarloPlayer player) throws Exception {
        Field field = MonteCarloPlayer.class.getDeclaredField("current"); field.setAccessible(true);
        var node = (MonteCarloTreeNode) field.get(player);
        Map<String, Double> result = new TreeMap<>();
        node.getParent().getChildren().forEach((command, child) ->
                result.put(command, ((MonteCarloTreeNode) child).getMeanReward()));
        return result;
    }

    private static double search(MonteCarloPlayer player, Solitaire board) throws Exception {
        Method method = MonteCarloPlayer.class.getDeclaredMethod("simulateDepthFirst",
                MonteCarloTreeNode.class, RandomGenerator.class, int.class);
        method.setAccessible(true);
        return (double) method.invoke(player, new MonteCarloTreeNode(board), new Random(19), 64);
    }

    @SuppressWarnings("unchecked")
    private static List<Move> frameMoves(MonteCarloPlayer player, Solitaire board) throws Exception {
        Method method = MonteCarloPlayer.class.getDeclaredMethod("searchFrame", MonteCarloTreeNode.class);
        method.setAccessible(true);
        Object frame = method.invoke(player, new MonteCarloTreeNode(board));
        Method moves = frame.getClass().getDeclaredMethod("moves"); moves.setAccessible(true);
        List<Move> result = new ArrayList<>();
        ((Iterator<Move>) moves.invoke(frame)).forEachRemaining(result::add);
        return result;
    }
}

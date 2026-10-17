package ai.games.unit.player;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.astar.AStarPlayer;
import ai.games.player.ai.astar.AStarTreeNode;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.tree.TreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class PlanningSearchTest {
    @Test
    void aStarEnumeratesSmallJointRevealsWithoutReplacement() throws Exception {
        Solitaire state = threeUnseenCards().planningCopy();
        state.turnThree();
        long key = state.getStateKey();
        AStarPlayer player = new AStarPlayer();
        AStarTreeNode node = new AStarTreeNode(state);
        node.setG(2);
        node.setProbability(0.25);
        expandReveal(player, node, newSearch());
        assertTrue(node.isChanceNode());
        assertEquals(6, node.getChildren().size());
        java.util.Set<List<Card>> permutations = new java.util.HashSet<>();
        double weight = 0;
        for (TreeNode branch : node.getChildren().values()) {
            AStarTreeNode child = (AStarTreeNode) branch;
            assertEquals(0.25 / 6, child.getProbability(), 1e-12);
            assertEquals(1.0 / 6, child.getOutcomeWeight(), 1e-12);
            assertEquals(2, child.getG());
            assertNull(child.getMove());
            assertNull(child.getState().nextUnresolvedCard());
            assertTrue(child.getState().getUnknownCards().isEmpty());
            assertEquals(3, child.getState().getTalon().stream().distinct().count());
            permutations.add(List.copyOf(child.getState().getTalon()));
            weight += child.getOutcomeWeight();
        }
        assertEquals(6, permutations.size());
        assertEquals(1, weight, 1e-12);
        assertEquals(key, state.getStateKey());
        assertEquals(3, state.getUnknownCards().size());
        assertTrue(state.getUnknownCardGuesses().isEmpty());
    }

    @Test
    void aStarSamplesCompleteStockOutcomesAndRetainsAllWeight() throws Exception {
        Solitaire state = SolitaireBuilder.newGame().build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null)).planningCopy();
        state.turnThree();
        AStarTreeNode node = new AStarTreeNode(state);
        expandReveal(new AStarPlayer(), node, newSearch());
        assertEquals(8, node.getChildren().size());
        double weight = 0;
        for (TreeNode branch : node.getChildren().values()) {
            AStarTreeNode child = (AStarTreeNode) branch;
            assertNull(child.getState().nextUnresolvedCard());
            assertEquals(3, child.getState().getTalon().stream().distinct().count());
            assertEquals(49, child.getState().getUnknownCards().size());
            assertEquals(1.0 / (52 * 51 * 50), child.getProbability(), 1e-12);
            assertEquals(1.0 / 8, child.getOutcomeWeight());
            weight += child.getOutcomeWeight();
        }
        assertEquals(1, weight);
        assertNotNull(state.nextUnresolvedCard());
    }

    @Test
    void aStarAveragesOutcomesBeforeChoosingAnAction() throws Exception {
        AStarTreeNode decision = costNode(100);
        decision.setExpanded(true);
        AStarTreeNode known = costNode(6);
        AStarTreeNode reveal = costNode(100);
        reveal.setExpanded(true);
        reveal.setChanceNode(true);
        AStarTreeNode lucky = costNode(0);
        lucky.setOutcomeWeight(1.0 / 3);
        lucky.setProbability(0.01); // Cumulative probability must not be applied a second time.
        AStarTreeNode other = costNode(12);
        other.setOutcomeWeight(2.0 / 3);
        reveal.addChild("lucky", lucky);
        reveal.addChild("other", other);
        decision.addChild("known", known);
        decision.addChild("reveal", reveal);
        assertEquals(8, evaluate(reveal), 1e-12);
        assertEquals(6, evaluate(decision), 1e-12);
        other.setH(3);
        other.recalculateF();
        assertEquals(2, evaluate(decision), 1e-12);
    }

    @Test
    void aStarKeepsHeuristicMassWhenAnOutcomeGroupDoesNotFit() throws Exception {
        Solitaire state = threeUnseenCards().planningCopy();
        state.turnThree();
        AStarTreeNode node = new AStarTreeNode(state);
        Object search = newSearch();
        setField(search, "nodes", 4996); // Six exact outcomes must fit together.
        expandReveal(new AStarPlayer(), node, search);
        assertFalse(node.isExpanded());
        assertTrue(node.getChildren().isEmpty());
        assertEquals(node.getF(), evaluate(node));
    }

    @Test
    void aStarDefersRevealsUntilAfterKnownExpansion() throws Exception {
        AStarPlayer player = new AStarPlayer();
        AStarTreeNode node = new AStarTreeNode(SolitaireFactory.stockOnly().planningCopy());
        Object search = newSearch();
        Method expand = AStarPlayer.class.getDeclaredMethod("expandKnown", AStarTreeNode.class, search.getClass());
        expand.setAccessible(true);
        expand.invoke(player, node, search);
        assertTrue(((PriorityQueue<?>) field(search, "known")).isEmpty());
        PriorityQueue<?> reveals = (PriorityQueue<?>) field(search, "reveals");
        assertEquals(1, reveals.size());
        AStarTreeNode turn = (AStarTreeNode) reveals.peek();
        assertNotNull(turn.getState().nextUnresolvedCard());
        assertTrue(turn.getChildren().isEmpty());
        assertTrue(turn.getState().getUnknownCardGuesses().isEmpty());
    }

    @Test
    void aStarOrdersKnownPositionsWithoutDividingByProbability() {
        AStarTreeNode node = costNode(7);
        node.setG(3);
        node.setProbability(1.0 / 1000);
        node.recalculateF();
        assertEquals(10, node.getF());
    }

    private static AStarTreeNode costNode(double cost) {
        AStarTreeNode node = new AStarTreeNode(threeUnseenCards().planningCopy());
        node.setH(cost);
        node.recalculateF();
        return node;
    }

    private static Object newSearch() throws Exception {
        var constructor = Class.forName(AStarPlayer.class.getName() + "$Search").getDeclaredConstructor();
        constructor.setAccessible(true);
        return constructor.newInstance();
    }

    private static void expandReveal(AStarPlayer player, AStarTreeNode node, Object search) throws Exception {
        Method expand = AStarPlayer.class.getDeclaredMethod("expandReveal", AStarTreeNode.class, search.getClass());
        expand.setAccessible(true);
        expand.invoke(player, node, search);
    }

    private static double evaluate(AStarTreeNode node) throws Exception {
        Method evaluate = AStarPlayer.class.getDeclaredMethod("evaluate", AStarTreeNode.class);
        evaluate.setAccessible(true);
        return (double) evaluate.invoke(new AStarPlayer(), node);
    }

    @Test
    void monteCarloResamplesRevealsWithoutReplacement() throws Exception {
        Solitaire state = threeUnseenCards().planningCopy();
        state.turnThree();
        long key = state.getStateKey();
        List<Card> unknowns = List.copyOf(state.getTalon());
        MonteCarloPlayer player = new MonteCarloPlayer();
        Method resolve = MonteCarloPlayer.class.getDeclaredMethod("resolveRevealedCards", Solitaire.class,
                java.util.random.RandomGenerator.class);
        resolve.setAccessible(true);
        int depth = state.getMoveHistorySize();
        resolve.invoke(player, state, new Random() {
            @Override public int nextInt(int bound) { return 0; }
        });
        List<Card> first = List.copyOf(state.getTalon());
        assertEquals(3, first.stream().distinct().count());
        state.rollbackTo(depth);
        resolve.invoke(player, state, new Random() {
            @Override public int nextInt(int bound) { return bound - 1; }
        });
        assertNotEquals(first, state.getTalon());
        state.rollbackTo(depth);
        assertEquals(key, state.getStateKey());
        for (int i = 0; i < 3; i++) assertSame(unknowns.get(i), state.getTalon().get(i));
        assertTrue(state.getUnknownCardGuesses().isEmpty());
    }

    @Test
    void bothPlayersSearchMaskedBoardsAndAcceptActualRevealsOnTheNextTurn() throws Exception {
        for (AIPlayer player : List.of(new AStarPlayer(), new MonteCarloPlayer())) {
            Solitaire real = spy(SolitaireFactory.stockOnly());
            long original = real.getStateKey();
            String command = player.nextCommand(real, "", "");
            assertTrue(LegalMovesHelper.listLegalMoves(real).contains(command));
            assertEquals("turn", command);
            assertEquals(original, real.getStateKey());
            verify(real).planningCopy();
            verify(real, never()).copy();
            TreeNode root = (TreeNode) field(player, "root");
            assertTrue(root.getState().isInPlanMode());
            assertTrue(root.getState().getStockpile().stream().allMatch(c -> c.getRank() == Rank.UNKNOWN));
            TreeNode applied = new AStarTreeNode(real);
            applied.applyMove(command);
            assertNotEquals(original, real.getStateKey());
            assertDoesNotThrow(() -> player.nextCommand(real, "", ""));
            TreeNode current = (TreeNode) field(player, "current");
            assertTrue(current.getState().isInPlanMode());
        }
    }

    private static Object field(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static Solitaire threeUnseenCards() {
        SolitaireBuilder builder = SolitaireBuilder.newGame();
        int pile = 1;
        for (Suit suit : List.of(Suit.CLUBS, Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS)) {
            int maximum = suit == Suit.DIAMONDS ? 10 : 13;
            builder.foundation("F" + pile++, Arrays.stream(Rank.values())
                    .filter(rank -> rank != Rank.UNKNOWN && rank.getValue() <= maximum)
                    .map(rank -> new Card(rank, suit).shortName()).toArray(String[]::new));
        }
        return builder.build(new ai.games.game.GameRules(3, ai.games.game.GameRules.StockVisibility.allDrawn, null));
    }
}

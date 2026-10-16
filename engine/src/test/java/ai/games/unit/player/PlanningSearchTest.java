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
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class PlanningSearchTest {
    @Test
    void aStarBranchesWithoutReplacementAndAccumulatesProbability() throws Exception {
        Solitaire state = threeUnseenCards().planningCopy();
        state.turnThree();
        AStarPlayer player = new AStarPlayer();
        setField(player, "openSet", new PriorityQueue<AStarTreeNode>());
        setField(player, "bestScore", new HashMap<Long, Double>());
        Method expand = AStarPlayer.class.getDeclaredMethod("expandReveal", AStarTreeNode.class);
        expand.setAccessible(true);
        AStarTreeNode node = new AStarTreeNode(state);
        node.setG(1);
        for (int remaining = 3; remaining >= 1; remaining--) {
            expand.invoke(player, node);
            assertEquals(remaining, node.getChildren().size());
            for (TreeNode branch : node.getChildren().values()) {
                AStarTreeNode child = (AStarTreeNode) branch;
                assertEquals(node.getProbability() / remaining, child.getProbability());
                assertEquals(remaining - 1, child.getState().getUnknownCards().size());
                assertEquals(node.getG(), child.getG());
                assertNull(child.getMove());
            }
            node = (AStarTreeNode) node.getChildren().values().iterator().next();
        }
        assertEquals(1.0 / 6, node.getProbability());
        assertNull(node.getState().nextUnresolvedCard());
        assertEquals(3, state.getUnknownCards().size());
        assertTrue(state.getUnknownCardGuesses().isEmpty());
    }

    @Test
    void aStarStillPenalisesSpeculativePathsWhenTheHeuristicIsZero() {
        AStarTreeNode node = new AStarTreeNode();
        node.setG(3);
        node.setH(0);
        node.setProbability(1.0 / 6);
        node.recalculateF();
        double speculative = node.getF();
        node.setProbability(1);
        node.recalculateF();
        assertTrue(speculative > node.getF());
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
        return builder.build();
    }
}

package ai.games.unit.player;

import ai.games.game.Solitaire;
import ai.games.player.ai.astar.AStarPlayer;
import ai.games.player.ai.astar.AStarTreeNode;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AStarResearchTest {
    @Test
    void balancedSearchSharesAttemptsBetweenRootActionsAndChanceOutcomes() throws Exception {
        String previous = System.getProperty("astar.balancedSearch");
        System.setProperty("astar.balancedSearch", "true");
        try {
            AStarPlayer player = new AStarPlayer();
            AStarTreeNode root = new AStarTreeNode(SolitaireFactory.oneMoveFromWin().planningCopy());
            var current = AStarPlayer.class.getDeclaredField("current");
            current.setAccessible(true);
            current.set(player, root);
            AStarTreeNode a = new AStarTreeNode(root.copyState()), b = new AStarTreeNode(root.copyState());
            root.addChild("a", a); root.addChild("b", b);
            AStarTreeNode a1 = new AStarTreeNode(root.copyState()), a2 = new AStarTreeNode(root.copyState());
            AStarTreeNode a3 = new AStarTreeNode(root.copyState());
            a.addChild("1", a1); a.addChild("2", a2); a.addChild("3", a3);
            a1.setOutcomeContext(1); a2.setOutcomeContext(1); a3.setOutcomeContext(2);
            a1.setF(0); a2.setF(1); a3.setF(50); b.setF(100);
            var queue = new java.util.PriorityQueue<AStarTreeNode>();
            queue.addAll(java.util.List.of(a1, a2, a3, b));
            Class<?> type = Class.forName(AStarPlayer.class.getName() + "$Search");
            var constructor = type.getDeclaredConstructor(); constructor.setAccessible(true);
            Object search = constructor.newInstance();
            Method poll = AStarPlayer.class.getDeclaredMethod("poll", java.util.PriorityQueue.class, type);
            poll.setAccessible(true);
            assertSame(a1, poll.invoke(player, queue, search));
            assertSame(b, poll.invoke(player, queue, search));
            assertSame(a3, poll.invoke(player, queue, search));
            assertSame(a2, poll.invoke(player, queue, search));
            assertTrue(queue.isEmpty());
        } finally {
            if (previous == null) System.clearProperty("astar.balancedSearch");
            else System.setProperty("astar.balancedSearch", previous);
        }
    }

    @Test
    void reuseLinksDuplicatesOnlyWithinTheirChanceContext() throws Exception {
        String[] names = {"astar.transpositions", "astar.reuseDuplicates"};
        String[] previous = java.util.Arrays.stream(names).map(System::getProperty).toArray(String[]::new);
        try {
            for (String name : names) System.setProperty(name, "true");
            AStarPlayer player = new AStarPlayer();
            Class<?> type = Class.forName(AStarPlayer.class.getName() + "$Search");
            var constructor = type.getDeclaredConstructor();
            constructor.setAccessible(true);
            Object search = constructor.newInstance();
            Method enqueue = AStarPlayer.class.getDeclaredMethod("enqueue", AStarTreeNode.class, type);
            enqueue.setAccessible(true);
            AStarTreeNode first = new AStarTreeNode(SolitaireFactory.oneMoveFromWin().planningCopy());
            first.setG(1);
            enqueue.invoke(player, first, search);
            AStarTreeNode duplicate = new AStarTreeNode(first.copyState());
            duplicate.setG(2);
            enqueue.invoke(player, duplicate, search);
            assertFalse(duplicate.isPruned());
            assertSame(first, duplicate.getEquivalent());
            AStarTreeNode separate = new AStarTreeNode(first.copyState());
            separate.setG(2);
            separate.setOutcomeContext(1);
            enqueue.invoke(player, separate, search);
            assertFalse(separate.isPruned());
            assertNull(separate.getEquivalent());
        } finally {
            for (int i = 0; i < names.length; i++) {
                if (previous[i] == null) System.clearProperty(names[i]);
                else System.setProperty(names[i], previous[i]);
            }
        }
    }

    @Test
    void duplicateValuesFollowContinuationWithCostOffsetAndTerminateCycles() throws Exception {
        AStarPlayer player = new AStarPlayer();
        Method evaluate = AStarPlayer.class.getDeclaredMethod("evaluate", AStarTreeNode.class);
        evaluate.setAccessible(true);
        AStarTreeNode representative = new AStarTreeNode(SolitaireFactory.oneMoveFromWin().planningCopy());
        representative.setG(2);
        representative.setH(10);
        representative.recalculateF();
        AStarTreeNode duplicate = new AStarTreeNode(representative.copyState());
        duplicate.setG(5);
        duplicate.setEquivalent(representative);
        assertEquals(15.0, (double) evaluate.invoke(player, duplicate));
        AStarTreeNode win = new AStarTreeNode(representative.copyState());
        win.applyMove("move T1 K♥ F4");
        win.setG(3);
        representative.addChild("win", win);
        representative.setExpanded(true);
        assertEquals(6.0, (double) evaluate.invoke(player, duplicate));
        representative.getChildren().clear();
        representative.addChild("cycle", duplicate);
        assertTrue(Double.isFinite((double) evaluate.invoke(player, duplicate)));
        assertTrue((double) evaluate.invoke(player, duplicate) >= 1000);
    }

    @Test
    void stockObservationPassRequiresRealTurnsAndStopsAfterOnePass() throws Exception {
        String previous = System.getProperty("astar.scanStock");
        System.setProperty("astar.scanStock", "true");
        try {
            Solitaire game = new Solitaire(new ai.games.game.Deck());
            AStarPlayer player = new AStarPlayer();
            assertEquals(45, game.planningCopy().getUnknownCards().size());
            for (int i = 0; i < 8; i++) {
                long before = game.getStateKey();
                assertEquals("turn", player.nextCommand(game, "", ""));
                assertEquals(before, game.getStateKey());
                game.turnThree();
            }
            assertTrue(game.getStockpile().isEmpty());
            assertEquals(37, game.planningCopy().getUnknownCards().size());
            player.nextCommand(game, "", "");
            var root = AStarPlayer.class.getDeclaredField("root");
            root.setAccessible(true);
            assertNotNull(root.get(player));
            player.reset();
            assertEquals("turn", player.nextCommand(new Solitaire(new ai.games.game.Deck()), "", ""));
            assertNull(root.get(player));
        } finally {
            if (previous == null) System.clearProperty("astar.scanStock");
            else System.setProperty("astar.scanStock", previous);
        }
    }

    @Test
    void hiddenArrangementsDoNotChangeDecisionsOrSearchValues() {
        String[] names = {"astar.transpositions", "astar.finishKnownWin", "astar.trace", "astar.reuseDuplicates", "astar.balancedSearch"};
        String[] previous = java.util.Arrays.stream(names).map(System::getProperty).toArray(String[]::new);
        try {
            System.setProperty("astar.trace", "true");
            for (int flags = 0; flags < 16; flags++) {
                System.setProperty("astar.transpositions", Boolean.toString((flags & 1) != 0 || (flags & 4) != 0));
                System.setProperty("astar.reuseDuplicates", Boolean.toString((flags & 4) != 0));
                System.setProperty("astar.balancedSearch", Boolean.toString((flags & 8) != 0));
                System.setProperty("astar.finishKnownWin", Boolean.toString((flags & 2) != 0));
                Solitaire a = ai.games.unit.helpers.SolitaireBuilder.newGame()
                        .tableau("T1", 1, "5♣", "A♠").build();
                Solitaire b = ai.games.unit.helpers.SolitaireBuilder.newGame()
                        .tableau("T1", 1, "7♦", "A♠").build();
                long keyA = a.getStateKey(), keyB = b.getStateKey();
                assertNotEquals(keyA, keyB);
                AStarPlayer first = new AStarPlayer(), second = new AStarPlayer();
                assertEquals(first.nextCommand(a, "", ""), second.nextCommand(b, "", ""));
                assertEquals(first.getDecisionTrace(), second.getDecisionTrace());
                assertEquals(keyA, a.getStateKey());
                assertEquals(keyB, b.getStateKey());
            }
        } finally {
            for (int i = 0; i < names.length; i++) {
                if (previous[i] == null) System.clearProperty(names[i]);
                else System.setProperty(names[i], previous[i]);
            }
        }
    }

    @Test
    void hillScorePresetMatchesTheExistingHillClimberEvaluation() throws Exception {
        var hill = new ai.games.player.ai.HillClimberPlayer(42);
        Method evaluate = hill.getClass().getDeclaredMethod("evaluate", Solitaire.class);
        evaluate.setAccessible(true);
        for (Solitaire game : java.util.List.of(SolitaireFactory.stockOnly(),
                SolitaireFactory.twoMovesFromWin(), new Solitaire(new ai.games.game.Deck()))) {
            int score = (int) evaluate.invoke(hill, game);
            assertEquals((1040.0 - score) / 5.0,
                    AStarTreeNode.computeHeuristic(game.planningCopy(), 3, 1.2, 1.6, 0, 1), 1e-9);
        }
    }

    @Test
    void transpositionsNeverDiscardAnotherChanceOutcomesContribution() throws Exception {
        String previous = System.getProperty("astar.transpositions");
        System.setProperty("astar.transpositions", "true");
        try {
            AStarPlayer player = new AStarPlayer();
            Class<?> type = Class.forName(AStarPlayer.class.getName() + "$Search");
            var constructor = type.getDeclaredConstructor();
            constructor.setAccessible(true);
            Object search = constructor.newInstance();
            Method enqueue = AStarPlayer.class.getDeclaredMethod("enqueue", AStarTreeNode.class, type);
            enqueue.setAccessible(true);
            AStarTreeNode first = new AStarTreeNode(SolitaireFactory.oneMoveFromWin().planningCopy());
            first.setG(1);
            enqueue.invoke(player, first, search);
            AStarTreeNode duplicate = new AStarTreeNode(first.copyState());
            duplicate.setG(2);
            enqueue.invoke(player, duplicate, search);
            assertTrue(duplicate.isPruned());
            AStarTreeNode otherOutcome = new AStarTreeNode(first.copyState());
            otherOutcome.setOutcomeContext(1);
            otherOutcome.setG(2);
            enqueue.invoke(player, otherOutcome, search);
            assertFalse(otherOutcome.isPruned());
        } finally {
            if (previous == null) System.clearProperty("astar.transpositions");
            else System.setProperty("astar.transpositions", previous);
        }
    }

    @Test
    void knownWinningContinuationExcludesChanceAndPrunedBranches() throws Exception {
        AStarTreeNode root = new AStarTreeNode(SolitaireFactory.oneMoveFromWin().planningCopy());
        AStarTreeNode win = new AStarTreeNode(root.copyState());
        win.applyMove("move T1 K♥ F4");
        root.addChild("move T1 K♥ F4", win);
        assertTrue(win.isWon());
        Method find = AStarPlayer.class.getDeclaredMethod("knownWinningChild", AStarTreeNode.class);
        find.setAccessible(true);
        AStarPlayer player = new AStarPlayer();
        assertSame(win, find.invoke(player, root));
        root.setChanceNode(true);
        assertNull(find.invoke(player, root));
        root.setChanceNode(false);
        win.setChanceNode(true);
        assertNull(find.invoke(player, root));
        win.setChanceNode(false);
        win.markPruned(ai.games.player.ai.tree.PruneReason.MANUAL, "test");
        assertNull(find.invoke(player, root));
    }

    @Test
    void committedKnownPlanWinsTwoMoveEndgame() {
        String previous = System.getProperty("astar.finishKnownWin");
        System.setProperty("astar.finishKnownWin", "true");
        try {
            Solitaire game = SolitaireFactory.twoMovesFromWin();
            AStarPlayer player = new AStarPlayer();
            for (int i = 0; i < 2; i++) {
                String command = player.nextCommand(game, "", "");
                assertTrue(command.startsWith("move "), command);
                String[] parts = command.split("\\s+");
                assertTrue(game.moveCard(parts[1], parts.length == 4 ? parts[2] : null,
                        parts[parts.length - 1]));
            }
            assertEquals(52, game.getFoundation().stream().mapToInt(java.util.List::size).sum());
            assertEquals("quit", player.nextCommand(game, "", ""));
        } finally {
            if (previous == null) System.clearProperty("astar.finishKnownWin");
            else System.setProperty("astar.finishKnownWin", previous);
        }
    }
}

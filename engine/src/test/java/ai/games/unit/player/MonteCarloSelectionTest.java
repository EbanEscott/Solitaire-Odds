package ai.games.unit.player;

import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireFactory;
import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class MonteCarloSelectionTest {
    @ParameterizedTest
    @ValueSource(doubles = {0.0, 0.5, 1.0})
    void equalRewardsPreferProgressOnTheCandidateBoard(double reward) throws Exception {
        MonteCarloTreeNode detour = new MonteCarloTreeNode(SolitaireFactory.twoMovesFromWin().planningCopy());
        MonteCarloTreeNode progress = new MonteCarloTreeNode(detour.getState().copy());
        progress.applyMove("move T1 K♠ F3");
        detour.updateStats(reward);
        progress.updateStats(reward);

        assertFalse(progress.isWon());
        assertTrue(progress.evaluate() > detour.evaluate());
        assertTrue(prefers(progress, detour));
        assertFalse(prefers(detour, progress));
        assertEquals(reward, progress.getMeanReward());
        assertEquals(1, progress.getVisits());
    }

    @Test
    void higherMeanRemainsMoreImportantThanHeuristicProgress() throws Exception {
        MonteCarloTreeNode higherMean = new MonteCarloTreeNode(SolitaireFactory.twoMovesFromWin());
        MonteCarloTreeNode progress = new MonteCarloTreeNode(higherMean.getState().copy());
        progress.applyMove("move T1 K♠ F3");
        higherMean.updateStats(0.75);
        progress.updateStats(0.5);

        assertTrue(progress.evaluate() > higherMean.evaluate());
        assertTrue(prefers(higherMean, progress));
        assertFalse(prefers(progress, higherMean));
    }

    @Test
    void immediateVictoryTakesPriorityOverTheHeuristicOnEqualMeans() throws Exception {
        MonteCarloTreeNode win = spy(new MonteCarloTreeNode(SolitaireFactory.wonGame()));
        MonteCarloTreeNode unfinished = spy(new MonteCarloTreeNode(SolitaireFactory.oneMoveFromWin()));
        win.updateStats(1.0);
        unfinished.updateStats(1.0);
        // Keep victory preference independent of the heuristic's current weights.
        doReturn(0).when(win).evaluate();
        doReturn(100).when(unfinished).evaluate();

        assertTrue(prefers(win, unfinished));
        assertFalse(prefers(unfinished, win));
        verify(win, never()).evaluate();
        verify(unfinished, never()).evaluate();
    }

    @Test
    void rejectedCandidatesCannotBecomeTheFirstChoiceButZeroRewardsCan() throws Exception {
        MonteCarloTreeNode rejected = new MonteCarloTreeNode();
        rejected.updateStats(-1.0);
        MonteCarloTreeNode eligible = new MonteCarloTreeNode();
        eligible.updateStats(0.0);

        assertFalse(prefers(rejected, null));
        assertTrue(prefers(eligible, null));
        assertFalse(prefers(rejected, eligible));
    }

    @Test
    void equalMeansAndProgressRetainTheExistingChoice() throws Exception {
        MonteCarloTreeNode first = new MonteCarloTreeNode(SolitaireFactory.twoMovesFromWin());
        MonteCarloTreeNode second = new MonteCarloTreeNode(first.getState().copy());
        first.updateStats(0.5);
        second.updateStats(0.5);

        assertFalse(prefers(second, first));
    }

    private static boolean prefers(MonteCarloTreeNode candidate, MonteCarloTreeNode best)
            throws Exception {
        Method method = MonteCarloPlayer.class.getDeclaredMethod("isBetterCandidate",
                MonteCarloTreeNode.class, MonteCarloTreeNode.class);
        method.setAccessible(true);
        return (boolean) method.invoke(new MonteCarloPlayer(), candidate, best);
    }
}

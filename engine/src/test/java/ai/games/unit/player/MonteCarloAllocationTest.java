package ai.games.unit.player;

import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToDoubleFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MonteCarloAllocationTest {
    @Test
    void allocatesMoreSamplesToBetterRewardsWhileStillExploring() throws Exception {
        MonteCarloTreeNode good = new MonteCarloTreeNode();
        MonteCarloTreeNode bad = new MonteCarloTreeNode();
        List<MonteCarloTreeNode> sampled = new ArrayList<>();

        MonteCarloTreeNode parent = allocate(List.of(good, bad), candidate -> {
            sampled.add(candidate);
            return candidate == good ? 1.0 : 0.0;
        });

        assertEquals(List.of(good, bad), sampled.subList(0, 2));
        assertEquals(128, sampled.size());
        assertEquals(128, parent.getVisits());
        assertTrue(good.getVisits() > 64);
        assertTrue(bad.getVisits() > 1 && bad.getVisits() < 64);
        assertEquals(good.getVisits(), parent.getTotalReward());
    }

    @Test
    void anInitialZeroDoesNotPreventLaterRecovery() throws Exception {
        MonteCarloTreeNode recovering = new MonteCarloTreeNode();
        MonteCarloTreeNode steady = new MonteCarloTreeNode();

        allocate(List.of(recovering, steady), candidate -> {
            if (candidate == steady) return 0.25;
            return candidate.getVisits() == 0 ? 0.0 : 1.0;
        });

        assertTrue(recovering.getVisits() > steady.getVisits());
        assertTrue(recovering.getMeanReward() > steady.getMeanReward());
        assertEquals(128, recovering.getVisits() + steady.getVisits());
    }

    @Test
    void rejectedCandidatesReceiveOneCheckAndDoNotInflateTheRolloutBudget() throws Exception {
        MonteCarloTreeNode rejected = new MonteCarloTreeNode();
        MonteCarloTreeNode eligible = new MonteCarloTreeNode();

        MonteCarloTreeNode parent = allocate(List.of(rejected, eligible),
                candidate -> candidate == rejected ? -1.0 : 0.0);

        assertEquals(1, rejected.getVisits());
        assertEquals(-1.0, rejected.getMeanReward());
        assertEquals(64, eligible.getVisits());
        assertEquals(64, parent.getVisits());
        assertEquals(0.0, parent.getTotalReward());
    }

    @Test
    void equalZeroRewardsKeepAllCandidatesInTheAllocation() throws Exception {
        List<MonteCarloTreeNode> candidates = List.of(new MonteCarloTreeNode(),
                new MonteCarloTreeNode(), new MonteCarloTreeNode());

        MonteCarloTreeNode parent = allocate(candidates, candidate -> 0.0);

        assertEquals(192, parent.getVisits());
        for (MonteCarloTreeNode candidate : candidates) {
            assertEquals(64, candidate.getVisits());
        }
    }

    @Test
    void noEligibleCandidatesEndsAllocationWithoutFurtherSamples() throws Exception {
        MonteCarloTreeNode rejected = new MonteCarloTreeNode();
        assertEquals(0, allocate(List.of(rejected), candidate -> -1.0).getVisits());
        assertEquals(1, rejected.getVisits());
        assertEquals(0, allocate(List.of(), candidate -> {
            throw new AssertionError("An empty decision must not sample");
        }).getVisits());
    }

    private static MonteCarloTreeNode allocate(List<MonteCarloTreeNode> candidates,
                                              ToDoubleFunction<MonteCarloTreeNode> rollout)
            throws Exception {
        MonteCarloTreeNode parent = new MonteCarloTreeNode();
        Method method = MonteCarloPlayer.class.getDeclaredMethod("evaluateCandidates",
                MonteCarloTreeNode.class, List.class, ToDoubleFunction.class);
        method.setAccessible(true);
        method.invoke(new MonteCarloPlayer(), parent, candidates, rollout);
        return parent;
    }
}

package ai.games.player.ai.astar;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.tree.TreeNode;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Bounded A* lookahead over observed cards, with deferred chance outcomes for reveals.
 * <p>
 * Known positions are ordered by ordinary {@code g + h}. Decision values take the best
 * continuation; reveal values average their outcomes. The chosen real path is retained for
 * pruning, while speculative descendants are discarded after each actual observation.
 */
@Component
@Profile("ai-astar")
public class AStarPlayer extends AIPlayer {

    /** Maximum expansions per decision, shared by known positions and reveals. */
    private static final int NODE_BUDGET = 1024;

    /** Initial allowance for exploring known cards before returning to reveals. */
    private static final int KNOWN_FIRST_BUDGET = NODE_BUDGET / 2;

    /** Known-position expansions between deferred reveals after the initial phase. */
    private static final int KNOWN_BATCH = 16;

    /** Maximum retained nodes per decision, including complete outcome groups. */
    private static final int MAX_SEARCH_NODES = 5000;

    /** Enumerate small outcome sets exactly; otherwise draw this many joint samples. */
    private static final int REVEAL_SAMPLES = 8;

    /** Finite failure cost so one losing outcome does not veto an entire reveal. */
    private static final double DEAD_END_COST = 1000.0;

    /** Root of the actual game path. */
    private AStarTreeNode root;

    /** Current observed position, with prior actual positions retained as ancestors. */
    private AStarTreeNode current;

    /**
     * Discards the retained game path before starting another game.
     */
    public void reset() {
        root = null;
        current = null;
    }

    /**
     * Searches known continuations first, then revisits deferred reveals within the budget.
     * <p>
     * Each root action is compared using its backed-up expected cost. Unknown outcomes are
     * averaged rather than selecting a fortunate guess. Fresh observations replace all prior
     * assumptions before searching again.
     *
     * @param solitaire the real observed board, which is never mutated by the search
     * @param moves unused; legal moves are generated from each planning position
     * @param feedback unused; the selected commands come from legal move generation
     * @return the selected game command, or {@code quit} when won or exhausted
     */
    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        Solitaire planning = solitaire.planningCopy();
        if (root == null) {
            root = new AStarTreeNode(planning);
            current = root;
        }
        current.setState(planning);
        current.setG(0.0);
        current.setH(AStarTreeNode.computeHeuristic(planning));
        current.setProbability(1.0);
        current.setExpanded(false);
        current.setChanceNode(false);
        current.getChildren().clear();
        current.recalculateF();
        if (current.isWon()) return "quit";

        Search search = new Search();
        enqueue(current, search);
        int expansions = 0;
        while (!search.known.isEmpty() && expansions < KNOWN_FIRST_BUDGET) {
            expandKnown(search.known.poll(), search);
            expansions++;
        }
        search.nodeLimit = MAX_SEARCH_NODES;
        int knownSinceReveal = KNOWN_BATCH;
        while (expansions < NODE_BUDGET && (!search.known.isEmpty() || !search.reveals.isEmpty())) {
            if (!search.reveals.isEmpty()
                    && (search.known.isEmpty() || knownSinceReveal >= KNOWN_BATCH)) {
                expandReveal(search.reveals.poll(), search);
                knownSinceReveal = 0;
            } else {
                expandKnown(search.known.poll(), search);
                knownSinceReveal++;
            }
            expansions++;
        }

        AStarTreeNode best = null;
        double bestValue = Double.POSITIVE_INFINITY;
        for (TreeNode entry : current.getChildren().values()) {
            AStarTreeNode child = (AStarTreeNode) entry;
            if (child.isPruned()) continue;
            double value = evaluate(child);
            if (value < bestValue || (value == bestValue && best != null && child.getH() < best.getH())) {
                best = child;
                bestValue = value;
            }
        }
        if (best == null || bestValue >= DEAD_END_COST) return "quit";
        String command = best.getMove().toCommandString();
        current.getChildren().clear();
        current.addChild(command, best);
        current = best;
        return command;
    }

    /**
     * Expands legal actions from a position whose revealed cards are known.
     * <p>
     * Shared move pruning applies within each path. Transpositions are not discarded across
     * chance outcomes: each outcome must retain its own contribution to the expected value.
     * If the retained-node limit prevents a complete expansion, the node keeps its heuristic
     * estimate instead of treating omitted actions as impossible.
     *
     * @param node the known position to expand
     * @param search queues and retained-node count for this decision
     */
    private void expandKnown(AStarTreeNode node, Search search) {
        if (node.isPruned() || node.isWon()) return;
        List<Move> moves = LegalMovesHelper.listMoves(node.getState()).stream()
                .filter(move -> !move.isQuit()).toList();
        if (search.nodes + moves.size() > search.nodeLimit) return;
        node.setExpanded(true);
        for (Move move : moves) {
            AStarTreeNode child = new AStarTreeNode(node.copyState());
            child.applyMove(move);
            node.addChild(move.toCommandString(), child);
            child.setG(node.getG() + 1.0);
            child.setH(AStarTreeNode.computeHeuristic(child.getState()));
            child.setProbability(node.getProbability());
            child.recalculateF();
            search.nodes++;
            if (!child.doPruning()) enqueue(child, search);
        }
    }

    /**
     * Defers unknown positions while making known positions available for ordinary lookahead.
     *
     * @param node the new search position
     * @param search queues for this decision
     */
    private void enqueue(AStarTreeNode node, Search search) {
        if (node.getState().nextUnresolvedCard() == null) search.known.add(node);
        else search.reveals.add(node);
    }

    /**
     * Resolves all currently exposed slots as one chance event before exploring further moves.
     * <p>
     * Small joint outcome sets are enumerated exactly. Larger sets use uniform joint samples:
     * identities are drawn without replacement within each sample, while independent samples
     * may repeat. Their equal weights estimate the complete probability distribution, rather
     * than retaining the first few identities in enumeration order. True path probabilities
     * remain on the children separately from these averaging weights.
     *
     * @param node the deferred position containing revealed UNKNOWN slots
     * @param search queues and retained-node count for this decision
     */
    private void expandReveal(AStarTreeNode node, Search search) {
        Solitaire state = node.getState();
        int slots = 0;
        for (List<Card> pile : state.getVisibleTableau()) {
            for (Card card : pile) if (card.getRank() == Rank.UNKNOWN) slots++;
        }
        for (Card card : state.getTalon()) if (card.getRank() == Rank.UNKNOWN) slots++;
        int remaining = state.getUnknownCards().size();
        int outcomes = 1;
        for (int i = 0; i < slots && outcomes <= REVEAL_SAMPLES; i++) outcomes *= remaining - i;
        int count = Math.min(outcomes, REVEAL_SAMPLES);
        if (count < 1) throw new IllegalStateException("No identities available for a revealed slot");
        if (search.nodes + count > MAX_SEARCH_NODES) return;

        List<AStarTreeNode> children = new ArrayList<>();
        if (outcomes <= REVEAL_SAMPLES) {
            enumerateOutcomes(node, state, 1.0, children);
        } else {
            // Reproducible samples for the same planning position; never use hidden identities.
            Random random = new Random(node.getStateKey());
            for (int i = 0; i < REVEAL_SAMPLES; i++) {
                Solitaire branch = state.copy();
                double probability = 1.0;
                Card unknown;
                while ((unknown = branch.nextUnresolvedCard()) != null) {
                    List<Card> unseen = branch.getUnknownCards();
                    probability *= branch.assumeCard(unknown, unseen.get(random.nextInt(unseen.size())));
                }
                children.add(outcome(node, branch, probability, 1.0 / REVEAL_SAMPLES));
            }
        }
        node.setExpanded(true);
        node.setChanceNode(true);
        for (int i = 0; i < children.size(); i++) {
            AStarTreeNode child = children.get(i);
            node.addChild("outcome " + i, child);
            search.nodes++;
            enqueue(child, search);
        }
    }

    /**
     * Enumerates a small reveal event, retaining its full conditional probability mass.
     *
     * @param parent the reveal position
     * @param state the partially assigned independent board
     * @param probability probability of assignments since the parent
     * @param outcomes destination for fully assigned outcome nodes
     */
    private void enumerateOutcomes(AStarTreeNode parent, Solitaire state, double probability,
                                   List<AStarTreeNode> outcomes) {
        Card unknown = state.nextUnresolvedCard();
        if (unknown == null) {
            outcomes.add(outcome(parent, state, probability, probability));
            return;
        }
        for (Card identity : state.getUnknownCards()) {
            Solitaire branch = state.copy();
            double conditional = branch.assumeCard(unknown, identity);
            enumerateOutcomes(parent, branch, probability * conditional, outcomes);
        }
    }

    /**
     * Creates an outcome with separate path probability and expectation weight.
     *
     * @param parent the reveal position
     * @param state the fully assigned board
     * @param probability conditional probability of this exact assignment sequence
     * @param weight exact probability or empirical sampling weight used for averaging
     * @return a node with no game command and no additional move cost
     */
    private AStarTreeNode outcome(AStarTreeNode parent, Solitaire state, double probability, double weight) {
        AStarTreeNode child = new AStarTreeNode(state);
        child.setG(parent.getG());
        child.setProbability(parent.getProbability() * probability);
        child.setOutcomeWeight(weight);
        child.recalculateF();
        return child;
    }

    /**
     * Backs up costs by choosing actions and averaging chance outcomes.
     * <p>
     * Unexpanded leaves retain their full {@code g + h} estimate. Exhausted decision nodes
     * receive a finite failure cost, so a losing outcome influences rather than automatically
     * vetoes its parent action. These are heuristic costs, not calibrated win probabilities.
     *
     * @param node the subtree to evaluate
     * @return estimated total cost from the current observation; lower is better
     */
    private double evaluate(AStarTreeNode node) {
        if (node.isWon()) return node.getG();
        if (!node.isExpanded()) return node.getF();
        if (node.isChanceNode()) {
            double expected = 0.0;
            for (TreeNode entry : node.getChildren().values()) {
                AStarTreeNode child = (AStarTreeNode) entry;
                expected += child.getOutcomeWeight() * evaluate(child);
            }
            return expected;
        }
        double best = Double.POSITIVE_INFINITY;
        for (TreeNode entry : node.getChildren().values()) {
            AStarTreeNode child = (AStarTreeNode) entry;
            if (!child.isPruned()) best = Math.min(best, evaluate(child));
        }
        return Double.isInfinite(best) ? DEAD_END_COST + node.getG() : best;
    }

    /**
     * Queues and allocation bounds belonging to one decision, never shared across observations.
     */
    private static final class Search {
        private static final Comparator<AStarTreeNode> ORDER = Comparator
                .comparingDouble(AStarTreeNode::getF)
                .thenComparingDouble(AStarTreeNode::getH)
                .thenComparingLong(AStarTreeNode::getStateKey);
        private final PriorityQueue<AStarTreeNode> known = new PriorityQueue<>(ORDER);
        private final PriorityQueue<AStarTreeNode> reveals = new PriorityQueue<>(ORDER);
        private int nodes = 1;
        private int nodeLimit = MAX_SEARCH_NODES / 2;
    }
}

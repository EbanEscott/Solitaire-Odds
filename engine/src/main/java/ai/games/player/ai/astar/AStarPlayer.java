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
import java.util.HashMap;
import java.util.Map;
import ai.games.player.ai.tree.PruneReason;
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
    private final int nodeBudget = Integer.getInteger("astar.nodeBudget", 1024);

    /** Initial allowance for exploring known cards before returning to reveals. */
    private final int knownFirstBudget = Integer.getInteger("astar.knownFirstBudget", nodeBudget / 2);

    /** Experimental heuristic weights; defaults preserve the established policy. */
    private final double foundationWeight = Double.parseDouble(System.getProperty("astar.foundationWeight", "1.0"));
    private final double stockWeight = Double.parseDouble(System.getProperty("astar.stockWeight", "0.5"));

    private final double talonWeight = Double.parseDouble(System.getProperty("astar.talonWeight", "0.0"));
    private final double hiddenWeight = Double.parseDouble(System.getProperty("astar.hiddenWeight", "2.0"));
    private final double emptyWeight = Double.parseDouble(System.getProperty("astar.emptyWeight", "3.0"));

    private final double moveCost = Double.parseDouble(System.getProperty("astar.moveCost", "1.0"));

    private final boolean scanStock = Boolean.getBoolean("astar.scanStock");
    private boolean initialScanComplete;

    private final boolean balancedSearch = Boolean.getBoolean("astar.balancedSearch");

    private final boolean reuseDuplicates = Boolean.getBoolean("astar.reuseDuplicates");

    private final boolean transpositions = Boolean.getBoolean("astar.transpositions");

    private final boolean finishKnownWin = Boolean.getBoolean("astar.finishKnownWin");

    /** Known-position expansions between deferred reveals after the initial phase. */
    private static final int KNOWN_BATCH = 16;

    /** Maximum retained nodes per decision, including complete outcome groups. */
    private final int maxSearchNodes = Integer.getInteger("astar.maxSearchNodes", 5000);

    /** Enumerate small outcome sets exactly; otherwise draw this many joint samples. */
    private static final int REVEAL_SAMPLES = 8;

    /** Finite failure cost so one losing outcome does not veto an entire reveal. */
    private final double failureCost = Double.parseDouble(System.getProperty("astar.failureCost", "1000.0"));

    /** Optional development trace, populated only with -Dastar.trace=true. */
    private String decisionTrace = "";

    public String getDecisionTrace() { return decisionTrace; }

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
        initialScanComplete = false;
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
        // Gather stock knowledge through real legal actions, never by reading hidden identities.
        if (scanStock && !initialScanComplete) {
            if (!solitaire.getStockpile().isEmpty()) {
                if (Boolean.getBoolean("astar.trace")) decisionTrace = "Initial stock observation pass";
                return "turn";
            }
            initialScanComplete = true;
        }
        Solitaire planning = solitaire.planningCopy();
        if (root == null) {
            root = new AStarTreeNode(planning);
            current = root;
        }
        // A known winning continuation contains no unresolved or sampled chance event.
        // Reuse it only while the observed board matches the position we planned from.
        if (finishKnownWin && current.getStateKey() == planning.getStateKey()) {
            AStarTreeNode winning = knownWinningChild(current);
            if (winning != null) {
                if (Boolean.getBoolean("astar.trace")) decisionTrace = "Reusing known winning continuation";
                return advance(winning);
            }
        }
        current.setState(planning);
        current.setEquivalent(null);
        current.setG(0.0);
        current.setOutcomeContext(0);
        current.setH(heuristic(planning));
        current.setProbability(1.0);
        current.setExpanded(false);
        current.setChanceNode(false);
        current.getChildren().clear();
        current.recalculateF();
        if (current.isWon()) return "quit";

        Search search = new Search();
        enqueue(current, search);
        int expansions = 0;
        while (!search.known.isEmpty() && expansions < knownFirstBudget) {
            expandKnown(poll(search.known, search), search);
            expansions++;
        }
        search.nodeLimit = maxSearchNodes;
        int knownSinceReveal = KNOWN_BATCH;
        while (expansions < nodeBudget && (!search.known.isEmpty() || !search.reveals.isEmpty())) {
            if (!search.reveals.isEmpty()
                    && (search.known.isEmpty() || knownSinceReveal >= KNOWN_BATCH)) {
                expandReveal(poll(search.reveals, search), search);
                knownSinceReveal = 0;
            } else {
                expandKnown(poll(search.known, search), search);
                knownSinceReveal++;
            }
            expansions++;
        }

        if (Boolean.getBoolean("astar.trace")) {
            StringBuilder trace = new StringBuilder("expansions=" + expansions + " nodes=" + search.nodes);
            for (TreeNode entry : current.getChildren().values()) {
                AStarTreeNode child = (AStarTreeNode) entry;
                trace.append("\n").append(child.getMove()).append(" h=").append(child.getH())
                        .append(" allocated=").append(search.actionExpansions.getOrDefault(child, 0))
                        .append(" value=").append(evaluate(child)).append(" prune=").append(child.getPruneReason());
            }
            decisionTrace = trace.toString();
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
        if (finishKnownWin) {
            AStarTreeNode winning = knownWinningChild(current);
            if (winning != null) return advance(winning);
        }
        if (best == null || bestValue >= failureCost) return "quit";
        return advance(best);
    }

    /** Round-robin expansion attempts across root actions, then chance contexts. */
    private AStarTreeNode poll(PriorityQueue<AStarTreeNode> queue, Search search) {
        if (!balancedSearch) return queue.poll();
        AStarTreeNode best = null;
        for (AStarTreeNode node : queue) {
            if (best == null || compareAllocation(node, best, search) < 0) best = node;
        }
        queue.remove(best);
        AStarTreeNode action = rootAction(best);
        search.actionExpansions.merge(action, 1, Integer::sum);
        search.outcomeExpansions.merge(best.getOutcomeContext(), 1, Integer::sum);
        return best;
    }

    private int compareAllocation(AStarTreeNode a, AStarTreeNode b, Search search) {
        int order = Integer.compare(search.actionExpansions.getOrDefault(rootAction(a), 0),
                search.actionExpansions.getOrDefault(rootAction(b), 0));
        if (order == 0) order = Integer.compare(search.outcomeExpansions.getOrDefault(a.getOutcomeContext(), 0),
                search.outcomeExpansions.getOrDefault(b.getOutcomeContext(), 0));
        return order == 0 ? Search.ORDER.compare(a, b) : order;
    }

    private AStarTreeNode rootAction(AStarTreeNode node) {
        while (node != current && node.getParent() != null && node.getParent() != current)
            node = (AStarTreeNode) node.getParent();
        return node;
    }

    /** Commits one legal search action while retaining its continuation. */
    private String advance(AStarTreeNode best) {
        String command = best.getMove().toCommandString();
        current.getChildren().clear();
        current.addChild(command, best);
        current = best;
        return command;
    }

    /** Finds a winning continuation using only already known cards, never sampled outcomes. */
    private AStarTreeNode knownWinningChild(AStarTreeNode node) {
        if (node.isPruned() || node.isChanceNode() || node.getState().nextUnresolvedCard() != null) return null;
        for (TreeNode entry : node.getChildren().values()) {
            AStarTreeNode child = (AStarTreeNode) entry;
            if (!child.isPruned() && !child.isChanceNode()
                    && child.getState().nextUnresolvedCard() == null
                    && (child.isWon() || knownWinningChild(child) != null)) return child;
        }
        return null;
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
            child.setOutcomeContext(node.getOutcomeContext());
            child.setG(node.getG() + moveCost);
            child.setH(heuristic(child.getState()));
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
        if (transpositions) {
            StateInOutcome key = new StateInOutcome(node.getOutcomeContext(), node.getStateKey());
            AStarTreeNode previous = search.bestG.get(key);
            if (previous != null && previous.getG() <= node.getG()) {
                if (reuseDuplicates) node.setEquivalent(previous);
                else node.markPruned(PruneReason.MANUAL, "Duplicate within the same chance outcome");
                return;
            }
            search.bestG.put(key, node);
        }
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
        int slots = state.unresolvedObservationCount();
        int remaining = state.getUnknownCards().size();
        int outcomes = 1;
        for (int i = 0; i < slots && outcomes <= REVEAL_SAMPLES; i++) outcomes *= remaining - i;
        int count = Math.min(outcomes, REVEAL_SAMPLES);
        if (count < 1) throw new IllegalStateException("No identities available for a revealed slot");
        if (search.nodes + count > maxSearchNodes) return;

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
            child.setOutcomeContext(++search.nextOutcomeContext);
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
        child.setH(heuristic(state));
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
        return evaluate(node, java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>()),
                new java.util.IdentityHashMap<>());
    }

    private double evaluate(AStarTreeNode node, java.util.Set<AStarTreeNode> active,
                            Map<AStarTreeNode, Double> values) {
        // Bound graph backup work to one completed evaluation per node per root action.
        // Cycle cutoffs and path-dependent pruning make this an approximate graph policy.
        if (active.contains(node)) return failureCost + node.getG();
        Double cached = values.get(node);
        if (cached != null) return cached;
        active.add(node);
        double value;
        if (node.getEquivalent() != null) {
            AStarTreeNode representative = node.getEquivalent();
            value = evaluate(representative, active, values) + node.getG() - representative.getG();
        } else if (node.isWon()) {
            value = node.getG();
        } else if (!node.isExpanded()) {
            value = node.getF();
        } else if (node.isChanceNode()) {
            value = 0.0;
            for (TreeNode entry : node.getChildren().values()) {
                AStarTreeNode child = (AStarTreeNode) entry;
                value += child.getOutcomeWeight() * evaluate(child, active, values);
            }
        } else {
            value = Double.POSITIVE_INFINITY;
            for (TreeNode entry : node.getChildren().values()) {
                AStarTreeNode child = (AStarTreeNode) entry;
                if (!child.isPruned()) value = Math.min(value, evaluate(child, active, values));
            }
            if (Double.isInfinite(value)) value = failureCost + node.getG();
        }
        active.remove(node);
        values.put(node, value);
        return value;
    }

    /** Scores experimental variants consistently at observations, actions and chance outcomes. */
    private double heuristic(Solitaire state) {
        return AStarTreeNode.computeHeuristic(state, foundationWeight, stockWeight, hiddenWeight, emptyWeight, talonWeight);
    }

    /**
     * Queues and allocation bounds belonging to one decision, never shared across observations.
     */
    private record StateInOutcome(int context, long stateKey) {}

    private static final class Search {
        private final Map<StateInOutcome, AStarTreeNode> bestG = new HashMap<>();
        private int nextOutcomeContext;
        private final Map<AStarTreeNode, Integer> actionExpansions = new java.util.IdentityHashMap<>();
        private final Map<Integer, Integer> outcomeExpansions = new HashMap<>();
        private static final Comparator<AStarTreeNode> ORDER = Comparator
                .comparingDouble(AStarTreeNode::getF)
                .thenComparingDouble(AStarTreeNode::getH)
                .thenComparingLong(AStarTreeNode::getStateKey);
        private final PriorityQueue<AStarTreeNode> known = new PriorityQueue<>(ORDER);
        private final PriorityQueue<AStarTreeNode> reveals = new PriorityQueue<>(ORDER);
        private int nodes = 1;
        private int nodeLimit = Integer.getInteger("astar.maxSearchNodes", 5000) / 2;
    }
}

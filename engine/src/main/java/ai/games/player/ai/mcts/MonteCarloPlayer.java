package ai.games.player.ai.mcts;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.List;
import java.util.Objects;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Monte Carlo player with UCB allocation among immediate moves for Klondike Solitaire.
 *
 * <h2>Algorithm Overview</h2>
 * <p>This player uses a flat bandit over immediate legal moves. It samples each eligible
 * candidate once, then allocates the remaining rollouts using an upper confidence bound
 * (UCB). The total budget is {@value #ROLLOUTS_PER_CANDIDATE} times the number of eligible
 * candidates. It does not expand a deeper search tree.
 *
 * <h2>How it Works</h2>
 * <ol>
 *   <li><b>Enumerate moves:</b> Create one planning board per legal candidate
 *   <li><b>Initialise:</b> Sample each candidate once and exclude rejected candidates
 *   <li><b>Allocate:</b> Use mean reward plus an exploration bonus to choose further samples
 *   <li><b>Select:</b> Choose the highest mean reward, breaking ties by an immediate win
 *       and then the existing board progress heuristic, without the exploration bonus
 * </ol>
 *
 * <p>By default, each evaluation samples every UNKNOWN identity without replacement and
 * runs bounded depth-first search on that fixed arrangement. This assumes full information
 * within a sample and can overestimate achievable play under uncertainty. Random rollouts
 * remain available as a comparator through {@code montecarlo.evaluator=random}.
 *
 * <h2>Configuration</h2>
 * <p>Key parameters can be tuned for different playing strengths:
 * <ul>
 *   <li>{@code ROLLOUTS_PER_CANDIDATE}: Budget contribution per eligible move (default {@value #ROLLOUTS_PER_CANDIDATE})
 * </ul>
 */
@Component
@Profile("ai-mcts")
public class MonteCarloPlayer extends AIPlayer {

    private static final Logger log = LoggerFactory.getLogger(MonteCarloPlayer.class);

    /**
     * Rollout budget contribution per eligible candidate, shared using UCB.
     * Individual candidates may receive more or fewer than this many samples.
     */
    private static final int ROLLOUTS_PER_CANDIDATE = 64;

    /**
     * Exploration weight for rollout rewards in [0, 1].
     */
    private static final double EXPLORATION_CONSTANT = Math.sqrt(2.0);

    /**
     * Maximum simulation steps per playout to avoid infinite loops.
     */
    private static final int MAX_SIMULATION_STEPS = 128;

    /**
     * Default visit budget per sampled depth-first search, including its starting position.
     */
    private static final int DEFAULT_SEARCH_VISITS = 128;

    /** Whether to use the original random rollout evaluator for comparison. */
    private final boolean useRandomRollouts;
    /** Maximum visited positions per sampled depth-first evaluation. */
    private final int maxSearchVisits;
    /** Experimental reward for an unresolved DFS budget cutoff; ordinary play uses 0.5. */
    private final double searchCutoffReward;
    /** Generator used to sample hidden arrangements and random comparator moves. */
    private final RandomGenerator random;
    /** Experimental DFS ordering; ordinary play retains generation order by default. */
    private final boolean revealFirst;
    /** Experimental per-sample cache, including incoming-move pruning context. */
    private final boolean useTranspositions;
    /** Experimental reuse of winning DFS continuations only when every card was already known. */
    private final boolean reuseKnownSolution;
    /** Experimental conservative foundation dominance inside sampled DFS only. */
    private final boolean safeFoundations;
    /** Winning continuations found during this decision, before any actual move is selected. */
    private final Map<MonteCarloTreeNode, List<Move>> knownSolutions = new HashMap<>();
    /** Remaining moves of the selected fully observed solution. */
    private final ArrayDeque<Move> knownContinuation = new ArrayDeque<>();

    /**
     * Creates a player using sampled depth-first search, or the random rollout comparator.
     * <p>
     * Set {@code montecarlo.evaluator=random} for the previous policy. The default is
     * {@code dfs}; {@code montecarlo.search.visits} controls its per-sample visit budget.
     */
    public MonteCarloPlayer() {
        this(ThreadLocalRandom.current());
    }

    /**
     * Creates a player with explicit randomness for reproducible sampled searches.
     * <p>
     * Experimental controls {@code montecarlo.search.revealFirst} and
     * {@code montecarlo.search.transpositions} default to false. They change only DFS;
     * the random rollout comparator retains its original move policy.
     * {@code montecarlo.search.reuseKnownSolution} optionally retains a winning continuation
     * only when all identities were known before sampling; it also defaults to false.
     * {@code montecarlo.search.cutoffReward} controls the score of unresolved DFS searches,
     * defaults to 0.5, and leaves solved and exhausted outcomes unchanged.
     *
     * @param random the generator used for sampling, independent of the real hidden deal
     */
    public MonteCarloPlayer(RandomGenerator random) {
        this.random = Objects.requireNonNull(random, "random");
        revealFirst = Boolean.getBoolean("montecarlo.search.revealFirst");
        useTranspositions = Boolean.getBoolean("montecarlo.search.transpositions");
        reuseKnownSolution = Boolean.getBoolean("montecarlo.search.reuseKnownSolution");
        safeFoundations = Boolean.getBoolean("montecarlo.search.safeFoundations");
        useRandomRollouts = switch (System.getProperty("montecarlo.evaluator", "dfs")) {
            case "dfs" -> false;
            case "random" -> true;
            default -> throw new IllegalArgumentException("montecarlo.evaluator must be dfs or random");
        };
        maxSearchVisits = Integer.getInteger("montecarlo.search.visits", DEFAULT_SEARCH_VISITS);
        if (maxSearchVisits < 1) throw new IllegalArgumentException("Search visit budget must be positive");
        searchCutoffReward = Double.parseDouble(System.getProperty("montecarlo.search.cutoffReward", "0.5"));
        if (!Double.isFinite(searchCutoffReward) || searchCutoffReward < 0 || searchCutoffReward > 1) {
            throw new IllegalArgumentException("DFS cutoff reward must be finite and between zero and one");
        }
    }

    /**
     * Persisted MCTS tree root across turns.
     */
    private MonteCarloTreeNode root;

    /**
     * Node corresponding to the current Solitaire state.
     */
    private MonteCarloTreeNode current;

    /**
     * Expected planning position after the previously selected move is applied.
     * Used to detect state drift when callers forget to apply the AI's chosen move.
     */
    private Solitaire expectedNextState;

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        Solitaire planning = solitaire.planningCopy();
        // Predicted UNKNOWN slots may now have real identities, but known cards and pile shapes must match.
        if (expectedNextState != null && !expectedNextState.matchesObservation(planning)) {
            throw new IllegalStateException("State drift detected: the selected move was not applied");
        }
        expectedNextState = null;

        if (log.isTraceEnabled()) {
            log.trace(LegalMovesHelper.listLegalMoves(solitaire).toString());
        }

        // Initialise or update MCTS tree
        if (root == null || current == null) {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(planning);
            node.setParent(null);
            this.root = node;
            this.current = node;
        } else {
            current.setState(planning);
            current.getChildren().clear();
            current.resetStats();
        }

        if(log.isTraceEnabled()) {
            log.trace("MCTS starting. Root has {} children that include {}", current.getChildren().size(), current.getChildren().keySet());
        }

        knownSolutions.clear();
        if (reuseKnownSolution && !knownContinuation.isEmpty()) {
            Move next = knownContinuation.peek();
            if (planning.getUnknownCards().isEmpty() && LegalMovesHelper.listMoves(planning).contains(next)) {
                MonteCarloTreeNode child = new MonteCarloTreeNode(planning.copy());
                current.addChild(next.toCommandString(), child);
                child.applyMove(next);
                knownContinuation.remove();
                current = child;
                expectedNextState = child.getState();
                return next.toCommandString();
            }
            knownContinuation.clear();
        }

        // Expand immediate candidates; each board is reused throughout allocation.
        long startTime = System.currentTimeMillis();
        List<Move> legalMoves = LegalMovesHelper.listMoves(current.getState());
        legalMoves.removeIf(Move::isQuit);
        List<MonteCarloTreeNode> candidates = new ArrayList<>(legalMoves.size());
        for (Move move : legalMoves) {
            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(current);
            current.addChild(move.toCommandString(), child);
            child.setState(planning.copy());
            child.applyMove(move);

            candidates.add(child);
        }
        evaluateCandidates(current, candidates, this::simulate);
        long elapsedMs = System.currentTimeMillis() - startTime;
        if(log.isDebugEnabled()) {
            log.debug("Monte Carlo evaluated {} candidates with {} rollouts:", candidates.size(), current.getVisits());
        }

        // Select best move
        String bestMove = null;
        MonteCarloTreeNode bestChild = null;
        for (String move : current.getChildren().keySet()) {
            MonteCarloTreeNode child = (MonteCarloTreeNode) current.getChildren().get(move);
            int visits = child.getVisits();
            double meanReward = child.getMeanReward();
            
            if(log.isDebugEnabled()) {
                log.debug("  '{}': visits={}, meanReward={}", move, visits, String.format("%.4f", meanReward));
            }
            
            if (isBetterCandidate(child, bestChild)) {
                bestChild = child;
                bestMove = move;
            }
        }

        // If MCTS didn't expand anything (or we're in a terminal/no-moves situation), fall back safely.
        if (bestMove == null) {
            if(log.isDebugEnabled()) {
                log.debug("MCTS found no explored moves; falling back to quit.");
            }
            return "quit";
        }

        int completedRollouts = current.getVisits();
        // Advance current to selected child for next decision (root stays at the initial game state)
        current = bestChild;
        if (reuseKnownSolution) knownContinuation.addAll(knownSolutions.getOrDefault(bestChild, List.of()));

        // Record the state we expect to see on the next call (after the move is applied).
        expectedNextState = current.getState();

        if(log.isDebugEnabled()) {
            log.debug("MCTS selected '{}' with {} visits (highest mean reward)", bestMove, bestChild.getVisits());
        }

        if(log.isTraceEnabled()) {
            log.trace(
                "Monte Carlo decision: {} rollouts in {}ms, selected move: {}",
                completedRollouts,
                elapsedMs,
                bestMove
            );
        }

        return bestMove;
    }

    /**
     * Compares final candidates by mean reward, then immediate victory and board progress.
     * <p>
     * Both unresolved searches and solved sampled worlds can leave moves tied. Prefer
     * progress on the restored planning board in that case, so equivalent rewards do not
     * favour detours. The heuristic only breaks exact ties; it does not alter rewards or UCB.
     *
     * @param candidate the evaluated candidate to consider
     * @param best the best candidate so far, or null before an eligible candidate is found
     * @return whether the candidate should replace the current choice
     */
    private boolean isBetterCandidate(MonteCarloTreeNode candidate, MonteCarloTreeNode best) {
        double mean = candidate.getMeanReward();
        if (best == null) return mean > -1.0;
        if (mean != best.getMeanReward()) return mean > best.getMeanReward();
        if (candidate.isWon() != best.isWon()) return candidate.isWon();
        return candidate.evaluate() > best.evaluate();
    }

    /**
     * Allocates a fixed rollout budget among immediate candidates using UCB.
     * <p>
     * Each candidate is sampled once before adaptive allocation. Rejected candidates retain
     * their rejection statistic but do not enter the bandit or contribute to its budget.
     * Eligible candidates share {@value #ROLLOUTS_PER_CANDIDATE} samples per candidate
     * in total, including those initial samples. Zero rewards remain eligible, since later
     * rollouts can have different outcomes.
     * Statistics on the parent count only eligible rollouts and supply UCB's total visits.
     *
     * @param parent the current decision node with fresh statistics
     * @param candidates the newly expanded candidates in legal-move order
     * @param rollout the evaluation policy, which restores each candidate board after use
     */
    private void evaluateCandidates(MonteCarloTreeNode parent, List<MonteCarloTreeNode> candidates,
                                    ToDoubleFunction<MonteCarloTreeNode> rollout) {
        List<MonteCarloTreeNode> eligible = new ArrayList<>(candidates.size());
        for (MonteCarloTreeNode candidate : candidates) {
            double reward = rollout.applyAsDouble(candidate);
            candidate.updateStats(reward);
            if (reward == -1.0) {
                continue;
            }
            eligible.add(candidate);
            parent.updateStats(reward);
        }

        int budget = ROLLOUTS_PER_CANDIDATE * eligible.size();
        while (parent.getVisits() < budget) {
            MonteCarloTreeNode selected = null;
            double bestValue = Double.NEGATIVE_INFINITY;
            for (MonteCarloTreeNode candidate : eligible) {
                double value = candidate.getUctValue(EXPLORATION_CONSTANT, parent.getVisits());
                if (value > bestValue) {
                    bestValue = value;
                    selected = candidate;
                }
            }
            double reward = rollout.applyAsDouble(selected);
            selected.updateStats(reward);
            parent.updateStats(reward);
        }
    }

    /**
     * Evaluates a candidate using the configured simulation policy.
     *
     * @param node the candidate whose board is restored after evaluation
     * @return the sampled reward, or -1 for a rejected starting candidate
     */
    private double simulate(MonteCarloTreeNode node) {
        return useRandomRollouts
                ? simulateRandom(node, random, MAX_SIMULATION_STEPS)
                : simulateDepthFirst(node, random, maxSearchVisits);
    }

    /**
     * Searches one sampled full-information arrangement using bounded depth-first traversal.
     * <p>
     * All unknown identities are sampled once, before traversal, and remain fixed across
     * sibling branches. Frames hold moves and history checkpoints rather than board copies.
     * Cycle checks are local to the active path. Optional transposition checks additionally
     * reject repeated board/incoming-move pairs within this sample.
     * The starting position and every attempted successor (including cycles) consume a visit.
     * Search policy is optimistic: it may find different continuations for different sampled
     * arrangements before those arrangements could be distinguished in actual play.
     *
     * @param node the candidate, already containing its first move
     * @param random the generator used only to sample the hidden arrangement
     * @param maxVisits the positive visit budget for this evaluation
     * @return 1 for a win, 0 for exhausted permitted search, the configured reward (default 0.5) for a budget cutoff,
     *         or -1 for a rejected starting candidate; exhaustion is relative to pruning
     */
    private double simulateDepthFirst(MonteCarloTreeNode node, RandomGenerator random, int maxVisits) {
        if (maxVisits < 1) throw new IllegalArgumentException("Search visit budget must be positive");
        if (node.isQuit() || node.isUselessKingMove() || node.isCycleDetected()) return -1.0;
        Solitaire board = node.getState();
        int checkpoint = board.getMoveHistorySize();
        // Check before sampling: a guessed complete world must never become a retained real-game plan.
        boolean rememberSolution = reuseKnownSolution && board.getUnknownCards().isEmpty();
        try {
            if (board.isInPlanMode()) board.sampleUnknownCards(random);
            MonteCarloTreeNode position = new MonteCarloTreeNode(board);
            position.setMove(node.getMove());
            if (position.isWon()) return 1.0;

            ArrayDeque<SearchFrame> frames = new ArrayDeque<>();
            Set<Long> path = new HashSet<>();
            Set<SearchPosition> visited = useTranspositions ? new HashSet<>() : Set.of();
            frames.push(searchFrame(position));
            path.add(position.getStateKey());
            if (useTranspositions) visited.add(new SearchPosition(position.getStateKey(), position.getMove()));
            int visits = 1;
            while (!frames.isEmpty()) {
                SearchFrame frame = frames.peek();
                board.rollbackTo(frame.historyDepth());
                position.setState(board);
                position.setMove(frame.move());
                if (!frame.moves().hasNext()) {
                    frames.pop();
                    path.remove(frame.stateKey());
                    continue;
                }
                if (visits >= maxVisits) return searchCutoffReward;

                position.applyMove(frame.moves().next());
                visits++;
                if (position.isWon()) {
                    if (rememberSolution) rememberKnownSolution(node, frames, position.getMove());
                    return 1.0;
                }
                if (path.contains(position.getStateKey())) continue;
                if (useTranspositions
                        && !visited.add(new SearchPosition(position.getStateKey(), position.getMove()))) continue;
                path.add(position.getStateKey());
                frames.push(searchFrame(position));
            }
            return 0.0;
        } finally {
            board.rollbackTo(checkpoint);
        }
    }

    /**
     * Retains a winning DFS path only for a candidate that had no unknown identities before sampling.
     * <p>
     * The first frame is the already-applied root candidate, so its incoming move is excluded.
     * Later decisions still validate observations and legality before following this path.
     *
     * @param candidate the candidate whose continuation reached a win
     * @param frames the active DFS path, with the starting candidate at the bottom
     * @param winningMove the final move, not yet represented by a frame
     */
    private void rememberKnownSolution(MonteCarloTreeNode candidate, ArrayDeque<SearchFrame> frames,
                                       Move winningMove) {
        List<Move> continuation = new ArrayList<>();
        Iterator<SearchFrame> path = frames.descendingIterator();
        path.next();
        while (path.hasNext()) continuation.add(path.next().move());
        continuation.add(winningMove);
        knownSolutions.put(candidate, List.copyOf(continuation));
    }

    /**
     * Captures a DFS continuation using the configured ordering at the current board position.
     *
     * @param position the temporary node used for move application and pruning
     * @return a frame containing no board snapshot
     */
    private SearchFrame searchFrame(MonteCarloTreeNode position) {
        List<Move> moves = listSimulationMoves(position);
        if (safeFoundations) moves = foundationDominance(position.getState(), moves);
        if (revealFirst) {
            List<List<Card>> visible = position.getState().getVisibleTableau();
            List<Integer> hidden = position.getState().getTableauFaceDownCounts();
            moves.sort(java.util.Comparator.comparingInt(move -> searchPriority(move, visible, hidden)));
        }
        return new SearchFrame(position.getState().getMoveHistorySize(), position.getStateKey(),
                position.getMove(), moves.iterator());
    }

    /**
     * Keeps conservatively safe cards on foundations and prioritises safe tableau placements.
     * <p>
     * Waste placements are never forced: removing a waste card changes future draw-three
     * access. Tableau placements leave stock order intact. This is an opt-in DFS policy,
     * applied only to the current sampled board, not a filter on real root candidates.
     *
     * @param board the sampled position
     * @param moves moves surviving the ordinary DFS pruning filters
     * @return a mutable list with one safe tableau placement, or the remaining moves
     */
    private List<Move> foundationDominance(Solitaire board, List<Move> moves) {
        List<List<Card>> foundations = board.getFoundation();
        List<List<Card>> visible = board.getVisibleTableau();
        int[] heights = new int[Suit.values().length];
        for (List<Card> pile : foundations) {
            if (!pile.isEmpty()) heights[pile.getLast().getSuit().ordinal()] = pile.size();
        }
        List<Move> remaining = new ArrayList<>();
        for (Move move : moves) {
            if (move.isMove() && move.from().type() == Move.PileType.FOUNDATION
                    && isSafeFoundation(foundations.get(move.from().index()).getLast(), heights)) continue;
            if (move.isMove() && move.from().type() == Move.PileType.TABLEAU
                    && move.to().type() == Move.PileType.FOUNDATION
                    && isSafeFoundation(visible.get(move.from().index()).getLast(), heights)) {
                return new ArrayList<>(List.of(move));
            }
            remaining.add(move);
        }
        return remaining;
    }

    /**
     * Checks a conservative foundation safety condition for ordinary alternating-colour tableau rules.
     * <p>
     * Aces and twos need no supporting tableau cards. Higher ranks require both opposite-colour
     * foundations through rank minus one and the other same-colour foundation through rank minus two.
     *
     * @param card the concrete card to place or retain on a foundation
     * @param heights foundation heights indexed by suit ordinal
     * @return whether this card qualifies for the experimental dominance filter
     */
    private boolean isSafeFoundation(Card card, int[] heights) {
        if (card.getRank() == Rank.UNKNOWN) return false;
        int rank = card.getRank().getValue();
        if (rank <= 2) return true;
        for (Suit suit : Suit.values()) {
            if (suit == Suit.UNKNOWN || suit == card.getSuit()) continue;
            int required = suit.isRed() == card.getSuit().isRed() ? rank - 2 : rank - 1;
            if (heights[suit.ordinal()] < required) return false;
        }
        return true;
    }

    /**
     * Orders DFS exploration toward reveals and progress while retaining every permitted move.
     * <p>
     * Turns precede tableau shuffles and foundation withdrawals. Equal priorities retain
     * legal-generation order. Only visible cards and pile visibility are inspected here.
     *
     * @param move a legal move after the existing pruning filters
     * @param visible the visible tableau snapshot for this frame
     * @param hidden the face-down counts for this frame
     * @return a lower value for a move to explore earlier
     */
    private int searchPriority(Move move, List<List<Card>> visible, List<Integer> hidden) {
        if (move.isTurn()) return 3;
        if (move.from().type() == Move.PileType.TABLEAU && hidden.get(move.from().index()) > 0) {
            List<Card> pile = visible.get(move.from().index());
            Card bottom = pile.getFirst();
            if ((move.card() != null && move.card().rank() == bottom.getRank()
                    && move.card().suit() == bottom.getSuit())
                    || (move.card() == null && pile.size() == 1)) return 0;
        }
        if (move.to().type() == Move.PileType.FOUNDATION) return 1;
        if (move.from().type() == Move.PileType.WASTE) return 2;
        if (move.from().type() == Move.PileType.FOUNDATION) return 5;
        return 4;
    }

    /**
     * Generates moves with the existing inverse, useless-king and sibling-symmetry filters.
     *
     * @param position the board and incoming move against which pruning is evaluated
     * @return the remaining moves in generation order, excluding quit
     */
    private List<Move> listSimulationMoves(MonteCarloTreeNode position) {
        List<Move> legalMoves = LegalMovesHelper.listMoves(position.getState());
        List<Move> candidates = new ArrayList<>(legalMoves.size());
        for (Move candidate : legalMoves) {
            if (candidate.isQuit() || position.isInverseOfLastMove(candidate)
                    || position.isUselessKingMove(candidate)
                    || position.isSimilarSibling(candidate, candidates)) continue;
            candidates.add(candidate);
        }
        return candidates;
    }

    // Explicit randomness and step budget also allow deterministic, bounded rollout tests.
    /**
     * Runs a bounded rollout using explicit randomness for move and reveal selection.
     * <p>
     * Samples revealed identities without replacement, then generates and prunes ordinary
     * legal moves. The candidate's board is reused and restored to its starting history depth
     * in a {@code finally} block, including when sampling or move selection throws.
     *
     * @param node the candidate node whose board is temporarily explored
     * @param random the generator used for move selection and speculative reveals
     * @param maxSteps the maximum number of accepted game moves, excluding assignments
     * @return {@code 1.0} for a win, {@code 0.0} for exhausted continuations, the
     *         normalised heuristic for a cutoff, or {@code -1.0} for a pruned candidate
     */
    private double simulateRandom(MonteCarloTreeNode node, RandomGenerator random, int maxSteps) {
        if(log.isTraceEnabled()) {
            log.trace(
                "Simulating move: {} from parent: {}",
                node.getMove(),
                node.getParent() != null && node.getParent().getMove() != null
                        ? node.getParent().getMove() : "ROOT");
        }

        // Skip quitting
        if(node.isQuit()) {
            if(log.isTraceEnabled()) {
                log.trace("Quitting reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        // Skip useless king move
        if(node.isUselessKingMove()) {
            if(log.isTraceEnabled()) {
                log.trace("Useless king move reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        // Skip cycle detected
        if(node.isCycleDetected()) {
            if(log.isTraceEnabled()) {
                log.trace("Cycle detected reward: -1.0");
            }
            return -1.0; // No rewards from undesirable nodes
        }

        Solitaire board = node.getState();
        int rolloutStart = board.getMoveHistorySize();
        // This temporary node operates on the candidate's board. Its path stores keys only;
        // the retained candidate node keeps its original move, key, parent, and statistics.
        MonteCarloTreeNode position = new MonteCarloTreeNode(board);
        position.setParent(node);
        position.setMove(node.getMove());
        List<Long> pathKeys = new ArrayList<>();
        // Preserve the old rollout-root occurrence as well as the retained candidate ancestor.
        pathKeys.add(board.getStateKey());
        try {
            resolveRevealedCards(board, random);
            position.setState(board);
            for (int steps = 0; steps < maxSteps && !position.isWon(); steps++) {
                // Generate and prune against the current board before applying candidates.
                List<Move> candidates = listSimulationMoves(position);

                // Try candidates without replacement, reversing a rejected move before retrying.
                int stepStart = board.getMoveHistorySize();
                Move previousMove = position.getMove();
                Move selected = null;
                while (!candidates.isEmpty()) {
                    int index = random.nextInt(candidates.size());
                    Move candidate = candidates.get(index);
                    Move last = candidates.remove(candidates.size() - 1);
                    if (index < candidates.size()) {
                        candidates.set(index, last);
                    }

                    position.applyMove(candidate);
                    resolveRevealedCards(board, random);
                    position.setState(board);
                    if (position.isCycleDetected(pathKeys)) {
                        board.rollbackTo(stepStart);
                        position.setState(board);
                        position.setMove(previousMove);
                        continue;
                    }
                    selected = candidate;
                    break;
                }

                if (selected == null) {
                    return 0.0;
                }
                pathKeys.add(board.getStateKey());
                if (log.isTraceEnabled()) {
                    log.trace("Rollout move: {}", selected);
                }
            }

            // Includes a win reached on the last permitted step. All other positions
            // here exhausted the step budget, so their outcome is still unresolved.
            if (position.isWon()) {
                return 1.0;
            }
            double normalised = Math.max(0.0, Math.min(Math.nextDown(1.0),
                    position.evaluate() / MonteCarloTreeNode.MAX_SCORE));
            if (log.isTraceEnabled()) {
                log.trace("Normalised reward: {}", String.format("%.4f", normalised));
            }
            return normalised;
        } finally {
            // Also restore after a terminal position, exhausted candidates, or an exception.
            board.rollbackTo(rolloutStart);
        }
    }

    /**
     * Samples identities for all currently revealed UNKNOWN slots on the rollout board.
     * <p>
     * Each identity is chosen uniformly from the remaining unseen pool and committed through
     * {@link Solitaire#assumeCard(Card, Card)}. Later assignments cannot reuse that identity.
     * The enclosing rollout restores these assignments through the board's action history.
     *
     * @param board the board being simulated; GAME boards have no unresolved planning slots
     * @param random the generator used to sample identities without replacement
     * @throws IllegalStateException if an UNKNOWN slot remains but no unseen identity is available
     */
    private void resolveRevealedCards(Solitaire board, RandomGenerator random) {
        Card unknown;
        while ((unknown = board.nextUnresolvedCard()) != null) {
            List<Card> unseen = board.getUnknownCards();
            if (unseen.isEmpty()) throw new IllegalStateException("No unseen identity for a revealed slot");
            board.assumeCard(unknown, unseen.get(random.nextInt(unseen.size())));
        }
    }

    private record SearchFrame(int historyDepth, long stateKey, Move move, Iterator<Move> moves) {}

    private record SearchPosition(long stateKey, Move incomingMove) {}

}

package ai.games.player.ai;

import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Bounded-depth beam search over observed and remembered card identities.
 *
 * <p>Expand every legal non-quit successor of each retained node, including worry-back.
 * Deduplicate each layer by its best path score, then retain at most the configured
 * beam width. Unresolved observations and terminal leaves compete in that same beam
 * without expansion. Choose the first move of the best final surviving sequence,
 * even when its score is below the root. Seeded randomness breaks equal-score ties only.
 *
 * <p>Ancestor checks prevent simulated cycles; real observation history prevents revisits
 * across decisions. Attempted root actions also prevent repeats when an unknown prediction
 * cannot be matched to the actual observed identity. No hidden-card sampling is performed.
 * Construct one player per deal or call {@link #resetGame()} before reuse.
 */
@Component
@Profile("ai-beam")
public class BeamSearchPlayer extends AIPlayer implements Player {
    /** Default maximum number of simulated actions in a path. */
    private static final int DEFAULT_DEPTH_LIMIT = 2;
    /** Default maximum number of retained states per layer, including observation leaves. */
    private static final int DEFAULT_BEAM_WIDTH = 8;
    /** Cost of each simulated action, discouraging unnecessary setup moves. */
    private static final int PER_DEPTH_PENALTY = 2;
    /** Additional cost of a stock turn along the simulated path. */
    private static final int PER_TURN_PENALTY = 4;
    /** Maximum path depth for this player. */
    private final int depthLimit;
    /** Maximum retained frontier size for this player. */
    private final int beamWidth;
    /** Seed retained so resetting a game also resets tie-breaking. */
    private final long seed;
    /** Randomness used only to order equal-scoring retained candidates. */
    private final Random random;
    /** Actual observed positions, never speculative unknown-card assignments. */
    private final Set<Long> visitedStates = new HashSet<>();
    /** Commands already selected at each actual observation. */
    private final Map<Long, Set<String>> attemptedActions = new HashMap<>();

    /** Creates a depth-two, width-eight player with a fresh tie-breaking seed. */
    public BeamSearchPlayer() {
        this(DEFAULT_DEPTH_LIMIT, DEFAULT_BEAM_WIDTH, System.nanoTime());
    }

    /** Creates a seeded beam, clamping depth and width to at least one. */
    public BeamSearchPlayer(int depthLimit, int beamWidth, long seed) {
        /** Preserve the existing constructor's positive-budget normalization. */
        this.depthLimit = Math.max(1, depthLimit);
        this.beamWidth = Math.max(1, beamWidth);
        this.seed = seed;
        this.random = new Random(seed);
    }

    /** Clears per-game observations and restores the initial tie-breaking sequence. */
    public void resetGame() {
        /** Speculative search nodes are local to each decision and need no lifecycle reset. */
        visitedStates.clear();
        attemptedActions.clear();
        random.setSeed(seed);
    }

    /** Searches a bounded beam and returns its best surviving first action, or quits. */
    @Override
    public String nextCommand(Solitaire solitaire, String recommendedMoves, String feedback) {
        /** Mask unknown identities before any scoring or move generation. */
        Solitaire board = solitaire.planningCopy();
        long key = board.getStateKey();
        visitedStates.add(key);
        Set<String> attempted = attemptedActions.computeIfAbsent(key, ignored -> new HashSet<>());
        Node root = new Node(board, null, null, 0, 0, evaluate(board));
        List<Node> frontier = List.of(root);

        /** Each iteration expands one layer and prunes only after every successor is evaluated. */
        for (int depth = 0; depth < depthLimit; depth++) {
            Map<Long, Node> candidates = new LinkedHashMap<>();
            boolean expanded = false;
            /** Expand retained paths independently before merging equal endpoints. */
            for (Node node : frontier) {
                /** Unknown observations are leaves; looking past them would require chance modeling. */
                if (node.state.nextUnresolvedCard() != null || foundationCount(node.state) == 52) {
                    retainBest(candidates, node);
                    continue;
                }
                boolean hasChild = false;
                /** Enumerate the full legal action set without a generation-order branch cutoff. */
                for (Move move : LegalMovesHelper.listMoves(node.state)) {
                    /** Root history also handles predictions that reveal a previously unknown identity. */
                    if (move.isQuit() || (node == root && attempted.contains(move.toCommandString()))) {
                        continue;
                    }
                    Solitaire next = node.state.copy();
                    applyMove(next, move);
                    long nextKey = next.getStateKey();
                    /** Reject ancestors and real visited positions without banning other generated paths. */
                    if (visitedStates.contains(nextKey) || hasAncestor(node, nextKey)) {
                        continue;
                    }
                    int turns = node.turns + (move.isTurn() ? 1 : 0);
                    int pathDepth = node.depth + 1;
                    int score = evaluate(next) - pathDepth * PER_DEPTH_PENALTY - turns * PER_TURN_PENALTY;
                    String firstMove = node == root ? move.toCommandString() : node.firstMove;
                    retainBest(candidates, new Node(next, node, firstMove, pathDepth, turns, score));
                    hasChild = true;
                    expanded = true;
                }
                /** Preserve a genuine dead-end leaf, but never return the root as a candidate action. */
                if (!hasChild && node != root) {
                    retainBest(candidates, node);
                }
            }
            /** No surviving action means every legal continuation is absent or already exhausted. */
            if (candidates.isEmpty()) {
                return "quit";
            }
            frontier = new ArrayList<>(candidates.values());
            Collections.shuffle(frontier, random);
            frontier.sort(Comparator.comparingInt((Node node) -> node.score).reversed());
            /** Prune the scored layer, including terminal and observation leaves, to the beam budget. */
            if (frontier.size() > beamWidth) {
                frontier = new ArrayList<>(frontier.subList(0, beamWidth));
            }
            /** A beam containing only retained leaves cannot benefit from another expansion pass. */
            if (!expanded) {
                break;
            }
        }

        /** Always trust the ranked beam; a negative improvement does not justify random play. */
        String command = frontier.getFirst().firstMove;
        /** An unresolved root has no executable first step until a real observation arrives. */
        if (command == null) {
            return "quit";
        }
        attempted.add(command);
        return command;
    }

    /** Keeps the highest-scoring route to each position within this candidate layer only. */
    private void retainBest(Map<Long, Node> candidates, Node node) {
        /** Replacing an inferior duplicate preserves the cheaper path's first action and ancestors. */
        candidates.merge(node.state.getStateKey(), node,
                (previous, replacement) -> replacement.score > previous.score ? replacement : previous);
    }

    /** Detects cycles in a single path without letting pruned branches poison later layers. */
    private boolean hasAncestor(Node node, long key) {
        /** Search depth bounds this parent walk. */
        for (Node ancestor = node; ancestor != null; ancestor = ancestor.parent) {
            /** Reject a path that returns to any of its own earlier endpoints. */
            if (ancestor.state.getStateKey() == key) {
                return true;
            }
        }
        return false;
    }

    /** Applies one legal structured move to a copied planning board. */
    private void applyMove(Solitaire board, Move move) {
        /** Stock turns use configured draw and recycling rules. */
        if (move.isTurn()) {
            board.turnStock();
        } else {
            /** Include foundation withdrawals and explicit tableau run heads without string heuristics. */
            board.moveCard(move.from().toCode(), move.card() == null ? null : move.card().shortName(),
                    move.to().toCode());
        }
    }

    /** Counts foundation progress without interpreting unknown-card identities. */
    private int foundationCount(Solitaire board) {
        /** Foundation sizes remain observable in both standard and thoughtful modes. */
        return board.getFoundation().stream().mapToInt(List::size).sum();
    }

    /**
     * Scores the original Beam features independently of Hill-climbing's different weights.
     *
     * <p>Rewards are 25 per foundation card, 6 per face-up tableau card, and 10 per empty
     * column; costs are 4 per face-down tableau card and 1 per stock card. Path costs are
     * applied separately. Retaining these weights isolates search-control changes from tuning.
     */
    private int evaluate(Solitaire board) {
        /** Preserve the established foundation and stock contributions. */
        int score = foundationCount(board) * 25 - board.getStockpile().size();
        List<Integer> faceUps = board.getTableauFaceUpCounts();
        List<Integer> faceDowns = board.getTableauFaceDownCounts();
        /** Count physical tableau exposure even when thoughtful mode knows face-down identities. */
        for (int column = 0; column < faceUps.size(); column++) {
            int up = faceUps.get(column);
            int down = faceDowns.get(column);
            score += up * 6 - down * 4;
            /** Only a physically empty column provides a king destination. */
            if (up == 0 && down == 0) {
                score += 10;
            }
        }
        return score;
    }

    /** One retained path endpoint, including enough history for ancestor checks and action selection. */
    private static final class Node {
        /** Planning board at this path endpoint. */
        private final Solitaire state;
        /** Previous endpoint on this path, or null for the root. */
        private final Node parent;
        /** First command from the real root, or null for the root itself. */
        private final String firstMove;
        /** Number of actions taken from the root. */
        private final int depth;
        /** Number of stock turns along the path. */
        private final int turns;
        /** Heuristic endpoint value after cumulative path costs. */
        private final int score;

        /** Captures a scored path endpoint without modifying its planning board. */
        private Node(Solitaire state, Node parent, String firstMove, int depth, int turns, int score) {
            /** Keep path metadata immutable while successor simulations use fresh board copies. */
            this.state = state;
            this.parent = parent;
            this.firstMove = firstMove;
            this.depth = depth;
            this.turns = turns;
            this.score = score;
        }
    }
}

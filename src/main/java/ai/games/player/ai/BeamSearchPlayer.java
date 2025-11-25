package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.HillClimberPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Beam-search player for Klondike Solitaire.
 *
 * <p>At each decision point, this AI:
 * <ul>
 *     <li>Expands legal moves from the current state into a frontier of partial sequences.</li>
 *     <li>Simulates sequences up to DEPTH_LIMIT using {@link Solitaire#copy()}.</li>
 *     <li>Keeps only the top BEAM_WIDTH states by heuristic score at each depth (beam search).</li>
 *     <li>Chooses the first move from the highest-scoring surviving sequence.</li>
 * </ul>
 *
 * <p>Scoring uses the same heuristic as {@link HillClimberPlayer} to stay consistent across AIs.
 */
@Component
@Profile("ai-beam")
public class BeamSearchPlayer extends AIPlayer implements Player {

    private static final int DEFAULT_DEPTH_LIMIT = 2;
    private static final int DEFAULT_BEAM_WIDTH = 8;

    // Soft penalties to discourage endless churning.
    private static final int PER_DEPTH_PENALTY = 2;
    private static final int PER_TURN_PENALTY = 4;

    private final int depthLimit;
    private final int beamWidth;
    private final Random random;

    public BeamSearchPlayer() {
        this(DEFAULT_DEPTH_LIMIT, DEFAULT_BEAM_WIDTH, System.nanoTime());
    }

    public BeamSearchPlayer(int depthLimit, int beamWidth, long seed) {
        this.depthLimit = Math.max(1, depthLimit);
        this.beamWidth = Math.max(1, beamWidth);
        this.random = new Random(seed);
    }

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        if (legal.isEmpty()) {
            return "quit";
        }
        // If quit is the only legal command, honour it.
        if (legal.size() == 1 && isQuit(legal.getFirst())) {
            return "quit";
        }

        int rootEval = evaluate(solitaire);
        Node root = new Node(solitaire.copy(), null, null, 0, 0, rootEval);
        List<Node> frontier = Collections.singletonList(root);

        Node bestNode = null;
        // Global visited set for this decision to avoid ping-ponging between the same states.
        java.util.Set<Long> globalSeenKeys = new java.util.HashSet<>();
        globalSeenKeys.add(root.state.getStateKey());

        for (int depth = 0; depth < depthLimit; depth++) {
            List<Node> nextFrontier = new ArrayList<>();
            for (Node node : frontier) {
                List<String> moves = LegalMovesHelper.listLegalMoves(node.state);
                // Mild branching factor cap per node.
                int expanded = 0;
                for (String move : moves) {
                    if (expanded >= beamWidth * 2) {
                        break;
                    }
                    if (isQuit(move)) {
                        continue;
                    }
                    // Foundation -> tableau moves are usually strategic backsteps; skip in lookahead.
                    if (move.startsWith("move F")) {
                        continue;
                    }
                    Solitaire copy = node.state.copy();
                    applyMove(copy, move);
                    long key = copy.getStateKey();
                    if (!globalSeenKeys.add(key)) {
                        continue;
                    }
                    int baseScore = evaluate(copy);
                    int turnsInPath = node.turns + (isTurn(move) ? 1 : 0);
                    int depthInPath = node.depth + 1;
                    int score = baseScore - depthInPath * PER_DEPTH_PENALTY - turnsInPath * PER_TURN_PENALTY;
                    Node child = new Node(copy, node, move, depthInPath, turnsInPath, score);
                    nextFrontier.add(child);
                    expanded++;
                    if (bestNode == null || score > bestNode.score) {
                        bestNode = child;
                    }
                }
            }
            if (nextFrontier.isEmpty()) {
                break;
            }
            // Keep only the top beamWidth nodes.
            nextFrontier.sort(Comparator.comparingInt((Node n) -> n.score).reversed());
            if (nextFrontier.size() > beamWidth) {
                nextFrontier = new ArrayList<>(nextFrontier.subList(0, beamWidth));
            }
            frontier = nextFrontier;
        }

        // If search failed to find anything clearly better than root, fall back to a simple move.
        if (bestNode == null || bestNode.score <= rootEval) {
            // Fallback: pick any non-quit legal move.
            return pickFallbackMove(legal);
        }

        // Walk back to root to find the first move in the chosen sequence.
        Node current = bestNode;
        Node previous = null;
        while (current.parent != null && current.parent != root) {
            previous = current;
            current = current.parent;
        }
        Node firstStep = (current.parent == root) ? current : previous;
        return firstStep != null && firstStep.move != null
                ? firstStep.move
                : pickFallbackMove(legal);
    }

    private String pickFallbackMove(List<String> legal) {
        List<String> nonQuit = new ArrayList<>();
        for (String m : legal) {
            if (!isQuit(m)) {
                nonQuit.add(m);
            }
        }
        if (nonQuit.isEmpty()) {
            return "quit";
        }
        return nonQuit.get(random.nextInt(nonQuit.size()));
    }

    private boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    private boolean isTurn(String move) {
        return move != null && move.trim().equalsIgnoreCase("turn");
    }

    private void applyMove(Solitaire solitaire, String move) {
        if (move == null) {
            return;
        }
        String trimmed = move.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else {
                solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
    }

    /**
     * Same heuristic as {@link HillClimberPlayer#evaluate(Solitaire)} to keep
     * evaluation consistent across AIs.
     */
    private int evaluate(Solitaire solitaire) {
        int score = 0;

        // Foundation progress: strongly encourage completing foundations.
        for (var pile : solitaire.getFoundation()) {
            score += pile.size() * 25;
        }

        // Tableau visibility: reward visible, penalise hidden.
        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        int emptyColumns = 0;
        for (int i = 0; i < faceUps.size(); i++) {
            int up = faceUps.get(i);
            int down = faceDowns.get(i);
            score += up * 6;
            score -= down * 4;
            if (up == 0 && down == 0) {
                emptyColumns++;
            }
        }

        // Reward creating empty tableau columns – they are valuable for king moves.
        score += emptyColumns * 10;

        // Stockpile drag: fewer buried cards is generally better.
        score -= solitaire.getStockpile().size();

        return score;
    }

    private static final class Node {
        final Solitaire state;
        final Node parent;
        final String move;
        final int depth;
        final int turns;
        final int score;

        Node(Solitaire state, Node parent, String move, int depth, int turns, int score) {
            this.state = state;
            this.parent = parent;
            this.move = move;
            this.depth = depth;
            this.turns = turns;
            this.score = score;
        }
    }
}

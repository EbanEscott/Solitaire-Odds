package ai.games.player.ai.alpha;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * AlphaSolitaire-backed AI player that delegates move selection to the
 * Python policy–value network via the HTTP service, using a small
 * Monte Carlo Tree Search (MCTS) loop for decision making.
 *
 * Requires the Python service to be running locally, for example
 * from the modeling project:
 *
 *   cd /Users/ebo/Code/solitaire/neural-network
 *   source .venv/bin/activate
 *   python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 */
@Component
@Profile("ai-alpha-solitaire")
public class AlphaSolitairePlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(AlphaSolitairePlayer.class);

    /**
     * Number of MCTS simulations to run per real move. This is intentionally
     * small because each simulation may trigger an HTTP call into the Python
     * policy–value service.
     */
    private static final int DEFAULT_MCTS_SIMULATIONS = 256;

    /**
     * Maximum depth (number of moves) for a single MCTS simulation path.
     */
    private static final int DEFAULT_MCTS_MAX_DEPTH = 12;

    /**
     * PUCT exploration constant controlling how strongly prior probabilities
     * bias selection relative to empirical value estimates.
     */
    private static final double DEFAULT_MCTS_CPUCT = 1.5;

    private final int mctsSimulations;
    private final int mctsMaxDepth;
    private final double mctsCpuct;

    private final AlphaSolitaireClient client;

    public AlphaSolitairePlayer(AlphaSolitaireClient client) {
        this.client = client;
        this.mctsSimulations = Integer.getInteger(
                "alphasolitaire.mcts.simulations",
                DEFAULT_MCTS_SIMULATIONS);
        this.mctsMaxDepth = Integer.getInteger(
                "alphasolitaire.mcts.maxDepth",
                DEFAULT_MCTS_MAX_DEPTH);
        this.mctsCpuct = Double.parseDouble(
                System.getProperty("alphasolitaire.mcts.cpuct",
                        Double.toString(DEFAULT_MCTS_CPUCT)));
    }

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        if (legal.isEmpty()) {
            return "quit";
        }
        if (legal.size() == 1) {
            String only = legal.getFirst();
            return isQuit(only) ? "quit" : only;
        }

        if (log.isDebugEnabled()) {
            log.debug(
                    "AlphaSolitairePlayer starting MCTS with {} legal moves (simulations={}, maxDepth={})",
                    legal.size(),
                    mctsSimulations,
                    mctsMaxDepth);
        }

        // Root node uses a copy of the current state so that MCTS can freely
        // explore without mutating the real game.
        MctsNode root = MctsNode.createRoot(solitaire.copy(), legal);

        // Ensure the root has priors and a value estimate from the neural net.
        root.ensureEvaluated(client);

        for (int sim = 0; sim < mctsSimulations; sim++) {
            runSimulation(root);
        }

        String chosen = root.bestMove();
        if (chosen == null || chosen.isBlank()) {
            log.warn("AlphaSolitaire MCTS returned no move; defaulting to \"quit\".");
            return "quit";
        }

        if (log.isDebugEnabled()) {
            log.debug(
                    "AlphaSolitaire MCTS chose move={} with visits={} (root valueEstimate={})",
                    chosen,
                    root.visitsForMove(chosen),
                    String.format("%.3f", root.getValueEstimate()));
        }

        return chosen;
    }

    private void runSimulation(MctsNode root) {
        List<MctsNode> pathNodes = new ArrayList<>();
        List<Integer> pathMoves = new ArrayList<>();

        MctsNode node = root;
        pathNodes.add(node);
        int depth = 0;

        while (true) {
            if (node.isTerminal() || depth >= mctsMaxDepth) {
                double terminalValue = node.getValueEstimate();
                backpropagate(pathNodes, pathMoves, terminalValue);
                return;
            }

            int moveIndex = node.selectChildIndex(mctsCpuct);
            if (moveIndex < 0) {
                double terminalValue = node.getValueEstimate();
                backpropagate(pathNodes, pathMoves, terminalValue);
                return;
            }

            pathMoves.add(moveIndex);

            MctsNode child = node.child(moveIndex);
            if (child == null) {
                // Expand this edge by creating a new state and evaluating it via the neural net.
                String move = node.moveAt(moveIndex);
                Solitaire nextState = node.copyState();
                applyMove(nextState, move);
                List<String> childLegal = LegalMovesHelper.listLegalMoves(nextState);

                child = MctsNode.createChild(nextState, childLegal);
                node.setChild(moveIndex, child);

                pathNodes.add(child);
                double value = child.ensureEvaluated(client);
                backpropagate(pathNodes, pathMoves, value);
                return;
            }

            node = child;
            pathNodes.add(node);
            depth++;
        }
    }

    private void backpropagate(List<MctsNode> nodes, List<Integer> moves, double value) {
        // Single-agent setting: value is always from the same perspective,
        // so we propagate the same scalar up the entire path.
        int steps = moves.size();
        for (int i = 0; i < steps; i++) {
            MctsNode node = nodes.get(i);
            int moveIndex = moves.get(i);
            node.updateStats(moveIndex, value);
        }
    }

    private static boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    private static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    private static void applyMove(Solitaire solitaire, String move) {
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
     * Simple state evaluation used when the neural service is unavailable.
     * Mirrors the structure of the MonteCarloPlayer heuristic but is kept
     * intentionally lightweight; primary guidance still comes from the neural
     * policy and value.
     */
    private static int heuristicScore(Solitaire solitaire) {
        int score = 0;

        int foundationCards = 0;
        for (var pile : solitaire.getFoundation()) {
            foundationCards += pile.size();
        }
        score += foundationCards * 40;

        List<Integer> faceUps = solitaire.getTableauFaceUpCounts();
        List<Integer> faceDowns = solitaire.getTableauFaceDownCounts();
        int emptyColumns = 0;
        for (int i = 0; i < faceUps.size(); i++) {
            int up = faceUps.get(i);
            int down = faceDowns.get(i);
            score += up * 4;
            score -= down * 9;
            if (up == 0 && down == 0) {
                emptyColumns++;
            }
        }

        score += emptyColumns * 20;
        score -= solitaire.getStockpile().size() * 2;

        return score;
    }

    /**
     * Single-node AlphaZero-style MCTS implementation.
     */
    private static final class MctsNode {
        private final Solitaire state;
        private final List<String> moves;
        private final double[] priors;
        private final double[] valueSums;
        private final int[] visits;
        private final MctsNode[] children;
        private boolean evaluated;
        private double valueEstimate;

        private MctsNode(Solitaire state, List<String> moves) {
            this.state = state;
            this.moves = moves;
            int n = moves.size();
            this.priors = new double[n];
            this.valueSums = new double[n];
            this.visits = new int[n];
            this.children = new MctsNode[n];
            this.evaluated = false;
            this.valueEstimate = 0.0;
        }

        static MctsNode createRoot(Solitaire state, List<String> legalMoves) {
            List<String> filtered = new ArrayList<>();
            for (String move : legalMoves) {
                if (!isQuit(move)) {
                    filtered.add(move);
                }
            }
            return new MctsNode(state, filtered);
        }

        static MctsNode createChild(Solitaire state, List<String> legalMoves) {
            List<String> filtered = new ArrayList<>();
            for (String move : legalMoves) {
                if (!isQuit(move)) {
                    filtered.add(move);
                }
            }
            return new MctsNode(state, filtered);
        }

        boolean isTerminal() {
            return moves.isEmpty() || isWon(state);
        }

        double getValueEstimate() {
            if (!evaluated) {
                // If we have not yet contacted the neural service, fall back
                // to a heuristic view of the current state.
                return valueEstimateFromHeuristic();
            }
            return valueEstimate;
        }

        double ensureEvaluated(AlphaSolitaireClient client) {
            if (evaluated) {
                return valueEstimate;
            }
            if (moves.isEmpty() || isWon(state)) {
                valueEstimate = isWon(state) ? 1.0 : 0.0;
                evaluated = true;
                return valueEstimate;
            }

            try {
                AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(state);
                AlphaSolitaireResponse response = client.evaluate(request);
                if (response != null) {
                    Map<String, Double> priorByMove = new HashMap<>();
                    List<AlphaSolitaireResponse.MoveScore> scored = response.getLegalMoves();
                    if (scored != null) {
                        for (AlphaSolitaireResponse.MoveScore ms : scored) {
                            if (ms.getCommand() != null) {
                                priorByMove.put(ms.getCommand().trim(), ms.getProbability());
                            }
                        }
                    }

                    double sum = 0.0;
                    for (int i = 0; i < moves.size(); i++) {
                        String m = moves.get(i);
                        double p = priorByMove.getOrDefault(m.trim(), 0.0);
                        priors[i] = p;
                        sum += p;
                    }
                    if (sum > 0.0) {
                        for (int i = 0; i < priors.length; i++) {
                            priors[i] /= sum;
                        }
                    } else {
                        double uniform = moves.isEmpty() ? 0.0 : 1.0 / moves.size();
                        for (int i = 0; i < priors.length; i++) {
                            priors[i] = uniform;
                        }
                    }

                    valueEstimate = clamp01(response.getWinProbability());
                } else {
                    valueEstimate = valueEstimateFromHeuristic();
                }
            } catch (Exception e) {
                valueEstimate = valueEstimateFromHeuristic();
            }

            evaluated = true;
            return valueEstimate;
        }

        private double valueEstimateFromHeuristic() {
            if (isWon(state)) {
                return 1.0;
            }
            int score = heuristicScore(state);
            double normalized = 1.0 / (1.0 + Math.exp(-score / 100.0));
            return normalized;
        }

        Solitaire copyState() {
            return state.copy();
        }

        String moveAt(int index) {
            return moves.get(index);
        }

        MctsNode child(int index) {
            return children[index];
        }

        void setChild(int index, MctsNode child) {
            children[index] = child;
        }

        void updateStats(int moveIndex, double value) {
            if (moveIndex < 0 || moveIndex >= moves.size()) {
                return;
            }
            valueSums[moveIndex] += value;
            visits[moveIndex] += 1;
        }

        int selectChildIndex(double cpuct) {
            int nMoves = moves.size();
            if (nMoves == 0) {
                return -1;
            }

            int totalVisits = 0;
            for (int n : visits) {
                totalVisits += n;
            }

            double bestScore = Double.NEGATIVE_INFINITY;
            int bestIndex = 0;

            for (int i = 0; i < nMoves; i++) {
                double p = priors[i];
                int n = visits[i];
                double q = n == 0 ? 0.0 : valueSums[i] / n;
                double u = cpuct * p * Math.sqrt(totalVisits + 1e-6) / (1 + n);
                double score = q + u;
                if (score > bestScore) {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        String bestMove() {
            int nMoves = moves.size();
            if (nMoves == 0) {
                return null;
            }

            int bestIndex = 0;
            int bestVisits = visits[0];

            for (int i = 1; i < nMoves; i++) {
                if (visits[i] > bestVisits) {
                    bestVisits = visits[i];
                    bestIndex = i;
                }
            }

            return moves.get(bestIndex);
        }

        int visitsForMove(String move) {
            if (move == null) {
                return 0;
            }
            String normalized = move.trim();
            for (int i = 0; i < moves.size(); i++) {
                if (normalized.equals(moves.get(i).trim())) {
                    return visits[i];
                }
            }
            return 0;
        }

        private static double clamp01(double v) {
            if (v < 0.0) {
                return 0.0;
            }
            if (v > 1.0) {
                return 1.0;
            }
            return v;
        }
    }
}

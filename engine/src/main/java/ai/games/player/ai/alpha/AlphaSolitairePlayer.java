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
 * AlphaSolitaire-backed AI player using Monte Carlo Tree Search (MCTS) guided by
 * a neural network policy–value network.
 *
 * Architecture:
 * - The neural network (Python service) outputs:
 *   * Policy head: probability distribution over legal moves
 *   * Value head: estimated win probability for the current position
 * - MCTS uses these signals to search the game tree efficiently:
 *   * Policy priors initialize move exploration (PUCT formula)
 *   * Value estimates provide rollout termination without full simulation
 *   * Tree is rebuilt fresh for each move (no persistence)
 *
 * Configuration:
 * - Simulations per move: controlled by -Dalphasolitaire.mcts.simulations (default 256)
 * - Max depth per simulation: controlled by -Dalphasolitaire.mcts.maxDepth (default 12)
 * - Exploration constant: controlled by -Dalphasolitaire.mcts.cpuct (default 1.5)
 *
 * Requirements:
 * - The Python service must be running at the configured endpoint
 * - Example: python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 *
 * If the neural service is unavailable, falls back to heuristic evaluation.
 */
@Component
@Profile("ai-alpha-solitaire")
public class AlphaSolitairePlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(AlphaSolitairePlayer.class);

    /**
     * Number of MCTS simulations to run per real move.
     * Intentionally small (256) because each simulation may trigger an HTTP call
     * to the Python service. Can be overridden via -Dalphasolitaire.mcts.simulations.
     */
    private static final int DEFAULT_MCTS_SIMULATIONS = 256;

    /**
     * Maximum depth (number of moves) for a single MCTS simulation path.
     * Prevents infinite exploration in rare deadlock scenarios.
     * Can be overridden via -Dalphasolitaire.mcts.maxDepth.
     */
    private static final int DEFAULT_MCTS_MAX_DEPTH = 12;

    /**
     * PUCT (Polynomial Upper Confidence bound applied to Trees) exploration constant.
     * Controls the balance between exploiting high-value moves and exploring
     * high-probability moves recommended by the policy head.
     * - Higher values: more exploration, follows policy priors more
     * - Lower values: more exploitation, follows empirical Q-values
     * Can be overridden via -Dalphasolitaire.mcts.cpuct.
     */
    private static final double DEFAULT_MCTS_CPUCT = 1.5;

    private final int mctsSimulations;
    private final int mctsMaxDepth;
    private final double mctsCpuct;

    private final AlphaSolitaireClient client;

    /**
     * Construct a new AlphaSolitairePlayer with HTTP client for the neural service.
     *
     * @param client the AlphaSolitaire HTTP client (auto-wired)
     */
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

    /**
     * Select the next move using MCTS guided by the neural network.
     *
     * Process:
     * 1. Extract legal moves from the current position
     * 2. Build a root MCTS node for this decision point
     * 3. Evaluate the root with the neural network (get priors and value)
     * 4. Run many simulations (default 256) exploring the tree
     * 5. Return the move with the highest visit count
     *
     * @param solitaire the current game state
     * @param moves unused (legacy parameter)
     * @param feedback unused (legacy parameter)
     * @return the next command to execute
     */
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
        TreeNode root = TreeNode.createRoot(solitaire.copy(), legal);

        // Ensure the root has priors and a value estimate from the neural net.
        root.ensureEvaluated(client);

        // Run the specified number of simulations
        for (int sim = 0; sim < mctsSimulations; sim++) {
            runSimulation(root);
        }

        // Select the best move based on visit counts alone (pure exploitation)
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

    /**
     * Run a single MCTS simulation from the root node.
     *
     * Process (tree policy and default policy combined):
     * 1. Select moves down the tree using PUCT (balance prior + value)
     * 2. When reaching an unexplored child, expand and evaluate it
     * 3. Backpropagate the value up the path
     * 4. Stop at terminal state, max depth, or first neural evaluation
     *
     * @param root the root node of the MCTS tree
     */
    private void runSimulation(TreeNode root) {
        List<TreeNode> pathNodes = new ArrayList<>();
        List<Integer> pathMoves = new ArrayList<>();

        TreeNode node = root;
        pathNodes.add(node);
        int depth = 0;

        while (true) {
            // Terminal state or max depth: stop and backpropagate
            if (node.isTerminal() || depth >= mctsMaxDepth) {
                double terminalValue = node.getValueEstimate();
                backpropagate(pathNodes, pathMoves, terminalValue);
                return;
            }

            // Select the best move using PUCT
            int moveIndex = node.selectChildIndex(mctsCpuct);
            if (moveIndex < 0) {
                // No moves (should be caught by isTerminal, but be safe)
                double terminalValue = node.getValueEstimate();
                backpropagate(pathNodes, pathMoves, terminalValue);
                return;
            }

            pathMoves.add(moveIndex);

            // Check if child already exists
            TreeNode child = node.child(moveIndex);
            if (child == null) {
                // Expand: create a new child node by applying the move
                String move = node.moveAt(moveIndex);
                Solitaire nextState = node.copyState();
                applyMove(nextState, move);
                List<String> childLegal = LegalMovesHelper.listLegalMoves(nextState);

                // Create child and store in parent
                child = TreeNode.createChild(nextState, childLegal);
                node.setChild(moveIndex, child);

                pathNodes.add(child);
                // Evaluate the new child with the neural network
                double value = child.ensureEvaluated(client);
                backpropagate(pathNodes, pathMoves, value);
                return;
            }

            // Descend to child
            node = child;
            pathNodes.add(node);
            depth++;
        }
    }

    /**
     * Backpropagate a value up the tree path.
     *
     * In single-agent Solitaire, the value perspective is always the same,
     * so we propagate the same scalar value up the entire path without negation.
     *
     * @param nodes the sequence of nodes from root to leaf (in order)
     * @param moves the sequence of move indices taken at each node (in order)
     * @param value the value to backpropagate (0.0 = loss, 1.0 = win)
     */
    private void backpropagate(List<TreeNode> nodes, List<Integer> moves, double value) {
        // Single-agent setting: value is always from the same perspective,
        // so we propagate the same scalar up the entire path.
        int steps = moves.size();
        for (int i = 0; i < steps; i++) {
            TreeNode node = nodes.get(i);
            int moveIndex = moves.get(i);
            node.updateStats(moveIndex, value);
        }
    }

    /**
     * Check if a move string is the "quit" command.
     *
     * @param move the move command string
     * @return true if the move is "quit"
     */
    private static boolean isQuit(String move) {
        return move != null && move.trim().equalsIgnoreCase("quit");
    }

    /**
     * Check if the game is won from the given state.
     *
     * @param solitaire the game state
     * @return true if all 52 cards are in the foundation piles
     */
    private static boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (var pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
    }

    /**
     * Apply a move command to a game state.
     * Handles "turn" (draw 3 cards from stockpile) and "move" (card movement).
     *
     * @param solitaire the game state to modify
     * @param move the command string (e.g., "turn", "move T1 A♥ F1")
     */
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
     * Evaluate a position using heuristics when the neural service is unavailable.
     *
     * Factors considered:
     * - Foundation cards (40 points each): direct progress toward winning
     * - Visible tableau cards (4 points each): movable cards
     * - Face-down cards (-9 points each): blocking cards
     * - Empty tableau columns (20 points each): valuable for future moves
     * - Stockpile cards (-2 points each): remaining cards to process
     *
     * The score is converted to a probability via sigmoid: 1 / (1 + exp(-score/100))
     *
     * @param solitaire the game state to evaluate
     * @return heuristic score (used as input to sigmoid)
     */
    static int heuristicScore(Solitaire solitaire) {
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
}

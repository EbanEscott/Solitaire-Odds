package ai.games;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exhaustive game tree analysis for understanding the scale of search spaces.
 * <p>
 * This test class explores complete game trees for multiple Solitaire games by:
 * <ol>
 *   <li>Generating N distinct games (with different random seeds).</li>
 *   <li>For each game, exhaustively exploring all possible moves and turns.</li>
 *   <li>Recording statistics: total nodes, max depth, branching factors, etc.</li>
 *   <li>Computing aggregate statistics across all games.</li>
 * </ol>
 * <p>
 * Results help inform AI training decisions, such as:
 * <ul>
 *   <li>What depth limits are practical for exhaustive search?</li>
 *   <li>How wide are game trees (branching factor)?</li>
 *   <li>What fraction of games are solvable via exhaustive search?</li>
 * </ul>
 */
public class GameTreeAnalysisTest {

    /**
     * Statistics collected for a single game's exhaustive tree exploration.
     */
    static class GameTreeStats {
        long totalNodes;
        int maxDepth;
        int winCount;
        int lossCount;
        double avgBranchingFactor;
        long explorationTimeMs;

        GameTreeStats(long totalNodes, int maxDepth, int winCount, int lossCount,
                      double avgBranchingFactor, long explorationTimeMs) {
            this.totalNodes = totalNodes;
            this.maxDepth = maxDepth;
            this.winCount = winCount;
            this.lossCount = lossCount;
            this.avgBranchingFactor = avgBranchingFactor;
            this.explorationTimeMs = explorationTimeMs;
        }

        @Override
        public String toString() {
            return String.format(
                    "Nodes=%d, MaxDepth=%d, Wins=%d, Losses=%d, AvgBranching=%.2f, Time=%dms",
                    totalNodes, maxDepth, winCount, lossCount, avgBranchingFactor, explorationTimeMs
            );
        }
    }

    /**
     * Memoization key: encodes game state for cycle detection.
     */
    static class StateKey {
        final long zobrist;

        StateKey(long zobrist) {
            this.zobrist = zobrist;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            StateKey stateKey = (StateKey) o;
            return zobrist == stateKey.zobrist;
        }

        @Override
        public int hashCode() {
            return Long.hashCode(zobrist);
        }
    }

    /**
     * Exhaustively explores a single game tree, collecting statistics.
     * <p>
     * Uses depth-first search with cycle detection (Zobrist hashing) to avoid revisiting states.
     * Records wins (all cards in foundation), losses (no more legal moves), and intermediate states.
     * <p>
     * Note: This uses a fresh copy of the game for each state to ensure clean move history.
     *
     * @param originalGame the Solitaire game to explore
     * @param maxDepthLimit safety limit to prevent stack overflow on pathological games
     * @return aggregated statistics for the entire tree
     */
    private GameTreeStats exploreGameTree(Solitaire originalGame, int maxDepthLimit) {
        long startTime = System.currentTimeMillis();

        Set<StateKey> visited = new HashSet<>();
        AtomicLong nodeCount = new AtomicLong(0);
        AtomicLong winCount = new AtomicLong(0);
        AtomicLong lossCount = new AtomicLong(0);
        int[] maxDepthFound = {0};
        double[] sumBranchingFactors = {0.0};
        long[] branchingFactorCount = {0};

        // Start DFS from a copy of the original game
        dfs(originalGame.copy(), visited, nodeCount, winCount, lossCount, maxDepthFound, 0, maxDepthLimit,
                sumBranchingFactors, branchingFactorCount);

        long endTime = System.currentTimeMillis();
        double avgBranching = branchingFactorCount[0] > 0
                ? sumBranchingFactors[0] / branchingFactorCount[0]
                : 0.0;

        return new GameTreeStats(
                nodeCount.get(),
                maxDepthFound[0],
                (int) winCount.get(),
                (int) lossCount.get(),
                avgBranching,
                endTime - startTime
        );
    }

    /**
     * Depth-first search of the game tree with cycle detection.
     * <p>
     * For each state:
     * <ol>
     *   <li>Check if state was already visited (cycle detection).</li>
     *   <li>Mark as visited and increment node count.</li>
     *   <li>Check for terminal states (win if all foundation cards at King, loss if no moves).</li>
     *   <li>Try all legal moves (card moves, turns).</li>
     *   <li>Recursively explore each child state using a fresh copy.</li>
     * </ol>
     * <p>
     * Uses game copies instead of undo to avoid move history synchronization issues.
     */
    private void dfs(Solitaire game, Set<StateKey> visited, AtomicLong nodeCount,
                     AtomicLong winCount, AtomicLong lossCount, int[] maxDepth,
                     int currentDepth, int maxDepthLimit,
                     double[] sumBranchingFactors, long[] branchingFactorCount) {

        // Depth limit check
        if (currentDepth > maxDepthLimit) {
            return;
        }

        // Cycle detection
        StateKey stateKey = new StateKey(game.getStateKey());
        if (visited.contains(stateKey)) {
            return;
        }
        visited.add(stateKey);
        nodeCount.incrementAndGet();

        // Update max depth
        if (currentDepth > maxDepth[0]) {
            maxDepth[0] = currentDepth;
        }

        // Check for terminal states
        if (isWin(game)) {
            winCount.incrementAndGet();
            return;
        }

        // Collect all legal moves
        List<Move> legalMoves = collectLegalMoves(game);

        if (legalMoves.isEmpty()) {
            lossCount.incrementAndGet();
            return;
        }

        // Record branching factor
        sumBranchingFactors[0] += legalMoves.size();
        branchingFactorCount[0]++;

        // Explore each move using a fresh copy
        for (Move move : legalMoves) {
            Solitaire gameCopy = game.copy();

            // Make move on the copy
            if (move.isTurn) {
                gameCopy.turnThree();
                gameCopy.recordAction(Solitaire.Action.turn());
            } else {
                gameCopy.moveCard(move.from, move.cardCode, move.to);
            }

            // Recurse on the copy
            dfs(gameCopy, visited, nodeCount, winCount, lossCount, maxDepth,
                    currentDepth + 1, maxDepthLimit, sumBranchingFactors, branchingFactorCount);
        }
    }

    /**
     * Checks if the game is in a winning state (all cards in foundation).
     *
     * @param game the Solitaire game to check
     * @return true if all four foundation piles contain all 13 cards
     */
    private boolean isWin(Solitaire game) {
        for (List<?> foundationPile : game.getFoundation()) {
            if (foundationPile.size() != 13) {
                return false;
            }
        }
        return true;
    }

    /**
     * Simple move representation for tree exploration.
     */
    static class Move {
        String from;
        String cardCode;
        String to;
        boolean isTurn;

        Move(String from, String cardCode, String to) {
            this.from = from;
            this.cardCode = cardCode;
            this.to = to;
            this.isTurn = false;
        }

        Move(boolean isTurn) {
            this.isTurn = isTurn;
        }
    }

    /**
     * Collects all legal moves from the current game state.
     * <p>
     * Includes:
     * <ul>
     *   <li>All card moves from tableau/foundation/stockpile/talon to tableau/foundation.</li>
     *   <li>Turn action (turn three cards from stock to talon).</li>
     * </ul>
     * <p>
     * Note: This method tests moves on a copy to avoid modifying the original game state.
     *
     * @param game the Solitaire game
     * @return list of all legal moves available
     */
    private List<Move> collectLegalMoves(Solitaire game) {
        List<Move> moves = new ArrayList<>();

        // Try moving from each pile to each destination
        String[] sources = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "F1", "F2", "F3", "F4", "W", "S"};
        String[] destinations = {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "F1", "F2", "F3", "F4"};

        for (String source : sources) {
            for (String dest : destinations) {
                if (source.equals(dest)) continue;
                // Test if this move is legal on a copy
                Solitaire gameCopy = game.copy();
                if (gameCopy.moveCard(source, dest)) {
                    // Move was successful; record it
                    moves.add(new Move(source, null, dest));
                }
            }
        }

        // Add turn action if stock is not empty
        if (!game.getStockpile().isEmpty()) {
            moves.add(new Move(true));
        }

        return moves;
    }

    /**
     * Main test: exhaustively explores game trees for 1000 unique games.
     * <p>
     * Generates games with different random shuffles, explores their complete trees,
     * and aggregates statistics across all games. Outputs summary statistics and per-game details.
     * <p>
     * Uses a depth limit of 15 to keep total runtime reasonable while still exploring substantial
     * portions of the game tree for each game.
     * <p>
     * Expected behavior:
     * <ul>
     *   <li>Completion in reasonable time (games with small trees only).</li>
     *   <li>Discovery of average branching factor (~3-5 moves per state).</li>
     *   <li>Identification of solvable games (reachable win states).</li>
     *   <li>Understanding of maximum practical search depths.</li>
     * </ul>
     */
    @Test
    void testExhaustiveGameTreeAnalysis_1000Games() {
        int numGames = 1000;
        int maxDepthPerGame = 15; // Reduced from 20 to keep runtime manageable

        List<GameTreeStats> allStats = new ArrayList<>();
        long overallStartTime = System.currentTimeMillis();

        System.out.println("\n========================================");
        System.out.println("Exhaustive Game Tree Analysis: " + numGames + " Games");
        System.out.println("Max Depth Limit per Game: " + maxDepthPerGame);
        System.out.println("========================================\n");

        for (int gameNum = 1; gameNum <= numGames; gameNum++) {
            // Create a unique game with a different random seed
            Deck deck = new Deck();
            Solitaire game = new Solitaire(deck);

            // Explore the complete tree
            GameTreeStats stats = exploreGameTree(game, maxDepthPerGame);
            allStats.add(stats);

            // Print progress every 100 games
            if (gameNum % 100 == 0 || gameNum == 1) {
                System.out.printf("Game %4d: %s%n", gameNum, stats);
            }
        }

        long overallEndTime = System.currentTimeMillis();

        // Aggregate statistics
        long totalNodesAcrossGames = allStats.stream().mapToLong(s -> s.totalNodes).sum();
        double avgNodesPerGame = allStats.stream().mapToLong(s -> s.totalNodes).average().orElse(0.0);
        long maxNodesInAnyGame = allStats.stream().mapToLong(s -> s.totalNodes).max().orElse(0);
        long minNodesInAnyGame = allStats.stream().mapToLong(s -> s.totalNodes).min().orElse(0);

        int totalWins = allStats.stream().mapToInt(s -> s.winCount).sum();
        int totalLosses = allStats.stream().mapToInt(s -> s.lossCount).sum();
        double avgMaxDepth = allStats.stream().mapToInt(s -> s.maxDepth).average().orElse(0.0);
        double avgBranchingFactor = allStats.stream().mapToDouble(s -> s.avgBranchingFactor).average().orElse(0.0);

        long totalExplorationTime = allStats.stream().mapToLong(s -> s.explorationTimeMs).sum();

        System.out.println("\n========================================");
        System.out.println("AGGREGATE STATISTICS");
        System.out.println("========================================");
        System.out.printf("Total Games Analyzed:       %d%n", numGames);
        System.out.printf("Total Nodes (all games):    %,d%n", totalNodesAcrossGames);
        System.out.printf("Avg Nodes per Game:         %.1f%n", avgNodesPerGame);
        System.out.printf("Max Nodes in a Game:        %,d%n", maxNodesInAnyGame);
        System.out.printf("Min Nodes in a Game:        %,d%n", minNodesInAnyGame);
        System.out.printf("Std Dev (Nodes):            %.1f%n", calculateStdDev(allStats));
        System.out.printf("%n");
        System.out.printf("Total Win States Found:     %d (%.2f%%)%n",
                totalWins, (100.0 * totalWins / totalNodesAcrossGames));
        System.out.printf("Total Loss States Found:    %d (%.2f%%)%n",
                totalLosses, (100.0 * totalLosses / totalNodesAcrossGames));
        System.out.printf("Total Intermediate States:  %,d (%.2f%%)%n",
                totalNodesAcrossGames - totalWins - totalLosses,
                (100.0 * (totalNodesAcrossGames - totalWins - totalLosses) / totalNodesAcrossGames));
        System.out.printf("%n");
        System.out.printf("Avg Max Depth per Game:     %.1f%n", avgMaxDepth);
        System.out.printf("Avg Branching Factor:       %.2f%n", avgBranchingFactor);
        System.out.printf("%n");
        System.out.printf("Total Exploration Time:     %,d ms (%.2f s)%n",
                totalExplorationTime, totalExplorationTime / 1000.0);
        System.out.printf("Overall Elapsed Time:       %,d ms (%.2f s)%n",
                overallEndTime - overallStartTime,
                (overallEndTime - overallStartTime) / 1000.0);
        System.out.println("========================================\n");

        // Basic sanity check
        assertTrue(avgNodesPerGame > 0, "Should have explored at least some nodes");
    }

    /**
     * Quick test: exhaustively explores game trees for 100 unique games.
     * <p>
     * Faster version of the full 1000-game test, suitable for quick feedback during development.
     */
    @Test
    void testExhaustiveGameTreeAnalysis_100Games_Quick() {
        int numGames = 100;
        int maxDepthPerGame = 15;

        List<GameTreeStats> allStats = new ArrayList<>();
        long overallStartTime = System.currentTimeMillis();

        System.out.println("\n========================================");
        System.out.println("Exhaustive Game Tree Analysis: " + numGames + " Games (QUICK)");
        System.out.println("Max Depth Limit per Game: " + maxDepthPerGame);
        System.out.println("========================================\n");

        for (int gameNum = 1; gameNum <= numGames; gameNum++) {
            Deck deck = new Deck();
            Solitaire game = new Solitaire(deck);
            GameTreeStats stats = exploreGameTree(game, maxDepthPerGame);
            allStats.add(stats);

            if (gameNum % 10 == 0 || gameNum == 1) {
                System.out.printf("Game %3d: %s%n", gameNum, stats);
            }
        }

        long overallEndTime = System.currentTimeMillis();

        // Aggregate statistics
        long totalNodesAcrossGames = allStats.stream().mapToLong(s -> s.totalNodes).sum();
        double avgNodesPerGame = allStats.stream().mapToLong(s -> s.totalNodes).average().orElse(0.0);
        long maxNodesInAnyGame = allStats.stream().mapToLong(s -> s.totalNodes).max().orElse(0);
        long minNodesInAnyGame = allStats.stream().mapToLong(s -> s.totalNodes).min().orElse(0);

        int totalWins = allStats.stream().mapToInt(s -> s.winCount).sum();
        int totalLosses = allStats.stream().mapToInt(s -> s.lossCount).sum();
        double avgMaxDepth = allStats.stream().mapToInt(s -> s.maxDepth).average().orElse(0.0);
        double avgBranchingFactor = allStats.stream().mapToDouble(s -> s.avgBranchingFactor).average().orElse(0.0);

        long totalExplorationTime = allStats.stream().mapToLong(s -> s.explorationTimeMs).sum();

        System.out.println("\n========================================");
        System.out.println("AGGREGATE STATISTICS (100 GAMES)");
        System.out.println("========================================");
        System.out.printf("Total Games Analyzed:       %d%n", numGames);
        System.out.printf("Total Nodes (all games):    %,d%n", totalNodesAcrossGames);
        System.out.printf("Avg Nodes per Game:         %.1f%n", avgNodesPerGame);
        System.out.printf("Max Nodes in a Game:        %,d%n", maxNodesInAnyGame);
        System.out.printf("Min Nodes in a Game:        %,d%n", minNodesInAnyGame);
        System.out.printf("Std Dev (Nodes):            %.1f%n", calculateStdDev(allStats));
        System.out.printf("%n");
        System.out.printf("Total Win States Found:     %d (%.2f%%)%n",
                totalWins, (100.0 * totalWins / totalNodesAcrossGames));
        System.out.printf("Total Loss States Found:    %d (%.2f%%)%n",
                totalLosses, (100.0 * totalLosses / totalNodesAcrossGames));
        System.out.printf("Total Intermediate States:  %,d (%.2f%%)%n",
                totalNodesAcrossGames - totalWins - totalLosses,
                (100.0 * (totalNodesAcrossGames - totalWins - totalLosses) / totalNodesAcrossGames));
        System.out.printf("%n");
        System.out.printf("Avg Max Depth per Game:     %.1f%n", avgMaxDepth);
        System.out.printf("Avg Branching Factor:       %.2f%n", avgBranchingFactor);
        System.out.printf("%n");
        System.out.printf("Total Exploration Time:     %,d ms (%.2f s)%n",
                totalExplorationTime, totalExplorationTime / 1000.0);
        System.out.printf("Overall Elapsed Time:       %,d ms (%.2f s)%n",
                overallEndTime - overallStartTime,
                (overallEndTime - overallStartTime) / 1000.0);
        System.out.println("========================================\n");

        assertTrue(avgNodesPerGame > 0, "Should have explored at least some nodes");
    }

    /**
     * Calculates standard deviation of node counts across games.
     *
     * @param stats list of statistics for all games
     * @return standard deviation of node counts
     */
    private double calculateStdDev(List<GameTreeStats> stats) {
        double mean = stats.stream().mapToLong(s -> s.totalNodes).average().orElse(0.0);
        double variance = stats.stream()
                .mapToDouble(s -> Math.pow(s.totalNodes - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }
}

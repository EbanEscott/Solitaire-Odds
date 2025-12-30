package ai.games.training;

import ai.games.Game;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.astar.AStarPlayer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates generation of endgame training data for the neural network.
 * 
 * <p>This test class:
 * <ul>
 *   <li>Seeds deterministic endgame positions using {@link TrainingOpponent}.</li>
 *   <li>Injects each seeded game into the {@link Game} orchestrator with a configurable player.</li>
 *   <li>The {@link Game} class automatically logs episodes when {@code -Dlog.episodes=true} is set.</li>
 * </ul>
 * 
 * <p>This approach reuses the same episode-logging infrastructure as production AI players,
 * ensuring training data matches the format and quality of real game logs.</p>
 * 
 * <p>Difficulty levels (cards off foundations):</p>
 * <ul>
 *   <li><strong>Level 1:</strong> 0 cards off; all 52 on foundations (baseline).</li>
 *   <li><strong>Level 2:</strong> 1 card off; 51 on foundations.</li>
 *   <li><strong>Level 3:</strong> 2 cards off; 50 on foundations.</li>
 *   <li><strong>Level 4:</strong> 4 cards off; 48 on foundations.</li>
 *   <li><strong>Level 5:</strong> 7 cards off; 45 on foundations.</li>
 * </ul>
 * 
 * <p>Usage: Run test methods with episode logging enabled:</p>
 * <pre>
 * ./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel2" -Dlog.episodes=true
 * </pre>
 */
public class EndgameTrainingDataGenerator {
    private static final Logger log = LoggerFactory.getLogger(EndgameTrainingDataGenerator.class);
    
    // Number of games per level. Can be overridden via system property -Dendgame.games.per.level
    // Default is 500 for full dataset generation; use smaller values for quick tests.
    private static final int GAMES_PER_LEVEL = Integer.getInteger("endgame.games.per.level", 5000);
    
    // The player to use for solving seeded positions.
    // Default: A* player (near-optimal solver for clean training data).
    // Can be overridden via setPlayer() for customization.
    private static Player solverPlayer = new AStarPlayer();

    /**
     * Sets a custom player to use for all subsequent test runs.
     * 
     * @param player the AI player to use for solving seeded endgame positions
     */
    public static void setPlayer(Player player) {
        solverPlayer = player;
    }

    /**
     * Gets the current solver player.
     */
    public static Player getPlayer() {
        return solverPlayer;
    }

    @Test
    @DisplayName("Level 1: All foundations full (0 cards off)")
    void testEndgameLevel1() {
        generateLevelDataset(1, "Level 1 (all foundations full)");
    }

    @Test
    @DisplayName("Level 2: 51 foundation cards (1 off)")
    void testEndgameLevel2() {
        generateLevelDataset(2, "Level 2 (1 card off)");
    }

    @Test
    @DisplayName("Level 3: 50 foundation cards (2 off)")
    void testEndgameLevel3() {
        generateLevelDataset(3, "Level 3 (2 cards off)");
    }

    @Test
    @org.junit.jupiter.api.Disabled("Level 3+ not yet implemented")
    @DisplayName("Level 4: 48 foundation cards (4 off)")
    void testEndgameLevel4() {
        generateLevelDataset(4, "Level 4 (4 cards off)");
    }

    @Test
    @org.junit.jupiter.api.Disabled("Level 3+ not yet implemented")
    @DisplayName("Level 5: 45 foundation cards (7 off)")
    void testEndgameLevel5() {
        generateLevelDataset(5, "Level 5 (7 cards off)");
    }

    /**
     * Generates a complete dataset for a single difficulty level.
     * 
     * <p>For each game:
     * <ol>
     *   <li>Seeds an endgame position via {@link TrainingOpponent}.</li>
     *   <li>Creates a {@link Game} instance with the seeded Solitaire and the solver player.</li>
     *   <li>Calls {@link Game#play()} which automatically logs episodes (if -Dlog.episodes=true).</li>
     * </ol>
     */
    private void generateLevelDataset(int level, String levelName) {
        log.info("Starting endgame training data generation: {}", levelName);
        
        TrainingOpponent opponent = new TrainingOpponent(level);
        List<TrainingOpponent.SeededGame> seededGames = opponent.seedGameWithMoves(GAMES_PER_LEVEL);
        
        Stats stats = new Stats();
        List<Integer> lostGameNumbers = new ArrayList<>();
        
        int gamesProcessed = 0;
        for (TrainingOpponent.SeededGame seededGameWithMoves : seededGames) {
            gamesProcessed++;
            int gameNum = gamesProcessed;
            if (log.isInfoEnabled()) {
                log.info("[{}] Playing game {}/{}", levelName, gameNum, seededGames.size());
            }
            
            // Log the reverse moves used to generate this game for easy debugging
            if (!seededGameWithMoves.reverseMoves.isEmpty()) {
                log.debug("Reverse moves: {}", seededGameWithMoves.reverseMoves);
            }
            
            // Create a Game instance with the solver player and seeded board
            // Episode logging is automatic when -Dlog.episodes=true is set
            Game game = new Game(solverPlayer);
            Game.GameResult result = game.play(seededGameWithMoves.game);
            
            stats.recordGame(result.isWon(), result.getMoves(), result.getDurationNanos());
            
            if (!result.isWon()) {
                lostGameNumbers.add(gamesProcessed);
            }
        }
        
        // Print summary stats
        printSummary(levelName, stats, lostGameNumbers);
    }
    
    /**
     * Prints summary statistics and list of lost games.
     */
    private void printSummary(String levelName, Stats stats, List<Integer> lostGameNumbers) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n").append("=".repeat(80)).append("\n");
        sb.append("TRAINING DATASET SUMMARY: ").append(levelName).append("\n");
        sb.append("=".repeat(80)).append("\n");
        sb.append(String.format("Games Played: %d%n", stats.games));
        sb.append(String.format("Games Won:    %d%n", stats.wins));
        sb.append(String.format("Games Lost:   %d%n", stats.losses()));
        sb.append(String.format("Win Rate:     %.2f%%%n", stats.winPercent()));
        sb.append(String.format("Avg Moves:    %.2f%n", stats.avgMoves()));
        sb.append(String.format("Avg Time:     %.3fs%n", stats.avgTimeSeconds()));
        sb.append(String.format("Total Time:   %.3fs%n", stats.totalTimeSeconds()));
        sb.append("=".repeat(80)).append("\n");
        
        if (!lostGameNumbers.isEmpty()) {
            sb.append("\nLost games (search for 'Playing game <number>' in logs):\n");
            for (int gameNum : lostGameNumbers) {
                sb.append(String.format("  >> game %d%n", gameNum));
            }
        }
        
        log.info(sb.toString());
    }
    
    /**
     * Statistics collector for training runs.
     */
    private static class Stats {
        int games = 0;
        int wins = 0;
        long totalTimeNanos = 0;
        int totalMoves = 0;

        void recordGame(boolean won, int moves, long nanos) {
            games++;
            if (won) {
                wins++;
            }
            totalMoves += moves;
            totalTimeNanos += nanos;
        }

        int losses() {
            return games - wins;
        }

        double winPercent() {
            return games == 0 ? 0.0 : (wins * 100.0) / games;
        }

        double avgMoves() {
            return games == 0 ? 0.0 : (double) totalMoves / games;
        }

        double totalTimeSeconds() {
            return totalTimeNanos / 1_000_000_000.0;
        }

        double avgTimeSeconds() {
            return games == 0 ? 0.0 : totalTimeSeconds() / games;
        }
    }
}

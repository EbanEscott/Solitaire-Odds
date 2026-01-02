package ai.games.training;

import ai.games.Game;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.alpha.AlphaSolitaireClient;
import ai.games.player.ai.alpha.AlphaSolitairePlayer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Tests the AlphaSolitairePlayer on progressively harder endgame difficulty levels.
 * 
 * <p>This test class evaluates the neural network-based AI player's performance
 * across different difficulty levels. It uses the same seeding mechanism as
 * {@link EndgameTrainingDataGenerator} to generate deterministic endgame positions.</p>
 * 
 * <p>Difficulty levels (cards off foundations):</p>
 * <ul>
 *   <li><strong>Level 2:</strong> 1 card off; 51 on foundations.</li>
 *   <li><strong>Level 3:</strong> 2 cards off; 50 on foundations.</li>
 *   <li><strong>Level 4:</strong> 4 cards off; 48 on foundations.</li>
 *   <li><strong>Level 5:</strong> 7 cards off; 45 on foundations.</li>
 * </ul>
 * 
 * <p>Usage: Run with system properties to control level and game count:</p>
 * <ul>
 *   <li><strong>Specific level only:</strong> {@code ./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponent" -Dendgame.games.difficulty.level=4 -Dendgame.games.per.level=10 -Dalphasolitaire.mcts.simulations=256}</li>
 *   <li><strong>All individual level tests:</strong> {@code ./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponentLevel*" -Dendgame.games.per.level=5}</li>
 * </ul>
 * 
 * <p><strong>Prerequisites:</strong> The AlphaSolitaire neural service must be running:
 * <pre>
 * cd neural-network
 * python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 * </pre>
 */
public class AlphaSolitaireLevelTest {
    private static final Logger log = LoggerFactory.getLogger(AlphaSolitaireLevelTest.class);
    
    // Difficulty level to test (only used if > 0). Can be overridden via system property -Dendgame.games.difficulty.level
    // When set, runs only that level. When not set (0), legacy individual tests are used.
    private static final int DIFFICULTY_LEVEL = Integer.getInteger("endgame.games.difficulty.level", 0);
    
    // Number of games per level. Can be overridden via system property -Dendgame.games.per.level
    // Default is 5 for quick testing; use larger values for comprehensive evaluation.
    private static final int GAMES_PER_LEVEL = Integer.getInteger("endgame.games.per.level", 5);

    /**
     * Creates a fresh AlphaSolitairePlayer instance for testing.
     * Each test gets its own player to avoid state pollution.
     * 
     * Note: This requires the AlphaSolitaire neural service to be running:
     * {@code python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000}
     */
    private Player createAlphaPlayer() {
        // Create HTTP client for neural service
        AlphaSolitaireClient client = new AlphaSolitaireClient();
        return new AlphaSolitairePlayer(client);
    }

    @Test
    @DisplayName("Level 2: 51 foundation cards (1 off)")
    void testOpponentLevel2() {
        testOpponentAtLevel(2, "Level 2 (1 card off)");
    }

    @Test
    @DisplayName("Level 3: 50 foundation cards (2 off)")
    void testOpponentLevel3() {
        testOpponentAtLevel(3, "Level 3 (2 cards off)");
    }

    @Test
    @DisplayName("Level 4: 48 foundation cards (4 off)")
    void testOpponentLevel4() {
        testOpponentAtLevel(4, "Level 4 (4 cards off)");
    }

    @Test
    @DisplayName("Level 5: 45 foundation cards (7 off)")
    void testOpponentLevel5() {
        testOpponentAtLevel(5, "Level 5 (7 cards off)");
    }

    /**
     * Single test method that evaluates AlphaSolitairePlayer on the difficulty level specified
     * via system property {@code -Dendgame.games.difficulty.level}.
     * 
     * <p>Usage: {@code ./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest.testOpponent" -Dendgame.games.difficulty.level=4 -Dendgame.games.per.level=10}</p>
     * 
     * <p>The difficulty level can be any positive integer. If not set, this test is skipped.</p>
     */
    @Test
    @DisplayName("Test AlphaSolitairePlayer on specified difficulty level")
    void testOpponent() {
        if (DIFFICULTY_LEVEL <= 0) {
            log.info("Skipping testOpponent: set -Dendgame.games.difficulty.level=<level> to run");
            return;
        }
        
        String levelName = String.format("Level %d", DIFFICULTY_LEVEL);
        testOpponentAtLevel(DIFFICULTY_LEVEL, levelName);
    }

    /**
     * Evaluates AlphaSolitairePlayer performance on a single difficulty level.
     * 
     * <p>For each game:
     * <ol>
     *   <li>Seeds an endgame position via {@link TrainingOpponent}.</li>
     *   <li>Creates a {@link Game} instance with the seeded Solitaire and AlphaSolitairePlayer.</li>
     *   <li>Calls {@link Game#play()} and records the result.</li>
     * </ol>
     */
    private void testOpponentAtLevel(int level, String levelName) {
        log.info("Starting AlphaSolitairePlayer evaluation: {}", levelName);
        
        TrainingOpponent opponent = new TrainingOpponent(level);
        List<TrainingOpponent.SeededGame> seededGames = opponent.seedGameWithMoves(GAMES_PER_LEVEL);
        
        Stats stats = new Stats();
        List<Integer> lostGameNumbers = new ArrayList<>();
        
        int gamesProcessed = 0;
        for (TrainingOpponent.SeededGame seededGameWithMoves : seededGames) {
            gamesProcessed++;
            int gameNum = gamesProcessed;
            if (log.isInfoEnabled()) {
                log.info("[{}] Testing game {}/{}", levelName, gameNum, seededGames.size());
            }
            
            // Log the reverse moves used to generate this game for easy debugging
            if (!seededGameWithMoves.reverseMoves.isEmpty()) {
                log.debug("Reverse moves: {}", seededGameWithMoves.reverseMoves);
            }
            
            // Create a Game instance with AlphaSolitairePlayer and seeded board
            Player alphaPlayer = createAlphaPlayer();
            Game game = new Game(alphaPlayer);
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
        sb.append("ALPHASOLITAIRE EVALUATION SUMMARY: ").append(levelName).append("\n");
        sb.append("=".repeat(80)).append("\n");
        sb.append(String.format("Games Tested: %d%n", stats.games));
        sb.append(String.format("Games Won:    %d%n", stats.wins));
        sb.append(String.format("Games Lost:   %d%n", stats.losses()));
        sb.append(String.format("Win Rate:     %.2f%%%n", stats.winPercent()));
        sb.append(String.format("Avg Moves:    %.2f%n", stats.avgMoves()));
        sb.append(String.format("Avg Time:     %.3fs%n", stats.avgTimeSeconds()));
        sb.append(String.format("Total Time:   %.3fs%n", stats.totalTimeSeconds()));
        sb.append("=".repeat(80)).append("\n");
        
        if (!lostGameNumbers.isEmpty()) {
            sb.append("\nLost games (search for 'Testing game <number>' in logs):\n");
            for (int gameNum : lostGameNumbers) {
                sb.append(String.format("  >> game %d%n", gameNum));
            }
        }
        
        log.info(sb.toString());
    }
    
    /**
     * Statistics collector for evaluation runs.
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

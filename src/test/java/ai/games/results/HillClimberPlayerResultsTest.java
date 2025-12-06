package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.Game;
import ai.games.Game.GameResult;
import ai.games.player.Player;
import ai.games.player.ai.HillClimberPlayer;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Aggregates multiple games for the hill-climbing AI and logs summary stats
 * to help fill the comparison table. Use -Dgames=N to adjust the number of games (default 20).
 */
public class HillClimberPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(HillClimberPlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    @Test
    void playMultipleGamesAndReport() {
        int gamesToPlay = ResultsConfig.GAMES;
        Stats stats = runGames("Hill-Climber", () -> new HillClimberPlayer(123L), gamesToPlay);
        String summary = String.format("| %s | %s | %d | %d | %.2f%% \u00b1 %.2f%% | %.3fs | %.3fs | %.2f | %d | %s |",
                "Hill-climbing Search",
                "Search",
                stats.games,
                stats.wins,
                stats.winPercent(),
                stats.winPercentConfidenceInterval(),
                stats.avgTimeSeconds(),
                stats.totalTimeSeconds(),
                stats.avgMoves(),
                stats.bestWinStreak,
                "Local hill-climbing with restarts over hashed game states; see [code](src/main/java/ai/games/player/ai/HillClimberPlayer.java).");
        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);
        System.out.println(summary);
        log.info(summary);
        assertTrue(stats.games == gamesToPlay);
    }

    private Stats runGames(String playerName, Supplier<Player> playerSupplier, int games) {
        Stats stats = new Stats(games);
        for (int i = 0; i < games; i++) {
            int gameNumber = i + 1;
            if (gameNumber == 1
                    || gameNumber % ResultsConfig.PROGRESS_LOG_INTERVAL == 0
                    || gameNumber == games) {
                System.out.printf("[%s] Running game %d/%d%n", playerName, gameNumber, games);
            }
            Player ai = playerSupplier.get();
            Game game = new Game(ai);
            GameResult result = game.play();
            stats.recordGame(result.isWon(), result.getMoves(), result.getDurationNanos());
        }
        return stats;
    }

    private static class Stats {
        final int games;
        int wins = 0;
        long totalTimeNanos = 0;
        int totalMoves = 0;
        int bestWinStreak = 0;
        int currentStreak = 0;

        Stats(int games) {
            this.games = games;
        }

        void recordGame(boolean won, int moves, long nanos) {
            if (won) {
                wins++;
                currentStreak++;
                bestWinStreak = Math.max(bestWinStreak, currentStreak);
            } else {
                currentStreak = 0;
            }
            totalMoves += moves;
            totalTimeNanos += nanos;
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

        double winPercentConfidenceInterval() {
            if (games == 0) {
                return 0.0;
            }
            double p = wins / (double) games;
            double standardError = Math.sqrt(p * (1.0 - p) / games);
            double halfWidth = 1.96 * standardError * 100.0;
            return halfWidth;
        }
    }
}

package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.games.Game;
import ai.games.Game.GameResult;
import ai.games.player.ai.TypeSafePlayer;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Batch runner for TypeSafe AI's Jev System One decision model. */
public class TypeSafePlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(TypeSafePlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    /** Runs a random-game sweep and prints a README-compatible summary row. */
    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("typesafe.tests"), "Enable with -Dtypesafe.tests=true");

        int gamesToPlay = ResultsConfig.GAMES;
        System.setProperty("max.moves.per.game", String.valueOf(ResultsConfig.MAX_MOVES_PER_GAME));
        Stats stats = new Stats(gamesToPlay);

        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);
        for (int index = 0; index < gamesToPlay; index++) {
            int gameNumber = index + 1;
            System.out.printf(
                    "[TypeSafe %s prompt=%s policy=%s] Running game %d/%d%n",
                    TypeSafePlayer.configuredModelName(),
                    TypeSafePlayer.configuredPromptProfile(),
                    TypeSafePlayer.configuredPolicyVersion(),
                    gameNumber,
                    gamesToPlay);
            System.setProperty("game.index", String.valueOf(gameNumber));
            System.setProperty("game.total", String.valueOf(gamesToPlay));

            TypeSafePlayer player = new TypeSafePlayer();
            Game game = new Game(player);
            game.setGuidanceEnabled(false);
            GameResult result = game.play();
            stats.recordGame(
                    result.isWon(),
                    result.getMoves(),
                    result.getDurationNanos(),
                    player.getTotalInputTokens(),
                    player.getTotalOutputTokens(),
                    player.getTotalLatencyMillis());
            log.info(
                    "TypeSafe game result: model={}, prompt={}, policy={}, game={}/{}, won={}, moves={}, durationSeconds={}, inputTokens={}, outputTokens={}, apiLatencyMillis={}",
                    TypeSafePlayer.configuredModelName(),
                    TypeSafePlayer.configuredPromptProfile(),
                    TypeSafePlayer.configuredPolicyVersion(),
                    gameNumber,
                    gamesToPlay,
                    result.isWon(),
                    result.getMoves(),
                    String.format("%.3f", result.getDurationNanos() / 1_000_000_000.0),
                    player.getTotalInputTokens(),
                    player.getTotalOutputTokens(),
                    player.getTotalLatencyMillis());
        }

        String notes = "TypeSafe AI " + TypeSafePlayer.configuredModelName()
                + " via System One API (typed Choice decisions, prompt="
                + TypeSafePlayer.configuredPromptProfile()
                + ", policy="
                + TypeSafePlayer.configuredPolicyVersion()
                + ", compact observed-board history in structured state, engine guidance disabled); see [code](src/main/java/ai/games/player/ai/TypeSafePlayer.java).";
        String summary = String.format(
                "| %s | Decision | %d | %d | %.2f%% ± %.2f%% | %s | %s | %.2f | %d | %s |",
                "TypeSafe " + TypeSafePlayer.configuredModelName(),
                stats.games,
                stats.wins,
                stats.winPercent(),
                stats.winPercentConfidenceInterval(),
                ResultsDurationFormatter.formatSeconds(stats.avgTimeSeconds()),
                ResultsDurationFormatter.formatSeconds(stats.totalTimeSeconds()),
                stats.avgMoves(),
                stats.bestWinStreak,
                notes);
        System.out.println(summary);
        log.info(summary);
        log.info(
                "TypeSafe sweep usage: inputTokens={}, outputTokens={}, apiLatencyMillis={}",
                stats.totalInputTokens,
                stats.totalOutputTokens,
                stats.totalApiLatencyMillis);
        assertEquals(gamesToPlay, stats.games);
    }

    /** Aggregates README statistics and API usage for the complete sweep. */
    private static class Stats {
        final int games;
        int wins;
        long totalTimeNanos;
        int totalMoves;
        int bestWinStreak;
        int currentStreak;
        long totalInputTokens;
        long totalOutputTokens;
        long totalApiLatencyMillis;

        Stats(int games) {
            this.games = games;
        }

        void recordGame(
                boolean won,
                int moves,
                long nanos,
                long inputTokens,
                long outputTokens,
                long apiLatencyMillis) {
            if (won) {
                wins++;
                currentStreak++;
                bestWinStreak = Math.max(bestWinStreak, currentStreak);
            } else {
                currentStreak = 0;
            }
            totalMoves += moves;
            totalTimeNanos += nanos;
            totalInputTokens += inputTokens;
            totalOutputTokens += outputTokens;
            totalApiLatencyMillis += apiLatencyMillis;
        }

        double winPercent() {
            return games == 0 ? 0.0 : wins * 100.0 / games;
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
            return 1.96 * Math.sqrt(p * (1.0 - p) / games) * 100.0;
        }
    }
}

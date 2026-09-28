package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.games.Game;
import ai.games.Game.GameResult;
import ai.games.player.ai.CopilotCliPlayer;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Batch runner for persistent-session GitHub Copilot CLI models. */
public class CopilotCliPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(CopilotCliPlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    /** Runs the requested random-game sweep and prints a README-compatible summary row. */
    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("copilot.tests"), "Enable with -Dcopilot.tests=true");

        int gamesToPlay = ResultsConfig.GAMES;
        System.setProperty("max.moves.per.game", String.valueOf(ResultsConfig.MAX_MOVES_PER_GAME));

        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);
        for (String modelName : configuredModels()) {
            Stats stats = runGames(modelName, gamesToPlay);
            String notes = "Anthropic " + modelName
                    + " via GitHub Copilot CLI (subscription, default reasoning, prompt="
                    + CopilotCliPlayer.configuredPromptProfile()
                    + ", persistent session, engine guidance disabled); see [code](src/main/java/ai/games/player/ai/CopilotCliPlayer.java).";
            String summary = String.format(
                    "| %s | LLM | %d | %d | %.2f%% ± %.2f%% | %s | %s | %.2f | %d | %s |",
                    "Copilot CLI " + modelName,
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
            assertEquals(gamesToPlay, stats.games);
        }
    }

    /** Resolves a comma-separated model sweep or falls back to the player default. */
    private List<String> configuredModels() {
        String configured = System.getProperty("copilot.models");
        if (configured == null || configured.isBlank()) {
            return List.of(CopilotCliPlayer.configuredModelName());
        }
        return Arrays.stream(configured.split(","))
                .map(String::trim)
                .filter(model -> !model.isEmpty())
                .toList();
    }

    /** Plays independent random deals with a fresh persistent session for every game. */
    private Stats runGames(String modelName, int games) {
        Stats stats = new Stats(games);
        for (int i = 0; i < games; i++) {
            int gameNumber = i + 1;
            System.out.printf("[Copilot CLI %s] Running game %d/%d%n", modelName, gameNumber, games);
            System.setProperty("game.index", String.valueOf(gameNumber));
            System.setProperty("game.total", String.valueOf(games));

            try (CopilotCliPlayer player = new CopilotCliPlayer(modelName)) {
                Game game = new Game(player);
                game.setGuidanceEnabled(false);
                GameResult result = game.play();
                stats.recordGame(result.isWon(), result.getMoves(), result.getDurationNanos());
                log.info(
                        "Copilot game result: model={}, game={}/{}, session={}, won={}, moves={}, durationSeconds={}, finalCommand={}, premiumRequests={}",
                        modelName,
                        gameNumber,
                        games,
                        player.getSessionId(),
                        result.isWon(),
                        result.getMoves(),
                        String.format("%.3f", result.getDurationNanos() / 1_000_000_000.0),
                        player.getLastCommand(),
                        String.format("%.2f", player.getTotalPremiumRequests()));
            }
        }
        return stats;
    }

    /** Aggregates the same statistics used by the root README result table. */
    private static class Stats {
        final int games;
        int wins;
        long totalTimeNanos;
        int totalMoves;
        int bestWinStreak;
        int currentStreak;

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

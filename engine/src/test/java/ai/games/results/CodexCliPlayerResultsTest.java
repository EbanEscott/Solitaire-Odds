package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.games.Game;
import ai.games.Game.GameResult;
import ai.games.player.ai.CodexCliPlayer;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Batch runner for the Codex CLI-backed player.
 *
 * <p>Requires Codex CLI to be authenticated through a ChatGPT subscription. Enable with
 * {@code -Dcodex.tests=true}; configure models with {@code -Dcodex.models} and reasoning with
 * {@code -Dcodex.reasoning.effort}.
 */
public class CodexCliPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(CodexCliPlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("codex.tests"), "Enable with -Dcodex.tests=true");

        int gamesToPlay = ResultsConfig.GAMES;
        System.setProperty("max.moves.per.game", String.valueOf(ResultsConfig.MAX_MOVES_PER_GAME));

        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);
        for (String modelName : configuredModels()) {
            String reasoningEffort = CodexCliPlayer.configuredReasoningEffort(modelName);
            Stats stats = runGames(modelName, reasoningEffort, gamesToPlay);

            String notes = "OpenAI " + modelName + " via Codex CLI (ChatGPT subscription, reasoning="
                    + reasoningEffort
                    + ", persistent session, self-authored strategy, engine guidance disabled); see [code](src/main/java/ai/games/player/ai/CodexCliPlayer.java).";
            String summary = String.format(
                    "| %s | %s | %d | %d | %.2f%% \u00b1 %.2f%% | %.3fs | %.3fs | %.2f | %d | %s |",
                    "Codex CLI " + modelName,
                    "LLM",
                    stats.games,
                    stats.wins,
                    stats.winPercent(),
                    stats.winPercentConfidenceInterval(),
                    stats.avgTimeSeconds(),
                    stats.totalTimeSeconds(),
                    stats.avgMoves(),
                    stats.bestWinStreak,
                    notes);

            System.out.println(summary);
            log.info(summary);
            assertTrue(stats.games == gamesToPlay);
        }
    }

    private List<String> configuredModels() {
        String configured = System.getProperty("codex.models");
        if (configured == null || configured.isBlank()) {
            return List.of(CodexCliPlayer.configuredModelName());
        }
        return Arrays.stream(configured.split(","))
                .map(String::trim)
                .filter(model -> !model.isEmpty())
                .toList();
    }

    private Stats runGames(String modelName, String reasoningEffort, int games) {
        Stats stats = new Stats(games);
        for (int i = 0; i < games; i++) {
            int gameNumber = i + 1;
            if (gameNumber == 1
                    || gameNumber % ResultsConfig.PROGRESS_LOG_INTERVAL == 0
                    || gameNumber == games) {
                System.out.printf("[Codex CLI %s] Running game %d/%d%n", modelName, gameNumber, games);
            }
            System.setProperty("game.index", String.valueOf(gameNumber));
            System.setProperty("game.total", String.valueOf(games));
            try (CodexCliPlayer player = new CodexCliPlayer(modelName, reasoningEffort)) {
                Game game = new Game(player);
                game.setGuidanceEnabled(false);
                GameResult result = game.play();
                stats.recordGame(result.isWon(), result.getMoves(), result.getDurationNanos());
                log.info(
                        "Codex game result: model={}, reasoning={}, game={}/{}, session={}, won={}, moves={}, durationSeconds={}, finalCommand={}",
                        modelName,
                        reasoningEffort,
                        gameNumber,
                        games,
                        player.getSessionId(),
                        result.isWon(),
                        result.getMoves(),
                        String.format("%.3f", result.getDurationNanos() / 1_000_000_000.0),
                        player.getLastCommand());
            }
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

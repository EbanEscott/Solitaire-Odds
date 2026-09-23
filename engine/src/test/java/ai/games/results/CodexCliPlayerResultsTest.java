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

/** Batch runner for Codex CLI authenticated through a ChatGPT subscription. */
public class CodexCliPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(CodexCliPlayerResultsTest.class);

    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("codex.tests"), "Enable with -Dcodex.tests=true");

        int gamesToPlay = ResultsConfig.GAMES;
        System.setProperty("max.moves.per.game", String.valueOf(ResultsConfig.MAX_MOVES_PER_GAME));
        for (String modelName : configuredModels()) {
            String reasoningEffort = CodexCliPlayer.configuredReasoningEffort(modelName);
            Stats stats = runGames(modelName, reasoningEffort, gamesToPlay);

            String notes = "OpenAI " + modelName + " via Codex CLI (ChatGPT subscription, reasoning="
                    + reasoningEffort + "); see [code](src/main/java/ai/games/player/ai/CodexCliPlayer.java).";
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
            GameResult result = new Game(new CodexCliPlayer(modelName, reasoningEffort)).play();
            stats.recordGame(result.isWon(), result.getMoves(), result.getDurationNanos());
        }
        return stats;
    }

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

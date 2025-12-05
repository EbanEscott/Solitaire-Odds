package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.OllamaModelInfo;
import ai.games.player.ai.OllamaPlayer;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Batch runner for the Ollama-backed AI. Requires local Ollama server; set -Dollama.tests=true to run.
 */
public class OllamaPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(OllamaPlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Algorithm                     | AI   | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("ollama.tests"), "Enable with -Dollama.tests=true (requires local Ollama)");

        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);

        int gamesToPlay = ResultsConfig.GAMES;
        List<String> models = resolveModels();
        assumeTrue(!models.isEmpty(), "Configure at least one model with -Dollama.models=model1,model2 or -Dollama.model=name");

        for (String modelName : models) {
            OllamaModelInfo modelInfo = OllamaModelInfo.byModelName(modelName).orElse(null);
            String algorithmLabel = modelInfo != null
                    ? modelInfo.algorithmLabel()
                    : "Ollama (" + modelName + ")";

            Stats stats = runGames(algorithmLabel, () -> new OllamaPlayer(modelName), gamesToPlay, ResultsConfig.MAX_MOVES_PER_GAME);

            if (modelInfo != null) {
                // Helper line that can be copied into docs:
                // e.g. "OpenAIPlayer gpt-oss:120b https://ollama.com/library/gpt-oss"
                System.out.println(modelInfo.getPlayerName() + " " + modelInfo.getModelName() + " " + modelInfo.getUrl());
            }

            String notes = modelInfo != null
                    ? "[`OllamaPlayer`](src/main/java/ai/games/player/ai/OllamaPlayer.java), [" + modelInfo.getProvider() + "'s " + modelInfo.getModelName() + "](" + modelInfo.getUrl() + ")"
                    : "[`OllamaPlayer`](src/main/java/ai/games/player/ai/OllamaPlayer.java)";

            String summary = String.format("| %s | %s | %d | %d | %.2f%% \u00b1 %.2f%% | %.3fs | %.3fs | %.2f | %d | %s |",
                    algorithmLabel,
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

    private List<String> resolveModels() {
        String modelsProp = System.getProperty("ollama.models");
        if (modelsProp != null && !modelsProp.isBlank()) {
            return Arrays.stream(modelsProp.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .toList();
        }
        String singleModel = System.getProperty("ollama.model");
        if (singleModel != null && !singleModel.isBlank()) {
            return List.of(singleModel.trim());
        }
        return List.of();
    }

    private Stats runGames(String playerName, Supplier<Player> playerSupplier, int games, int maxMovesPerGame) {
        Stats stats = new Stats(games);
        for (int i = 0; i < games; i++) {
            int gameNumber = i + 1;
            if (gameNumber == 1
                    || gameNumber % ResultsConfig.PROGRESS_LOG_INTERVAL == 0
                    || gameNumber == games) {
                System.out.printf("[%s] Running game %d/%d%n", playerName, gameNumber, games);
            }
            Player ai = playerSupplier.get();
            Solitaire solitaire = new Solitaire(new Deck());
            long start = System.nanoTime();
            int moves = 0;
            boolean won = false;
            for (int step = 0; step < maxMovesPerGame; step++) {
                String command = ai.nextCommand(solitaire, "");
                if (command == null || "quit".equalsIgnoreCase(command.trim())) {
                    break;
                }
                if (applyCommand(solitaire, command)) {
                    moves++;
                }
                if (isWon(solitaire)) {
                    won = true;
                    break;
                }
            }
            long duration = System.nanoTime() - start;
            stats.recordGame(won, moves, duration);
        }
        return stats;
    }

    private boolean applyCommand(Solitaire solitaire, String command) {
        String trimmed = command.trim();
        if ("turn".equalsIgnoreCase(trimmed)) {
            solitaire.turnThree();
            return true;
        }
        String[] parts = trimmed.split("\\s+");
        if ("move".equalsIgnoreCase(parts[0])) {
            if (parts.length == 4) {
                return solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else if (parts.length == 3) {
                return solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
        return false;
    }

    private boolean isWon(Solitaire solitaire) {
        int total = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total == 52;
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
            return (wins * 100.0) / games;
        }

        double avgTimeSeconds() {
            return totalTimeSeconds() / games;
        }

        double totalTimeSeconds() {
            return totalTimeNanos / 1_000_000_000.0;
        }

        double avgMoves() {
            return games == 0 ? 0 : totalMoves / (double) games;
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

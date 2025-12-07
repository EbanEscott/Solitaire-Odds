package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.games.Game;
import ai.games.Game.GameResult;
import ai.games.player.Player;
import ai.games.player.ai.alpha.AlphaSolitaireClient;
import ai.games.player.ai.alpha.AlphaSolitairePlayer;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Aggregates multiple games for the AlphaSolitaire (MCTS + neural policy/value) AI and logs summary
 * stats to help fill the comparison table.
 *
 * Requires the Python AlphaSolitaire model service to be running locally, and is gated behind
 * -Dalphasolitaire.tests=true so that it does not run in normal CI by default.
 *
 * Example:
 *
 * <pre>
 *   # In neural-network/ (Python side)
 *   python -m venv .venv
 *   source .venv/bin/activate
 *   python -m pip install -r requirements.txt
 *   python -m src.service --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 *
 *   # In engine/ (Java side)
 *   ./gradlew test --tests ai.games.results.AlphaSolitairePlayerResultsTest --console=plain --rerun-tasks -Dalphasolitaire.tests=true
 * </pre>
 */
public class AlphaSolitairePlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(AlphaSolitairePlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|";

    @Test
    void playMultipleGamesAndReport() {
        assumeTrue(Boolean.getBoolean("alphasolitaire.tests"),
                "Enable with -Dalphasolitaire.tests=true (requires Python AlphaSolitaire service)");

        int gamesToPlay = ResultsConfig.GAMES;
        System.setProperty("max.moves.per.game", String.valueOf(ResultsConfig.MAX_MOVES_PER_GAME));

        String playerLabel = "AlphaSolitaire (MCTS + NN)";
        Stats stats = runGames(playerLabel, () -> new AlphaSolitairePlayer(new AlphaSolitaireClient()), gamesToPlay);

        String notes =
                "MCTS guided by neural policy–value network served from the Python modeling stack; "
                        + "see [code](src/main/java/ai/games/player/ai/alpha/AlphaSolitairePlayer.java) "
                        + "and [Python model](../neural-network).";

        String summary = String.format(
                "| %s | %s | %d | %d | %.2f%% \u00b1 %.2f%% | %.3fs | %.3fs | %.2f | %d | %s |",
                playerLabel,
                "Search",
                stats.games,
                stats.wins,
                stats.winPercent(),
                stats.winPercentConfidenceInterval(),
                stats.avgTimeSeconds(),
                stats.totalTimeSeconds(),
                stats.avgMoves(),
                stats.bestWinStreak,
                notes);

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
            System.setProperty("game.index", String.valueOf(gameNumber));
            System.setProperty("game.total", String.valueOf(games));
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

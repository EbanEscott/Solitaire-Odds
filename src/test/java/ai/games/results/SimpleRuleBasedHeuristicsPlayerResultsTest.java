package ai.games.results;

import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.SimpleRuleBasedHeuristicsPlayer;
import java.util.List;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Aggregates multiple games for the simple rule-based heuristics AI and logs summary stats
 * to help fill the comparison table. Use -Dgames=N to adjust the number of games (default 20).
 */
public class SimpleRuleBasedHeuristicsPlayerResultsTest {
    private static final Logger log = LoggerFactory.getLogger(SimpleRuleBasedHeuristicsPlayerResultsTest.class);
    private static final String TABLE_HEADER = "| Algorithm                     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak |";
    private static final String TABLE_DIVIDER = "|------------------------------|--------------|-----------|-------|---------------|------------|-----------|-----------------|";

    @Test
    void playMultipleGamesAndReport() {
        int gamesToPlay = ResultsConfig.GAMES;
        Stats stats = runGames("Simple Rule-based Heuristics", SimpleRuleBasedHeuristicsPlayer::new, gamesToPlay, ResultsConfig.MAX_MOVES_PER_GAME);
        String summary = String.format("| %s | %d | %d | %.2f%% \u00b1 %.2f%% | %.3fs | %.3fs | %.2f | %d |",
                "Simple Rule-based Heuristics",
                stats.games,
                stats.wins,
                stats.winPercent(),
                stats.winPercentConfidenceInterval(),
                stats.avgTimeSeconds(),
                stats.totalTimeSeconds(),
                stats.avgMoves(),
                stats.bestWinStreak);
        System.out.println(TABLE_HEADER);
        System.out.println(TABLE_DIVIDER);
        System.out.println(summary);
        log.info(summary);
        assertTrue(stats.games == gamesToPlay);
    }

    private Stats runGames(String playerName, Supplier<Player> playerSupplier, int games, int maxMovesPerGame) {
        Stats stats = new Stats(games);
        for (int i = 0; i < games; i++) {
            int gameNumber = i + 1;
            if (gameNumber == 1 || gameNumber % 50 == 0 || gameNumber == games) {
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

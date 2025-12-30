package ai.games.training;

import ai.games.Game;
import ai.games.game.Solitaire;
import ai.games.player.Player;
import ai.games.player.ai.astar.AStarPlayer;
import ai.games.unit.helpers.GameStateDirector;
import ai.games.unit.helpers.SolitaireTestHelper;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Walkthrough test for debugging specific failing games.
 *
 * <p>Use this to reconstruct a specific game state from the training dataset
 * and step through the AI's decision-making with a debugger.
 *
 * <p><strong>Usage Steps:</strong></p>
 * <ol>
 *   <li><strong>Find a failing game:</strong> Run the level test and note which game failed.
 *       You'll see output like ">> game 34" in the summary.</li>
 *   <li><strong>Enable debug logs:</strong> Run with debug logging to see reverse moves:
 *       <pre>
 *       ./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel3" \
 *         --console=plain -Dlog.level=DEBUG -Dlog.episodes=false --rerun-tasks 2>&1 | tee test.log
 *       </pre>
 *   </li>
 *   <li><strong>Find the reverse moves:</strong> In test.log or game.log, search for "=== GAME 34 ===" and look backward for:
 *       <pre>
 *       Generated Level 3 game: move F4 K♠ T7 -> foundation_count=50
 *       Generated Level 2 game: move F3 K♥ T3 -> foundation_count=51
 *       Generated Level 2 game: move F2 K♦ T3 -> foundation_count=51
 *       Generated Level 1 game: move F1 K♣ T1 -> foundation_count=51
 *       </pre>
 *       Extract the moves in reverse order (bottom-to-top) and add them to the reverseMoves list below.</li>
 *   <li><strong>Set breakpoints:</strong> Open AStarPlayer.java and set a breakpoint in nextCommand() method.</li>
 *   <li><strong>Run this test:</strong> Run WalkthroughGameTest.debugGame34() with the debugger.</li>
 *   <li><strong>Step through:</strong> Step through the A* logic to see why it quit.</li>
 * </ol>
 */
public class WalkthroughGameTest {
    private static final Logger log = LoggerFactory.getLogger(WalkthroughGameTest.class);

    /**
     * Reconstructs and plays a specific game for debugging.
     *
     * <p>To find the reverse moves for a failing game:
     * <ol>
     *   <li>Run the test and note which game failed (e.g., ">> game 34")</li>
     *   <li>Find the game in game.log by searching for: <code>=== GAME 34 ===</code></li>
     *   <li>Copy the reverse moves from the log line. For example:
     *       <pre>
     *       === GAME 34 === Reverse moves: [move F1 K♣ T1, move F2 K♦ T3]
     *       </pre>
     *   </li>
     *   <li>Paste these moves into the reverseMoves list below (in the same order)</li>
     *   <li>Set a breakpoint in AStarPlayer.nextCommand()</li>
     *   <li>Run this test with the debugger</li>
     *   <li>Step through to see why it quits</li>
     * </ol>
     */
    @Test
    void debugGame34() {
        // Copy the reverse moves directly from game.log (=== GAME 34 === line)
        // Example from: === GAME 34 === Reverse moves: [move F1 K♣ T1, move F2 K♦ T3]
        List<String> reverseMoves = new ArrayList<>();
        reverseMoves.add("move F1 K♣ T1");
        reverseMoves.add("move F2 K♦ T3");

        playGameWithReverseMoves(reverseMoves);
    }

    /**
     * Reconstructs a game state by starting with a won board and applying reverse moves,
     * then plays it with the A* player so you can debug.
     *
     * @param reverseMoves list of moves to apply (e.g., ["move F1 K♣ T1", "move F2 K♦ T2"])
     */
    private void playGameWithReverseMoves(List<String> reverseMoves) {
        log.info("Reconstructing game with {} reverse moves", reverseMoves.size());

        // Start with a completely won board
        Solitaire board = createCompletelyWonBoard();
        SolitaireTestHelper.assertFullDeckState(board);

        // Apply the reverse moves to reconstruct the game state
        for (int i = 0; i < reverseMoves.size(); i++) {
            String move = reverseMoves.get(i);
            log.info("Applying reverse move {}/{}: {}", i + 1, reverseMoves.size(), move);
            
            boolean success = GameStateDirector.applyMoveDirectly(board, move);
            if (!success) {
                log.error("Failed to apply reverse move: {}", move);
                return;
            }
        }

        log.info("Game state reconstructed. Starting playthrough...");
        logBoardState(board);

        // Play the game with A* player
        Player aiPlayer = new AStarPlayer();
        Game game = new Game(aiPlayer);
        Game.GameResult result = game.play(board);

        log.info("Game finished: won={}, moves={}, duration={}ms",
                result.isWon(), result.getMoves(), result.getDurationNanos() / 1_000_000);
    }

    /**
     * Creates a Solitaire board with all 52 cards on foundations (completely won state).
     * Copied from TrainingOpponent for convenience.
     */
    private Solitaire createCompletelyWonBoard() {
        List<List<ai.games.game.Card>> foundationPiles = new ArrayList<>();
        for (ai.games.game.Suit suit : ai.games.game.Suit.values()) {
            List<ai.games.game.Card> suitPile = new ArrayList<>();
            for (ai.games.game.Rank rank : ai.games.game.Rank.values()) {
                suitPile.add(new ai.games.game.Card(rank, suit));
            }
            foundationPiles.add(suitPile);
        }

        Solitaire dummy = new Solitaire(new ai.games.game.Deck());

        SolitaireTestHelper.setTableau(dummy,
                List.of(
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile(),
                        SolitaireTestHelper.emptyPile()
                ),
                List.of(0, 0, 0, 0, 0, 0, 0)
        );
        SolitaireTestHelper.setFoundation(dummy, foundationPiles);
        SolitaireTestHelper.setTalon(dummy, SolitaireTestHelper.emptyPile());
        SolitaireTestHelper.setStockpile(dummy, SolitaireTestHelper.emptyPile());

        return dummy;
    }

    /**
     * Logs the current board state for reference.
     */
    private void logBoardState(Solitaire solitaire) {
        log.info("Board state:");
        log.info("  Foundations: {}", solitaire.getFoundation());
        log.info("  Tableau visible: {}", solitaire.getTableau());
        log.info("  Talon: {}", solitaire.getTalon());
        log.info("  Stock size: {}", solitaire.getStockpile().size());
    }
}

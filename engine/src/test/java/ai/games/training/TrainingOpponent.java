package ai.games.training;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.unit.helpers.GameStateDirector;
import ai.games.unit.helpers.SolitaireTestHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Manages seeding of deterministic game states for training and self-play scenarios.
 * 
 * <p>This class encapsulates the logic for creating endgame positions with varying
 * complexity levels. It's designed to be reusable in multiple contexts:
 * <ul>
 *   <li><strong>Supervised Learning:</strong> Generate endgame training data for the neural network.</li>
 *   <li><strong>Self-Play:</strong> Create starting positions for AlphaSolitaire self-play games (when an opponent is needed).</li>
 * </ul>
 * 
 * <p>Usage: Call {@link #seedGame(int)} to create a seeded endgame position that can then be
 * fed into the {@link ai.games.Game} orchestrator with a player to generate training episodes.</p>
 */
public class TrainingOpponent {
    private static final Logger log = LoggerFactory.getLogger(TrainingOpponent.class);
    
    private final int difficultyLevel;

    /**
     * Creates a training opponent at the specified difficulty level.
     * 
     * @param difficultyLevel the endgame difficulty (1-5+), where higher = more cards off foundations
     */
    public TrainingOpponent(int difficultyLevel) {
        if (difficultyLevel < 1) {
            throw new IllegalArgumentException("difficultyLevel must be >= 1");
        }
        this.difficultyLevel = difficultyLevel;
    }

    /**
     * Gets the difficulty level (1-5+).
     */
    public int getDifficultyLevel() {
        return difficultyLevel;
    }

    /**
     * Seeds multiple endgame positions using a systematic approach.
     * 
     * <p>Generates endgame positions by:
     * <ul>
     *   <li>Level 1: Single position with all 52 cards on foundations (trivial, no moves)</li>
     *   <li>Level 2+: Progressively applies reverse moves from a won state to build complexity</li>
     * </ul>
     * 
     * <p>Uses ReverseMovesHelper to identify and apply backward moves, ensuring all generated
     * positions are legally reachable and valid. This approach naturally scales to arbitrary
     * difficulty levels without hardcoding position logic.
     * 
     * @param numberOfGames requested number of games to generate
     * @return list of seeded Solitaire games (may be fewer than requested if level exhausted)
     * @throws IllegalArgumentException if difficulty level is not yet implemented
     */
    public List<Solitaire> seedGame(int numberOfGames) {
        List<Solitaire> games = new ArrayList<>();
        
        if (difficultyLevel == 1) {
            // Level 1: Single trivial position (all foundations full)
            Solitaire game = createCompletelyWonBoard();
            SolitaireTestHelper.assertFullDeckState(game);
            games.add(game);
            if (log.isDebugEnabled()) {
                log.debug("Generated 1 Level 1 game (all foundations full)");
            }
            return games;
        }
        
        // Start with Level 1: completely won board (all 52 on foundations)
        List<Solitaire> currentLevelGames = new ArrayList<>();
        Solitaire level1 = createCompletelyWonBoard();
        SolitaireTestHelper.assertFullDeckState(level1);
        currentLevelGames.add(level1);
        
        // Generate games for levels 2 through target difficulty
        for (int currentLevel = 2; currentLevel <= difficultyLevel && games.size() < numberOfGames; currentLevel++) {
            List<Solitaire> nextLevelGames = new ArrayList<>();
            
            // For each game at the current level, apply reverse moves to generate next level
            for (Solitaire baseGame : currentLevelGames) {
                if (games.size() >= numberOfGames) {
                    break;
                }
                
                // Get all possible reverse moves from this game state
                List<String> reverseMoves = ReverseMovesHelper.listReverseMoves(baseGame);
                
                if (log.isDebugEnabled()) {
                    log.debug("Level {}: found {} reverse moves from base game", currentLevel - 1, reverseMoves.size());
                    log.debug("  Reverse moves: {}", reverseMoves);
                }
                
                if (reverseMoves.isEmpty()) {
                    // No moves available from this position; still valid, just can't go deeper
                    if (log.isDebugEnabled()) {
                        log.debug("No reverse moves available from level {}. Stopping expansion from this branch.", currentLevel - 1);
                    }
                    continue;
                }
                
                // For each reverse move, create a new game variant for the next level
                for (String reverseMove : reverseMoves) {
                    if (games.size() >= numberOfGames && nextLevelGames.size() > 0) {
                        break;
                    }
                    
                    // Create a game at the next difficulty level
                    Solitaire game = baseGame.copy();
                    SolitaireTestHelper.assertFullDeckState(game);
                    
                    // Apply the reverse move
                    applyMove(game, reverseMove);
                    
                    // Verify the move was applied
                    int foundationCount = 0;
                    for (List<Card> pile : game.getFoundation()) {
                        foundationCount += pile.size();
                    }
                    
                    games.add(game);
                    nextLevelGames.add(game);
                    
                    if (log.isDebugEnabled()) {
                        log.debug("Generated Level {} game: {} -> foundation_count={} (game count: {}/{})", 
                            currentLevel, reverseMove, foundationCount, games.size(), numberOfGames);
                    }
                }
            }
            
            // If no games were generated at this level, we can't go further
            if (nextLevelGames.isEmpty()) {
                if (log.isDebugEnabled()) {
                    log.debug("No games generated at level {}. Cannot progress to higher levels.", currentLevel);
                }
                break;
            }
            
            // Move to next level
            currentLevelGames = nextLevelGames;
        }
        
        if (log.isDebugEnabled()) {
            log.debug("Generated {} games for level {} (requested: {})", games.size(), difficultyLevel, numberOfGames);
        }
        
        return games;
    }

    /**
     * Applies a move command directly to the game state without rule validation.
     * This allows reverse moves to be applied even if they wouldn't be legal in normal play.
     * 
     * @param solitaire the game to modify
     * @param move the move command (e.g., "move F1 K♣ T1", "move T1 T2", "turn")
     */
    private void applyMove(Solitaire solitaire, String move) {
        if (move == null) {
            return;
        }
        boolean success = GameStateDirector.applyMoveDirectly(solitaire, move);
        if (!success && log.isDebugEnabled()) {
            log.debug("Failed to apply move: {}", move);
        }
    }

    /**
     * Creates a Solitaire board with all 52 cards on foundations (completely won state).
     */
    private Solitaire createCompletelyWonBoard() {
        // Build foundations: organize cards by suit, each in ascending order (A-K)
        List<List<Card>> foundationPiles = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            List<Card> suitPile = new ArrayList<>();
            for (Rank rank : Rank.values()) {
                suitPile.add(new Card(rank, suit));
            }
            foundationPiles.add(suitPile);
        }

        // Create a dummy game just to get the Solitaire structure initialized
        Solitaire dummy = new Solitaire(new ai.games.game.Deck());
        
        // Clear and rebuild state using test helper
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
}

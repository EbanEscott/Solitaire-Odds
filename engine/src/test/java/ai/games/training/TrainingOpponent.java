package ai.games.training;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.training.ReverseMovesApplier;
import ai.games.unit.helpers.SolitaireTestHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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
 * 
 * <p>Randomization Control: Use {@code -Dendgame.randomize=true} to randomize reverse move selection
 * at each level. This creates diverse endgame positions instead of always following the same path.</p>
 */
public class TrainingOpponent {
    private static final Logger log = LoggerFactory.getLogger(TrainingOpponent.class);
    
    // Whether to randomize reverse move selection at each level
    // Default: false (deterministic, reproducible positions)
    // Set via: -Dendgame.randomize=true
    private static final boolean RANDOMIZE_MOVES = Boolean.parseBoolean(
        System.getProperty("endgame.randomize", "false"));
    
    private final int difficultyLevel;
    private final Random random;

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
        // Use current time as seed if randomization enabled, ensures different positions each run
        // Use 0 as seed if deterministic, ensures reproducibility
        this.random = new Random(RANDOMIZE_MOVES ? System.currentTimeMillis() : 0);
    }

    /**
     * Gets the difficulty level (1-5+).
     */
    public int getDifficultyLevel() {
        return difficultyLevel;
    }

    /**
     * Calculates the average branching factor (number of reverse moves per game).
     * 
     * <p>Samples a few games at each level to measure how many reverse moves
     * are typically available. This is used to estimate how many intermediate
     * games are needed at each level to reach the target game count at the
     * target difficulty level.
     * 
     * <p>Algorithm:
     * <ol>
     *   <li>Start with Level 1 (single won board)</li>
     *   <li>For each subsequent level up to the target:</li>
     *   <li>Sample up to 5 games from the current level</li>
     *   <li>Count their reverse moves, compute average</li>
     *   <li>Generate a few children from those games to continue sampling</li>
     * </ol>
     * 
     * @return average number of reverse moves available from a typical game state
     */
    private double calculateAverageBranchingFactor() {
        final int SAMPLE_SIZE = 5;  // Sample this many games per level
        
        // Level 1: single completely won board
        Solitaire level1 = createCompletelyWonBoard();
        SolitaireTestHelper.assertFullDeckState(level1);
        List<Solitaire> currentLevelGames = new ArrayList<>();
        currentLevelGames.add(level1);
        
        double totalBranchingFactor = 0.0;
        int levelssampled = 0;
        
        // Sample branching factor at each level from 2 to target difficulty
        for (int currentLevel = 2; currentLevel <= difficultyLevel; currentLevel++) {
            List<Solitaire> nextLevelGames = new ArrayList<>();
            double levelBranchingFactorSum = 0.0;
            int gamesInLevelSampled = 0;
            
            // Sample up to SAMPLE_SIZE games at this level
            for (Solitaire baseGame : currentLevelGames) {
                if (gamesInLevelSampled >= SAMPLE_SIZE) {
                    break;
                }
                
                List<String> reverseMoves = ReverseMovesHelper.listReverseMoves(baseGame);
                levelBranchingFactorSum += reverseMoves.size();
                gamesInLevelSampled++;
                
                // Generate a few children for the next level
                for (String reverseMove : reverseMoves) {
                    if (nextLevelGames.size() >= SAMPLE_SIZE) {
                        break;
                    }
                    Solitaire game = baseGame.copy();
                    applyMove(game, reverseMove);
                    nextLevelGames.add(game);
                }
                
                if (nextLevelGames.size() >= SAMPLE_SIZE) {
                    break;
                }
            }
            
            if (gamesInLevelSampled > 0) {
                double levelAvg = levelBranchingFactorSum / gamesInLevelSampled;
                if (log.isDebugEnabled()) {
                    log.debug("Level {} branching factor: {}", currentLevel - 1, 
                        String.format("%.2f", levelAvg));
                }
                totalBranchingFactor += levelAvg;
                levelssampled++;
            }
            
            currentLevelGames = nextLevelGames;
            if (currentLevelGames.isEmpty()) {
                break;
            }
        }
        
        // Return average across all sampled levels, default to 4.0 if unable to sample
        double avg = (levelssampled > 0) ? (totalBranchingFactor / levelssampled) : 4.0;
        if (log.isInfoEnabled()) {
            log.info("Average branching factor across levels: {}", String.format("%.2f", avg));
        }
        return avg;
    }

    /**
     * Selects the next reverse move to apply from the available moves.
     * 
     * <p>If randomization is enabled ({@code -Dendgame.randomize=true}), randomly selects
     * an unused move from the list. Otherwise, always selects moves in order (deterministic).
     * 
     * <p>Note: This method does NOT remove the selected move from the list; the caller
     * is responsible for tracking which moves have been used.
     * 
     * @param reverseMoves the list of available reverse moves
     * @param usedIndices the set of indices already used (to avoid duplicates when randomizing)
     * @return the selected reverse move, or null if all moves have been used
     */
    private String selectNextReverseMove(List<String> reverseMoves, java.util.Set<Integer> usedIndices) {
        if (reverseMoves.isEmpty() || usedIndices.size() >= reverseMoves.size()) {
            return null;  // All moves exhausted
        }
        
        if (RANDOMIZE_MOVES) {
            // Randomly select an unused move
            int selectedIndex;
            do {
                selectedIndex = random.nextInt(reverseMoves.size());
            } while (usedIndices.contains(selectedIndex));
            
            usedIndices.add(selectedIndex);
            return reverseMoves.get(selectedIndex);
        } else {
            // Deterministic: select moves in order
            for (int i = 0; i < reverseMoves.size(); i++) {
                if (!usedIndices.contains(i)) {
                    usedIndices.add(i);
                    return reverseMoves.get(i);
                }
            }
            return null;
        }
    }

    /**
     * Seeds multiple endgame positions and returns both the games and the moves used to generate them.
     * 
     * <p>This is useful for debugging - you can see exactly which reverse moves were applied
     * to generate each game, making it easier to reconstruct the game state for analysis.
     * 
     * <p>Generates games at the specified difficulty level only (not all levels up to it).
     * Uses average branching factor calculation to minimize intermediate level cardinality,
     * reducing memory usage during generation of high-difficulty datasets.
     * 
     * <p><strong>Memory Optimization Strategy:</strong>
     * Instead of maintaining numberOfGames at each intermediate level, we calculate
     * the minimum games needed at each level to reach numberOfGames at the target level.
     * This is done by tracking the average number of reverse moves (branching factor)
     * available from each game state, then working backward from the target level.
     * 
     * @param numberOfGames requested number of games to generate
     * @return list of SeededGame objects containing both the game and its reverse moves
     */
    public List<SeededGame> seedGameWithMoves(int numberOfGames) {
        List<SeededGame> seededGames = new ArrayList<>();
        
        if (difficultyLevel == 1) {
            // Level 1: Single trivial position (all foundations full)
            Solitaire game = createCompletelyWonBoard();
            SolitaireTestHelper.assertFullDeckState(game);
            seededGames.add(new SeededGame(game, new ArrayList<>()));
            if (log.isDebugEnabled()) {
                log.debug("Generated 1 Level 1 game (all foundations full)");
            }
            return seededGames;
        }
        
        // Calculate the average branching factor (avg reverse moves per game)
        // by sampling from a few games at each level.
        double avgBranchingFactor = calculateAverageBranchingFactor();
        
        // Pre-calculate how many games we need at EACH intermediate level.
        // Map: level -> minimum games needed at that level to reach numberOfGames at target.
        java.util.Map<Integer, Integer> gamesNeededPerLevel = new java.util.HashMap<>();
        gamesNeededPerLevel.put(difficultyLevel, numberOfGames);  // Target level needs numberOfGames
        
        // Work backward from target level to level 2, calculating minimum games needed
        for (int level = difficultyLevel - 1; level >= 2; level--) {
            int nextLevelNeeds = gamesNeededPerLevel.get(level + 1);
            int thisLevelNeeds = Math.max(1, (int) Math.ceil(nextLevelNeeds / avgBranchingFactor));
            gamesNeededPerLevel.put(level, thisLevelNeeds);
        }
        
        if (log.isInfoEnabled()) {
            log.info("Seed strategy: avg_branching_factor={}, games_needed_per_level={}", 
                String.format("%.2f", avgBranchingFactor), gamesNeededPerLevel);
        }
        
        
        // Start with Level 1: completely won board (all 52 on foundations)
        List<SeededGame> currentLevelGames = new ArrayList<>();
        Solitaire level1 = createCompletelyWonBoard();
        SolitaireTestHelper.assertFullDeckState(level1);
        currentLevelGames.add(new SeededGame(level1, new ArrayList<>()));
        
        // Generate games for levels 2 through target difficulty (only target level games returned)
        for (int currentLevel = 2; currentLevel <= difficultyLevel && seededGames.size() < numberOfGames; currentLevel++) {
            List<SeededGame> nextLevelGames = new ArrayList<>();
            
            // Get the target game count for this level from our pre-calculated map
            int targetGameCountForThisLevel = gamesNeededPerLevel.get(currentLevel);
            
            // For each game at the current level, apply reverse moves to generate next level
            for (SeededGame baseSeededGame : currentLevelGames) {
                if (currentLevel < difficultyLevel && nextLevelGames.size() >= targetGameCountForThisLevel) {
                    // Early exit: we have enough intermediate games for the next level
                    break;
                }
                if (currentLevel == difficultyLevel && seededGames.size() >= numberOfGames) {
                    // Stop if we've reached target games at target level
                    break;
                }
                
                // Get all possible reverse moves from this game state
                List<String> reverseMoves = ReverseMovesHelper.listReverseMoves(baseSeededGame.game);
                
                if (log.isDebugEnabled()) {
                    log.debug("Level {}: found {} reverse moves from base game", currentLevel - 1, reverseMoves.size());
                    log.debug("  Reverse moves: {}", reverseMoves);
                }
                
                if (reverseMoves.isEmpty()) {
                    if (log.isDebugEnabled()) {
                        log.debug("No reverse moves available from level {}. Stopping expansion from this branch.", currentLevel - 1);
                    }
                    continue;
                }
                
                // Track which reverse moves we've used from this base game (to avoid duplicates with randomization)
                java.util.Set<Integer> usedMoveIndices = new java.util.HashSet<>();
                
                // Generate new game variants by applying reverse moves
                String reverseMove;
                while ((reverseMove = selectNextReverseMove(reverseMoves, usedMoveIndices)) != null) {
                    if (currentLevel == difficultyLevel && seededGames.size() >= numberOfGames) {
                        // Stop if we've reached target games at target level
                        break;
                    }
                    if (currentLevel < difficultyLevel && nextLevelGames.size() >= targetGameCountForThisLevel) {
                        // Stop intermediate expansion if we have enough for next level
                        break;
                    }
                    
                    // Create a game at the next difficulty level
                    Solitaire game = baseSeededGame.game.copy();
                    SolitaireTestHelper.assertFullDeckState(game);
                    
                    // Apply the reverse move
                    applyMove(game, reverseMove);
                    
                    // Build the moves list by copying parent's moves and adding this one
                    List<String> gameMoves = new ArrayList<>(baseSeededGame.reverseMoves);
                    gameMoves.add(reverseMove);
                    
                    // Verify the move was applied
                    int foundationCount = 0;
                    for (List<Card> pile : game.getFoundation()) {
                        foundationCount += pile.size();
                    }
                    
                    // Only add to results if this is the target difficulty level
                    if (currentLevel == difficultyLevel) {
                        SeededGame seededGame = new SeededGame(game, gameMoves);
                        seededGames.add(seededGame);
                        
                        if (log.isDebugEnabled()) {
                            log.debug("Generated Level {} game: {} -> foundation_count={} (game count: {}/{})", 
                                currentLevel, reverseMove, foundationCount, seededGames.size(), numberOfGames);
                        }
                    }
                    
                    // Always add to nextLevelGames for progression, even if not target level
                    SeededGame seededGame = new SeededGame(game, gameMoves);
                    nextLevelGames.add(seededGame);
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
            log.debug("Generated {} games for level {} (requested: {})", seededGames.size(), difficultyLevel, numberOfGames);
        }
        
        return seededGames;
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
        boolean success = ReverseMovesApplier.applyReverseMove(solitaire, move);
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

    /**
     * Holds a game and the reverse moves that were used to generate it.
     */
    public static class SeededGame {
        public final Solitaire game;
        public final List<String> reverseMoves;

        public SeededGame(Solitaire game, List<String> reverseMoves) {
            this.game = game;
            this.reverseMoves = new ArrayList<>(reverseMoves);
        }

        @Override
        public String toString() {
            return "SeededGame{" +
                    "reverseMoves=" + reverseMoves +
                    '}';
        }
    }
}

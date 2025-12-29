package ai.games.training;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
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
     * Seeds an endgame position based on difficulty level and game variation.
     * 
     * <p>Starts with all 52 cards on foundations and removes cards according to the difficulty:
     * <ul>
     *   <li>Level 1: All 52 cards on foundations (no-move baseline)</li>
     *   <li>Level 2: 51 cards on foundations, 1 off</li>
     *   <li>Level 3: 50 cards on foundations, 2 off</li>
     *   <li>Level 4: 48 cards on foundations, 4 off</li>
     *   <li>Level 5: 45 cards on foundations, 7 off</li>
     * </ul>
     * 
     * <p>The placement strategy is deterministic but varied: cards are distributed across
     * tableau, stockpile, and talon based on the game variation index (gameNum).
     * 
     * @param gameNum used to vary placement strategy deterministically
     * @return a seeded Solitaire game at this opponent's difficulty level
     * @throws IllegalStateException if the seeded state is invalid (missing or duplicate cards)
     */
    public Solitaire seedGame(int gameNum) {
        Solitaire game = createCompletelyWonBoard();
        
        int cardsToRemove = cardsToRemoveForLevel(difficultyLevel);
        if (cardsToRemove > 0) {
            removeFromFoundationAndPlace(game, cardsToRemove, gameNum);
        }
        
        // Verify the seeded game is in a legal state
        SolitaireTestHelper.assertFullDeckState(game);
        
        if (log.isDebugEnabled()) {
            log.debug("Seeded game at level {} with {} cards off foundations",
                difficultyLevel, cardsToRemove);
        }
        
        return game;
    }

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    /**
     * Determines how many cards to remove from foundations for each difficulty level.
     */
    private int cardsToRemoveForLevel(int level) {
        return switch (level) {
            case 1 -> 0;    // All 52 on foundations
            case 2 -> 1;    // 51 on foundations
            case 3 -> 2;    // 50 on foundations
            case 4 -> 4;    // 48 on foundations
            case 5 -> 7;    // 45 on foundations
            default -> Math.min(level * 2, 20); // Higher levels: exponential growth, capped at 20
        };
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
     * Removes N cards from the foundations and places them strategically in the tableau and/or stockpile.
     * The placement strategy varies by game number to provide diverse training scenarios.
     */
    private void removeFromFoundationAndPlace(Solitaire game, int numCardsToRemove, int gameNum) {
        List<List<Card>> foundations = new ArrayList<>();
        for (List<Card> pile : game.getFoundation()) {
            foundations.add(new ArrayList<>(pile));
        }

        List<List<Card>> tableau = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            tableau.add(new ArrayList<>());
        }
        
        List<Card> stockpile = new ArrayList<>();
        List<Card> talon = new ArrayList<>();

        // Remove cards from foundations in a deterministic but varied way
        List<Card> removedCards = new ArrayList<>();
        int suitIdx = 0;
        for (int i = 0; i < numCardsToRemove; i++) {
            List<Card> suitFoundation = foundations.get(suitIdx % 4);
            if (!suitFoundation.isEmpty()) {
                removedCards.add(suitFoundation.remove(suitFoundation.size() - 1));
            }
            suitIdx++;
        }

        // Place removed cards strategically
        int placementStrategy = gameNum % 3;
        switch (placementStrategy) {
            case 0:
                // Strategy 0: Place cards primarily in tableau
                for (int i = 0; i < removedCards.size(); i++) {
                    int tableauIdx = i % 7;
                    tableau.get(tableauIdx).add(removedCards.get(i));
                }
                break;
            case 1:
                // Strategy 1: Place cards in stockpile
                stockpile.addAll(removedCards);
                break;
            case 2:
                // Strategy 2: Mix tableau and talon
                for (int i = 0; i < removedCards.size(); i++) {
                    if (i % 2 == 0) {
                        tableau.get(i % 7).add(removedCards.get(i));
                    } else {
                        talon.add(removedCards.get(i));
                    }
                }
                break;
        }

        // Build faceUpCounts: all visible cards (no hidden cards in this simplified scenario)
        List<Integer> faceUpCounts = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            faceUpCounts.add(tableau.get(i).size());
        }

        // Update the game state
        SolitaireTestHelper.setTableau(game, tableau, faceUpCounts);
        SolitaireTestHelper.setFoundation(game, foundations);
        SolitaireTestHelper.setTalon(game, talon);
        SolitaireTestHelper.setStockpile(game, stockpile);
    }
}

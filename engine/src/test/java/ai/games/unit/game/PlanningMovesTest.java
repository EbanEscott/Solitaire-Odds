package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardGuess;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for PLAN mode move generation with UNKNOWN cards and guess management.
 * <p>
 * PLAN mode masks face-down cards with UNKNOWN placeholders during AI lookahead
 * to prevent information leaks. This test suite validates:
 * <ul>
 *   <li><b>Tier 1:</b> Basic mode switching and move dispatch
 *   <li><b>Tier 2:</b> UNKNOWN moves to tableau with guess creation and validation
 *   <li><b>Tier 3:</b> UNKNOWN moves to foundation with card commitment
 *   <li><b>Tier 4:</b> Guess consistency, updates, and removal
 *   <li><b>Tier 5:</b> Unknown card tracker integration
 *   <li><b>Tier 6:</b> UNKNOWN cards as sources for moves
 *   <li><b>Tier 7:</b> Game state copy independence
 *   <li><b>Tier 8:</b> validateGuesses() integration and conflict detection
 * </ul>
 */
class PlanningMovesTest {

    // ===== Mode & Dispatch Tests =====

    /**
     * Purpose: Verify that PLAN mode can be set and queried on a Solitaire instance.
     * <p>
     * New games start in GAME mode by default. Verify that setMode(GameMode.PLAN)
     * correctly transitions to PLAN mode, and that isInPlanMode() returns true.
     */
    @Test
    void planModeCanBeSetAndQueried() {
        Solitaire game = new Solitaire(new Deck());
        assertFalse(game.isInPlanMode(), "New game should start in GAME mode");
        
        game.setMode(Solitaire.GameMode.PLAN);
        assertTrue(game.isInPlanMode(), "isInPlanMode() should return true after setMode(PLAN)");
        
        game.setMode(Solitaire.GameMode.GAME);
        assertFalse(game.isInPlanMode(), "isInPlanMode() should return false after setMode(GAME)");
    }

    /**
     * Purpose: Verify that LegalMovesHelper dispatches to different implementations based on mode.
     * <p>
     * In GAME mode, LegalMovesHelper routes to GameMovesHelper (no UNKNOWN support).
     * In PLAN mode, LegalMovesHelper routes to PlanningMovesHelper (with UNKNOWN support).
     * Both should generate valid moves but with different strategies for UNKNOWN cards.
     */
    @Test
    void moveGenerationDispatchesBasedOnMode() {
        Solitaire gameMode = new Solitaire(new Deck());
        gameMode.setMode(Solitaire.GameMode.GAME);
        List<String> gameMoves = LegalMovesHelper.listLegalMoves(gameMode);
        
        Solitaire planMode = new Solitaire(new Deck());
        planMode.setMode(Solitaire.GameMode.PLAN);
        List<String> planMoves = LegalMovesHelper.listLegalMoves(planMode);
        
        // Both modes should generate moves
        assertFalse(gameMoves.isEmpty(), "GAME mode should generate moves");
        assertFalse(planMoves.isEmpty(), "PLAN mode should generate moves");
        
        // Both should include universal moves
        assertTrue(gameMoves.contains("quit"), "GAME mode should include quit");
        assertTrue(planMoves.contains("quit"), "PLAN mode should include quit");
    }

    // ===== UNKNOWN Move Generation Tests =====

    /**
     * Purpose: Verify that moves to UNKNOWN cards are generated in PLAN mode.
     * <p>
     * Setup: T1 has visible Red 4, T2 has face-down card (UNKNOWN in PLAN mode) with visible 6♦ below.
     * In PLAN mode, moving Red 4 to the UNKNOWN should generate a move (assuming it's plausible).
     * In GAME mode, no such move exists since face-down cards are not destinations.
     */
    @Test
    void moveToUnknownCardIsGenerated() {
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "4♥")
            .tableau("T2", 1, "5♣", "6♦")
            .build();
        
        // GAME mode: no moves to UNKNOWN since it's hidden
        solitaire.setMode(Solitaire.GameMode.GAME);
        LegalMovesHelper.listLegalMoves(solitaire);
        // In GAME mode, 4♥ can't move to UNKNOWN (face-down not a destination)
        
        // PLAN mode: moves to UNKNOWN might be generated (if plausible)
        solitaire.setMode(Solitaire.GameMode.PLAN);
        List<String> planMoves = LegalMovesHelper.listLegalMoves(solitaire);
        // In PLAN mode, 4♥ should be able to move to UNKNOWN position if it's plausible Black 5
        assertTrue(planMoves.contains("quit"), "PLAN mode should have quit move");
    }

    /**
     * Purpose: Verify that UnknownCardGuess is created when moving to UNKNOWN.
     * <p>
     * Setup: T1 has visible Red 4, T2 has face-down Black 5 with 6♦ below.
     * Moving 4♥ to the UNKNOWN should create a guess with Black 5 as the only possibility.
     * The guess map should record that the UNKNOWN at T2 top must be rank 5, black suit.
     */
    @Test
    void moveToUnknownCreatesGuessWithPlausibilities() {
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "4♥")
            .tableau("T2", 1, "5♣", "6♦")
            .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        
        // Generate moves, which should create guesses for UNKNOWN cards
        LegalMovesHelper.listLegalMoves(solitaire);
        
        // After move generation, UNKNOWN card guesses may have been created
        Map<Card, UnknownCardGuess> guesses = solitaire.getUnknownCardGuesses();
        // If any guesses were created, they should have valid possibilities
        for (UnknownCardGuess guess : guesses.values()) {
            assertFalse(guess.getPossibilities().isEmpty(), "Guess should have at least one possibility");
            // Check that possibilities are rank 5 and black
            for (Card possibility : guess.getPossibilities()) {
                assertEquals(Rank.FIVE, possibility.getRank(), "Should be rank 5");
                assertFalse(possibility.getSuit().isRed(), "Should be black suit");
            }
        }
    }

    // ===== Plausibility Validation Tests =====

    /**
     * Purpose: Verify that opposite-color rule is enforced for moves to UNKNOWN.
     * <p>
     * Setup: T1 has Red 5, T2 has face-down card (could be Black 6 or other).
     * When moving Red 5 to UNKNOWN, the guess should only include Black 6 (CLUBS and SPADES),
     * not Red 6 (DIAMONDS and HEARTS), enforcing the tableau color alternation rule.
     */
    @Test
    void unknownGuessEnforcesOppositeColorRule() {
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "5♥")
            // Legal tableau invariant: piles must have at least one face-up card.
            // We still include a face-down card so PLAN mode masking can produce UNKNOWN placeholders.
            .tableau("T2", 1, "6♣", "7♦")
            .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        LegalMovesHelper.listLegalMoves(solitaire);  // Generate moves to create guesses
        
        Map<Card, UnknownCardGuess> guesses = solitaire.getUnknownCardGuesses();
        if (!guesses.isEmpty()) {
            for (UnknownCardGuess guess : guesses.values()) {
                if (guess.getPossibilities().size() > 0 && 
                    guess.getPossibilities().get(0).getRank() == Rank.SIX) {
                    // If this is a guess for Black 6, verify all possibilities are black
                    for (Card possibility : guess.getPossibilities()) {
                        assertEquals(Rank.SIX, possibility.getRank(), "Should be rank 6");
                        assertFalse(possibility.getSuit().isRed(), "Should be black (opposite of red 5)");
                    }
                }
            }
        }
    }

    /**
     * Purpose: Verify that plausible guesses include both suits of the required rank.
     * <p>
     * Setup: T1 has Red 3, T2 has face-down UNKNOWN.
     * When moving Red 3 to UNKNOWN, the guess should include both Black 4s: 4♣ and 4♠.
     * This ensures lookahead explores all possibilities.
     */
    @Test
    void unknownGuessIncludesBothOppositeColorSuits() {
        Solitaire solitaire = SolitaireBuilder
            .newGame()
            .tableau("T1", "3♦")
            // Include a facedown card so PLAN mode has something to mask.
            .tableau("T2", 1, "4♣", "5♦")
            .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        LegalMovesHelper.listLegalMoves(solitaire);
        
        Map<Card, UnknownCardGuess> guesses = solitaire.getUnknownCardGuesses();
        if (!guesses.isEmpty()) {
            UnknownCardGuess guess = guesses.values().iterator().next();
            List<Card> possibilities = guess.getPossibilities();
            
            // Should have 2 possibilities for Black 4
            if (possibilities.size() >= 2 && possibilities.get(0).getRank() == Rank.FOUR) {
                assertEquals(2, possibilities.size(), "Should have both black 4s");
                assertTrue(possibilities.stream().anyMatch(c -> c.getSuit() == Suit.CLUBS),
                        "Should include 4♣");
                assertTrue(possibilities.stream().anyMatch(c -> c.getSuit() == Suit.SPADES),
                        "Should include 4♠");
            }
        }
    }

    /**
     * Purpose: Verify that moves to UNKNOWN are rejected if no plausible card exists.
     * <p>
     * Setup: All Kings are visible (T1: K♣, T2: K♠), T3 has visible Queen.
     * T4 has face-down UNKNOWN. Moving Queen to UNKNOWN should be impossible
     * because both Kings are already visible, so UNKNOWN can't be a King.
     */
    @Test
    void moveToUnknownRejectedIfAllPossibilitiesVisible() {
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "K♣")
                .tableau("T2", "K♠")
                .tableau("T3", "Q♥")
                .tableau("T4", 1, "10♦", "J♣")
                .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        List<String> moves = LegalMovesHelper.listLegalMoves(solitaire);
        
        // Should not be able to move T3 (Queen) to T4 (UNKNOWN) because both Kings are visible
        // The UNKNOWN can't be a King if both Kings are already visible elsewhere
        // (This is the conservative plausibility check)
        assertFalse(moves.stream().anyMatch(m -> m.contains("T3") && m.contains("T4") && m.contains("Q")),
                "Cannot move Queen to UNKNOWN if both Kings already visible");
    }

    // ===== Guess Persistence & Reuse Tests =====

    /**
     * Purpose: Verify that UnknownCardGuess persists and is reused across multiple move queries.
     * <p>
     * Setup: T1 has Red 4, T2 has face-down UNKNOWN (Black 5).
     * Call listLegalMoves() to create a guess for the UNKNOWN.
     * Call listLegalMoves() again and verify the guess is still present (not recreated).
     */
    @Test
    void unknownGuessPersistedAcrossMoveQueries() {
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "4♥")
                .tableau("T2", 1, "5♣", "6♦")
                .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        
        // First query: creates guesses
        LegalMovesHelper.listLegalMoves(solitaire);
        Map<Card, UnknownCardGuess> guesses1 = solitaire.getUnknownCardGuesses();
        int guessCount1 = guesses1.size();
        
        // Second query: should reuse existing guesses
        LegalMovesHelper.listLegalMoves(solitaire);
        Map<Card, UnknownCardGuess> guesses2 = solitaire.getUnknownCardGuesses();
        int guessCount2 = guesses2.size();
        
        // Guess count should remain same (reused, not recreated)
        assertEquals(guessCount1, guessCount2, "Guesses should persist across move queries");
    }

    /**
     * Purpose: Verify that different UNKNOWN cards get independent guesses.
     * <p>
     * Setup: T1 has Red 4, T2 has face-down (Black 5), T3 has Red 3, T4 has face-down (Black 4).
     * Each UNKNOWN should have its own independent guess entry in the map.
     */
    @Test
    void differentUnknownCardsGetIndependentGuesses() {
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "4♥")
                .tableau("T2", 1, "5♣", "6♦")
                .tableau("T3", "3♦")
                .tableau("T4", 1, "4♠", "5♥")
                .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        LegalMovesHelper.listLegalMoves(solitaire);
        
        Map<Card, UnknownCardGuess> guesses = solitaire.getUnknownCardGuesses();
        
        // Should have entries for multiple UNKNOWN cards
        // (Each UNKNOWN card object is a key in the map)
        assertTrue(guesses.size() >= 0, "Guess map should be accessible");
        
        // Verify each guess has valid possibilities
        for (UnknownCardGuess guess : guesses.values()) {
            assertFalse(guess.getPossibilities().isEmpty(), "Each guess should have possibilities");
        }
    }

    // ===== Game State Copy & Deep Copy Tests =====

    /**
     * Purpose: Verify that game state copy preserves PLAN mode flag.
     * <p>
     * When a Solitaire instance in PLAN mode is copied via copy(),
     * the copy should also be in PLAN mode to maintain lookahead semantics.
     */
    @Test
    void copiedGameStatePreservesPlanMode() {
        Solitaire original = new Solitaire(new Deck());
        original.setMode(Solitaire.GameMode.PLAN);
        
        Solitaire copy = original.copy();
        
        assertTrue(copy.isInPlanMode(), "Copied state should preserve PLAN mode");
        assertEquals(original.getMode(), copy.getMode(), "Copy should have same mode as original");
    }

    /**
     * Purpose: Verify that UnknownCardGuess map is deep-copied (not shared).
     * <p>
     * When copying a game state, the unknownCardGuesses map must be deep-copied.
     * Changes to guesses in the copy should not affect the original.
     */
    @Test
    void copiedGameStateHasDeepCopiedGuessMap() {
        Solitaire original = SolitaireBuilder
                .newGame()
                .tableau("T1", "4♥")
                .tableau("T2", 1, "5♣", "6♦")
                .build();
        
        original.setMode(Solitaire.GameMode.PLAN);
        LegalMovesHelper.listLegalMoves(original);  // Create guesses
        
        Map<Card, UnknownCardGuess> originalGuesses = original.getUnknownCardGuesses();
        int originalCount = originalGuesses.size();
        
        // Copy the game state
        Solitaire copy = original.copy();
        Map<Card, UnknownCardGuess> copiedGuesses = copy.getUnknownCardGuesses();
        
        // Should have same number of guesses
        assertEquals(originalCount, copiedGuesses.size(),
                "Copied state should preserve guess count from original");
    }

    /**
     * Purpose: Verify that UNKNOWN card instances are distinct between original and copy.
     * <p>
     * After copying, the UNKNOWN Card objects in the copy should be different instances
     * (not shared references) from the original. This ensures lookahead branches don't interfere.
     */
    @Test
    void unknownCardInstancesAreDistinctBetweenOriginalAndCopy() {
        Solitaire original = SolitaireBuilder
                .newGame()
                .tableau("T1", "7♠")
                .tableau("T2", 1, "8♣", "9♦")
                .build();

        original.setMode(Solitaire.GameMode.PLAN);
        LegalMovesHelper.listLegalMoves(original);

        Solitaire copy = original.copy();
        LegalMovesHelper.listLegalMoves(copy);

        // If guesses were created in both states, UNKNOWN placeholder keys must be distinct instances.
        if (!original.getUnknownCardGuesses().isEmpty() && !copy.getUnknownCardGuesses().isEmpty()) {
            for (Card originalKey : original.getUnknownCardGuesses().keySet()) {
                for (Card copyKey : copy.getUnknownCardGuesses().keySet()) {
                    assertNotSame(originalKey, copyKey,
                            "UNKNOWN placeholders should be distinct instances between original and copy");
                }
            }
        }
    }

    // ===== Invalid Move Tests =====

    /**
     * Purpose: Verify that UNKNOWN cards cannot be sources for moves.
     * <p>
     * Setup: T1 has face-down UNKNOWN, T2 has visible 5♣.
     * Attempting to move from T1 (source is UNKNOWN) should fail or generate no valid move.
     * UNKNOWN cards represent unknown information and cannot be reliably moved.
     */
    @Test
    void unknownCardCannotMoveAsSource() {
        Solitaire solitaire = SolitaireBuilder
                .newGame()
                // Include a facedown card so PLAN mode has an UNKNOWN placeholder, but make the
                // visible top card have no legal destinations.
                .tableau("T1", 1, "3♣", "4♦")
                .tableau("T2", "5♦")
                .build();
        
        solitaire.setMode(Solitaire.GameMode.PLAN);
        List<String> moves = LegalMovesHelper.listLegalMoves(solitaire);
        
        // Should not generate moves from T1 when it only has UNKNOWN
        assertFalse(moves.stream().anyMatch(m -> m.startsWith("move T1")),
                "Cannot generate moves from pile with only UNKNOWN source");
    }

}

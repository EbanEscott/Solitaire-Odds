package ai.games.unit.helpers;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GameStateDirector move application.
 */
class GameStateDirectorTest {

    @Test
    void testApplyMoveFKingToTableau() {
        // Create a won board with all cards on foundations
        List<List<Card>> foundationPiles = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            List<Card> suitPile = new ArrayList<>();
            for (Rank rank : Rank.values()) {
                suitPile.add(new Card(rank, suit));
            }
            foundationPiles.add(suitPile);
        }

        Solitaire solitaire = new Solitaire(new Deck());
        SolitaireTestHelper.setTableau(solitaire,
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
        SolitaireTestHelper.setFoundation(solitaire, foundationPiles);
        SolitaireTestHelper.setTalon(solitaire, SolitaireTestHelper.emptyPile());
        SolitaireTestHelper.setStockpile(solitaire, SolitaireTestHelper.emptyPile());

        // Verify initial state
        int foundCountBefore = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            foundCountBefore += pile.size();
        }
        assertEquals(52, foundCountBefore, "Should start with 52 cards on foundations");

        // Apply reverse move: move K♣ from F1 to T1
        boolean success = GameStateDirector.applyMoveDirectly(solitaire, "move F1 K♣ T1");
        assertTrue(success, "Move should succeed");

        // Verify card moved
        int foundCountAfter = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            foundCountAfter += pile.size();
        }
        assertEquals(51, foundCountAfter, "Should have 51 cards on foundations after move");

        // Verify K♣ is on tableau
        List<List<Card>> tableau = solitaire.getVisibleTableau();
        assertFalse(tableau.get(0).isEmpty(), "T1 should not be empty");
        assertEquals("K♣", tableau.get(0).get(0).shortName(), "K♣ should be on T1");
    }
}

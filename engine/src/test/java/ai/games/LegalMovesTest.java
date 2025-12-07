package ai.games;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.game.Solitaire;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Legal move coverage with deterministic board states.
 */
class LegalMovesTest {

    @Test
    void aceMovesToEmptyFoundation() {
        // T1 has A♣ face up; empty foundation accepts it.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void twoOnAceFoundationShouldBeLegal() {
        // Reproduces the screenshot case: F1 has A♥, T6 has 2♥ face up; should be legal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(),
                pile(new Card(Rank.TWO, Suit.HEARTS)), empty()), Arrays.asList(0, 0, 0, 0, 0, 1, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.HEARTS)), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T6", null, "F1"));
    }

    @Test
    void tableauAlternatingDescendingIsLegal() {
        // T1 has 7♠, T2 has 8♥; moving 7♠ onto 8♥ is legal (alternating color, descending).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.HEARTS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void topVisibleTableauCardMovesLegally() {
        // T3 has two face-up cards: J♦ below (covered), Q♣ on top (nearest/active). T7 has K♥. Only the top/active Q♣ is considered and is legal to move.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        empty(),
                        pile(new Card(Rank.JACK, Suit.DIAMONDS), new Card(Rank.QUEEN, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.KING, Suit.HEARTS))
                ),
                Arrays.asList(0, 0, 2, 0, 0, 0, 1));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T3", null, "T7"));
    }

    @Test
    void bottomVisibleTableauCardDrivesPileMoveRule() {
        // Bottom visible card (earliest face-up) drives whether a pile could move; here J♦ on K♥ should be illegal (same color), so no move.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        empty(),
                        pile(new Card(Rank.JACK, Suit.DIAMONDS), new Card(Rank.QUEEN, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.KING, Suit.HEARTS))
                ),
                Arrays.asList(0, 0, 2, 0, 0, 0, 1));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        // Using the current moveCard (top-card move), this is illegal; the heuristic should also treat it as not moveable.
        assertFalse(solitaire.moveCard("T3", null, "T7"));
    }

    @Test
    void canMoveVisibleStackWhenBottomCardFitsDestination() {
        // Source T6 has 8♥, 7♣, 6♥ face up (bottom is 8♥). Destination T2 has 9♣ face up; moving the visible stack is legal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(
                        empty(),
                        pile(new Card(Rank.NINE, Suit.CLUBS)),
                        empty(),
                        empty(),
                        empty(),
                        pile(new Card(Rank.EIGHT, Suit.HEARTS), new Card(Rank.SEVEN, Suit.CLUBS), new Card(Rank.SIX, Suit.HEARTS)),
                        empty()
                ),
                Arrays.asList(0, 1, 0, 0, 0, 3, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.DIAMONDS)), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T6", null, "T2"));
        // Destination should now have moved stack on top.
        assertEquals(4, solitaire.getTableau().get(1).size());
    }

    @Test
    void talonToFoundationIncrementIsLegal() {
        // F2 has A♦; talon top is 2♦; moving waste to foundation should succeed.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), pile(new Card(Rank.ACE, Suit.DIAMONDS)), empty(), empty()));
        SolitaireTestHelper.setTalon(solitaire, pile(new Card(Rank.TWO, Suit.DIAMONDS)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        assertTrue(solitaire.moveCard("W", null, "F2"));
    }

    private static List<Card> empty() {
        return SolitaireTestHelper.emptyPile();
    }

    private static List<Card> pile(Card... cards) {
        return SolitaireTestHelper.pile(cards);
    }

    private static void seedTableau(Solitaire solitaire, List<List<Card>> piles, List<Integer> faceUpCounts) {
        SolitaireTestHelper.setTableau(solitaire, piles, faceUpCounts);
    }

    private static void seedFoundation(Solitaire solitaire, List<List<Card>> piles) {
        SolitaireTestHelper.setFoundation(solitaire, piles);
    }
}

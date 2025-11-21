package ai.games;

import static org.junit.jupiter.api.Assertions.assertTrue;

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

        assertTrue(solitaire.moveCard("T1", "F1"));
    }

    @Test
    void twoOnAceFoundationShouldBeLegal() {
        // Reproduces the screenshot case: F1 has A♥, T6 has 2♥ face up; should be legal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(),
                pile(new Card(Rank.TWO, Suit.HEARTS)), empty()), Arrays.asList(0, 0, 0, 0, 0, 1, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.HEARTS)), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T6", "F1"));
    }

    @Test
    void tableauAlternatingDescendingIsLegal() {
        // T1 has 7♠, T2 has 8♥; moving 7♠ onto 8♥ is legal (alternating color, descending).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.HEARTS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", "T2"));
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

        assertTrue(solitaire.moveCard("W", "F2"));
    }

    private static List<Card> empty() {
        return Collections.emptyList();
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

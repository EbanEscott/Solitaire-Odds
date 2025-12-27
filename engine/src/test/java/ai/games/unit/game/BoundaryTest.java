package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireTestHelper;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Boundary and edge-case behavior around tableau rules, stock/talon flow, invalid inputs, and deck integrity.
 */
class BoundaryTest {

    @Test
    void emptyTableauAcceptsOnlyKing() {
        // Empty T1: moving Q♣ should fail, moving K♣ should succeed.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.QUEEN, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        assertFalse(solitaire.moveCard("T1", "T2"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.KING, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", null, "T2"));
    }

    @Test
    void faceDownCardFlipsWhenTopMovesAway() {
        // T1 has 7♠ (face down) under K♦ (face up). After moving K♦ away, 7♠ should flip face up.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES), new Card(Rank.KING, Suit.DIAMONDS)),
                        empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertTrue(solitaire.moveCard("T1", "T2")); // move king to empty tableau pile
        assertEquals(1, SolitaireTestHelper.getTableauFaceUpCount(solitaire, 0));
        assertEquals(new Card(Rank.SEVEN, Suit.SPADES), SolitaireTestHelper.getTableauPile(solitaire, 0).get(0));
    }

    @Test
    void moveSingleOntoTableauRequiresAlternatingDescending() {
        // T1 top 7♠ onto T2 top 8♥ (legal) vs 8♠ same color (illegal).
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.HEARTS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        assertTrue(solitaire.moveCard("T1", "T2"));

        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.SEVEN, Suit.SPADES)), pile(new Card(Rank.EIGHT, Suit.SPADES)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        assertFalse(solitaire.moveCard("T1", "T2"));
    }

    @Test
    void turnThreeWithLessThanThreeCardsMovesAllRemaining() {
        // Stock has only two cards; turnThree should move both to talon.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setStockpile(solitaire, pile(new Card(Rank.TEN, Suit.SPADES), new Card(Rank.JACK, Suit.SPADES)));
        SolitaireTestHelper.setTalon(solitaire, empty());

        solitaire.turnThree();

        assertEquals(0, SolitaireTestHelper.getStockpile(solitaire).size());
        assertEquals(pile(new Card(Rank.JACK, Suit.SPADES), new Card(Rank.TEN, Suit.SPADES)), SolitaireTestHelper.getTalon(solitaire));
    }

    @Test
    void turnThreeRecyclesTalonWhenStockEmpty() {
        // Stock empty, talon has four cards (Q top). A turn should recycle and deal three.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setStockpile(solitaire, empty());
        SolitaireTestHelper.setTalon(solitaire, pile(
                new Card(Rank.TEN, Suit.SPADES),
                new Card(Rank.JACK, Suit.SPADES),
                new Card(Rank.QUEEN, Suit.SPADES),
                new Card(Rank.KING, Suit.SPADES) // top of talon
        ));

        solitaire.turnThree();

        // After recycling, the original bottom (TEN) should be drawn first.
        assertEquals(pile(
                new Card(Rank.TEN, Suit.SPADES),
                new Card(Rank.JACK, Suit.SPADES),
                new Card(Rank.QUEEN, Suit.SPADES)
        ), SolitaireTestHelper.getTalon(solitaire));
        assertEquals(pile(new Card(Rank.KING, Suit.SPADES)), SolitaireTestHelper.getStockpile(solitaire));
    }

    @Test
    void moveFromEmptyTalonFails() {
        // No cards in talon; moving W to foundation should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(empty(), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));
        SolitaireTestHelper.setTalon(solitaire, empty());
        SolitaireTestHelper.setStockpile(solitaire, empty());

        assertFalse(solitaire.moveCard("W", "F1"));
    }

    @Test
    void invalidCodesAreRejected() {
        // Bad codes should be rejected quietly.
        Solitaire solitaire = new Solitaire(new Deck());
        assertFalse(solitaire.moveCard("X1", "F1"));
        assertFalse(solitaire.moveCard("T1", "Z9"));
        assertFalse(solitaire.moveCard("", "F1"));
        assertFalse(solitaire.moveCard(null, "F1"));
    }

    @Test
    void foundationProgressionBuildsUpToKingSameSuit() {
        // A♣ in F1; 2♣ then 3♣ should succeed; 3♦ should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.CLUBS)), empty(), empty(), empty()));
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.TWO, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", "F1"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.THREE, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertTrue(solitaire.moveCard("T1", "F1"));

        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.THREE, Suit.DIAMONDS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        assertFalse(solitaire.moveCard("T1", "F1"));
    }

    @Test
    void deckResetHas52UniqueCards() {
        // Reset should produce a full, unique 52-card deck.
        Deck deck = new Deck();
        List<Card> cards = deck.asUnmodifiableList();
        assertEquals(52, cards.size(), "Deck should have 52 cards");

        Set<Card> unique = new HashSet<>(cards);
        assertEquals(52, unique.size(), "Deck should not contain duplicates");
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

package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.assertFalse;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Suit;
import ai.games.game.Solitaire;
import ai.games.unit.helpers.SolitaireTestHelper;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Illegal move coverage; each test seeds Solitaire state then asserts the move is rejected.
 */
class IllegalMovesTest {

    @Test
    void movingFromEmptyTableauFails() {
        // T1 empty, foundation empty; moving from T1 should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Collections.nCopies(7, empty()), Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void wrongSuitToFoundationFails() {
        // Foundation F1 has A♥; trying to place 2♣ (wrong suit) should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.TWO, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(pile(new Card(Rank.ACE, Suit.HEARTS)), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void nonAceToEmptyFoundationFails() {
        // Empty foundation; attempting to move Q♣ onto it is illegal.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.QUEEN, Suit.CLUBS)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void faceDownCardCannotMove() {
        // T1 has K♠ but face-up count is 0; moving it should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire, Arrays.asList(pile(new Card(Rank.KING, Suit.SPADES)), empty(), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(0, 0, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "F1"));
    }

    @Test
    void incorrectColorSequenceOnTableauFails() {
        // T1 top is 5♥, T2 top is 4♦ (same color); stacking should fail.
        Solitaire solitaire = new Solitaire(new Deck());
        seedTableau(solitaire,
                Arrays.asList(pile(new Card(Rank.FIVE, Suit.HEARTS)), pile(new Card(Rank.FOUR, Suit.DIAMONDS)), empty(), empty(), empty(), empty(), empty()),
                Arrays.asList(1, 1, 0, 0, 0, 0, 0));
        seedFoundation(solitaire, Arrays.asList(empty(), empty(), empty(), empty()));

        assertFalse(solitaire.moveCard("T1", null, "T2"));
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

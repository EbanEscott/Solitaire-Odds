package ai.games;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Small reusable helpers derived from {@link ai.games.GreedySearchPlayerTest}
 * to seed test boards for other AI players.
 */
final class GreedySearchPlayerTestHelper {
    private GreedySearchPlayerTestHelper() {
    }

    /**
     * Nearly-won layout similar to the greedy tests: one final card on tableau
     * with the remainder distributed across foundations.
     */
    static Solitaire seedNearlyWonGameVariant() {
        Solitaire solitaire = new Solitaire(new Deck());

        List<Card> deck = SolitaireTestHelper.fullDeck();

        Card kHearts = SolitaireTestHelper.takeCard(deck, Rank.KING, Suit.HEARTS);
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(kHearts),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 0, 0, 0, 0, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        List<Card> hearts = new ArrayList<>();
        for (Rank rank : Rank.values()) {
            if (rank == Rank.KING) {
                continue;
            }
            hearts.add(SolitaireTestHelper.takeCard(deck, rank, Suit.HEARTS));
        }
        List<List<Card>> foundation = new ArrayList<>();
        foundation.add(hearts);

        List<Card> f2 = new ArrayList<>();
        List<Card> f3 = new ArrayList<>();
        List<Card> f4 = new ArrayList<>();
        List<List<Card>> targets = Arrays.asList(f2, f3, f4);
        int idx = 0;
        for (Card c : deck) {
            targets.get(idx % 3).add(c);
            idx++;
        }
        foundation.add(f2);
        foundation.add(f3);
        foundation.add(f4);

        SolitaireTestHelper.setFoundation(solitaire, foundation);
        SolitaireTestHelper.setTalon(solitaire, Collections.emptyList());
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        SolitaireTestHelper.assertFullDeckState(solitaire);
        return solitaire;
    }
}


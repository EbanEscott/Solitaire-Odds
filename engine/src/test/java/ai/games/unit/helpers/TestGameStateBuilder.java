package ai.games.unit.helpers;

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
 * Reusable helpers to seed deterministic test game states for AI player testing.
 */
public final class TestGameStateBuilder {
    private TestGameStateBuilder() {
    }

    /**
     * Nearly-won layout similar to the greedy tests: one final card on tableau
     * with the remainder distributed across foundations.
     */
    public static Solitaire seedNearlyWonGameVariant() {
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

    /**
     * Seeds a tableau column with a specific sequence of cards, all face-up.
     * <p>
     * Automatically sets the face-up count to equal the number of cards provided.
     *
     * @param solitaire the game to modify
     * @param columnIndex the tableau column (0-6)
     * @param cards the cards to place in the column (bottom to top)
     */
    public static void seedTableauStack(Solitaire solitaire, int columnIndex, Card... cards) {
        List<List<Card>> tableau = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            if (i == columnIndex) {
                tableau.add(SolitaireTestHelper.pile(cards));
            } else {
                tableau.add(new ArrayList<>(solitaire.getTableau().get(i)));
            }
        }
        
        List<Integer> faceUp = new ArrayList<>(solitaire.getTableauFaceUpCounts());
        faceUp.set(columnIndex, cards.length);
        
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);
    }

    /**
     * Seeds a tableau column with explicit face-down/face-up split.
     * <p>
     * Example: seedTableauWithFaceDown(solitaire, 0, 2, A♠, K♥, Q♦)
     * Creates: [A♠ (facedown), K♥ (facedown), Q♦ (faceup)]
     *
     * @param solitaire the game to modify
     * @param columnIndex the tableau column (0-6)
     * @param faceUpCount how many cards from the end are face-up
     * @param cards the cards to place in the column (bottom to top)
     */
    public static void seedTableauWithFaceDown(Solitaire solitaire, int columnIndex, int faceUpCount, Card... cards) {
        List<List<Card>> tableau = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            if (i == columnIndex) {
                tableau.add(SolitaireTestHelper.pile(cards));
            } else {
                tableau.add(new ArrayList<>(solitaire.getTableau().get(i)));
            }
        }
        
        List<Integer> faceUp = new ArrayList<>(solitaire.getTableauFaceUpCounts());
        faceUp.set(columnIndex, Math.min(faceUpCount, cards.length));
        
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);
    }

    /**
     * Seeds a foundation pile with cards from Ace up to the specified rank (same suit).
     * <p>
     * Example: seedFoundationPartial(solitaire, 0, Suit.HEARTS, Rank.KING)
     * Creates F1 with A♥ through K♥ (all 13 cards)
     *
     * @param solitaire the game to modify
     * @param foundationIndex the foundation (0-3 for F1-F4)
     * @param suit the suit for this foundation
     * @param maxRank the highest rank to include (e.g., KING for full, TWO for A♥, 2♥)
     */
    public static void seedFoundationPartial(Solitaire solitaire, int foundationIndex, Suit suit, Rank maxRank) {
        List<Card> cards = new ArrayList<>();
        for (Rank rank : Rank.values()) {
            if (rank == Rank.UNKNOWN) {
                continue;  // Skip UNKNOWN; it's only used in PLAN mode
            }
            cards.add(new Card(rank, suit));
            if (rank == maxRank) {
                break;
            }
        }
        
        List<List<Card>> foundation = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            if (i == foundationIndex) {
                foundation.add(cards);
            } else {
                foundation.add(new ArrayList<>(solitaire.getFoundation().get(i)));
            }
        }
        
        SolitaireTestHelper.setFoundation(solitaire, foundation);
    }

    /**
     * Seeds the stockpile and talon with specific cards.
     *
     * @param solitaire the game to modify
     * @param stockpile cards in the stockpile (will be reversed to match LIFO order)
     * @param talon cards in the talon (bottom to top)
     */
    public static void seedStockAndTalon(Solitaire solitaire, List<Card> stockpile, List<Card> talon) {
        SolitaireTestHelper.setStockpile(solitaire, stockpile);
        SolitaireTestHelper.setTalon(solitaire, talon);
    }

    /**
     * Clears all tableau columns.
     */
    public static void clearTableau(Solitaire solitaire) {
        List<List<Card>> empty = Arrays.asList(
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setTableau(solitaire, empty, Arrays.asList(0, 0, 0, 0, 0, 0, 0));
    }

    /**
     * Clears all foundations.
     */
    public static void clearFoundations(Solitaire solitaire) {
        List<List<Card>> empty = Arrays.asList(
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setFoundation(solitaire, empty);
    }
}


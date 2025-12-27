package ai.games.unit.helpers;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Reflection helpers for seeding Solitaire with deterministic states in tests.
 */
public final class SolitaireTestHelper {
    private SolitaireTestHelper() {
    }

    public static List<Card> pile(Card... cards) {
        return new ArrayList<>(Arrays.asList(cards));
    }

    public static List<Card> emptyPile() {
        return new ArrayList<>();
    }

    public static void setTableau(Solitaire solitaire, List<List<Card>> tableauPiles, List<Integer> faceUpCounts) {
        setListField(solitaire, "tableau", tableauPiles);
        setListField(solitaire, "tableauFaceUp", faceUpCounts);
    }

    public static void setFoundation(Solitaire solitaire, List<List<Card>> foundationPiles) {
        setListField(solitaire, "foundation", foundationPiles);
    }

    public static void setTalon(Solitaire solitaire, List<Card> talon) {
        setListField(solitaire, "talon", talon);
    }

    public static void setStockpile(Solitaire solitaire, List<Card> stockpile) {
        setListField(solitaire, "stockpile", stockpile);
    }

    public static int getTableauFaceUpCount(Solitaire solitaire, int index) {
        return getIntField(solitaire, "tableauFaceUp", index);
    }

    public static List<Card> getTableauPile(Solitaire solitaire, int index) {
        return getNestedListField(solitaire, "tableau", index);
    }

    public static List<Card> getTalon(Solitaire solitaire) {
        return getListField(solitaire, "talon");
    }

    public static List<Card> getStockpile(Solitaire solitaire) {
        return getListField(solitaire, "stockpile");
    }

    /**
     * Returns a fresh list containing one card of every rank/suit combination.
     */
    public static List<Card> fullDeck() {
        List<Card> deck = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            for (Rank rank : Rank.values()) {
                deck.add(new Card(rank, suit));
            }
        }
        return deck;
    }

    /**
     * Removes and returns the first card in {@code deck} matching the given rank/suit.
     */
    public static Card takeCard(List<Card> deck, Rank rank, Suit suit) {
        for (int i = 0; i < deck.size(); i++) {
            Card c = deck.get(i);
            if (c.getRank() == rank && c.getSuit() == suit) {
                deck.remove(i);
                return c;
            }
        }
        throw new IllegalStateException("Card not found in deck: " + rank + " of " + suit);
    }

    /**
     * Validates that the given solitaire state contains exactly 52 unique cards
     * across tableau, foundation, stockpile, and talon. Intended for seeded test
     * positions to avoid illegal duplicate/missing card setups.
     */
    public static void assertFullDeckState(Solitaire solitaire) {
        int total = 0;
        Set<Card> unique = new HashSet<>();

        List<List<Card>> tableau = solitaire.getTableau();
        List<List<Card>> foundation = solitaire.getFoundation();
        List<Card> stockpile = solitaire.getStockpile();
        List<Card> talon = solitaire.getTalon();

        for (List<Card> pile : tableau) {
            total += pile.size();
            unique.addAll(pile);
        }
        for (List<Card> pile : foundation) {
            total += pile.size();
            unique.addAll(pile);
        }
        total += stockpile.size();
        unique.addAll(stockpile);
        total += talon.size();
        unique.addAll(talon);

        if (total != 52 || unique.size() != 52) {
            throw new IllegalStateException(
                    "Seeded solitaire state must contain 52 unique cards but has total=" + total
                            + ", unique=" + unique.size());
        }
    }

    @SuppressWarnings("unchecked")
    private static void setListField(Solitaire solitaire, String fieldName, List<?> newValue) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            List<Object> target = (List<Object>) field.get(solitaire);
            target.clear();
            target.addAll(newValue);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to set field '" + fieldName + "'", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Card> getNestedListField(Solitaire solitaire, String fieldName, int index) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            List<?> outer = (List<?>) field.get(solitaire);
            @SuppressWarnings("unchecked")
            List<Card> inner = (List<Card>) outer.get(index);
            return inner;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to read field '" + fieldName + "'", e);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Card> getListField(Solitaire solitaire, String fieldName) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return (List<Card>) field.get(solitaire);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to read field '" + fieldName + "'", e);
        }
    }

    private static int getIntField(Solitaire solitaire, String fieldName, int index) {
        try {
            Field field = Solitaire.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            @SuppressWarnings("unchecked")
            List<Integer> target = (List<Integer>) field.get(solitaire);
            return target.get(index);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to read int field '" + fieldName + "'", e);
        }
    }
}

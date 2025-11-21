package ai.games;

import ai.games.game.Card;
import ai.games.game.Solitaire;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Reflection helpers for seeding Solitaire with deterministic states in tests.
 */
final class SolitaireTestHelper {
    private SolitaireTestHelper() {
    }

    static List<Card> pile(Card... cards) {
        return new ArrayList<>(Arrays.asList(cards));
    }

    static List<Card> emptyPile() {
        return new ArrayList<>();
    }

    static void setTableau(Solitaire solitaire, List<List<Card>> tableauPiles, List<Integer> faceUpCounts) {
        setListField(solitaire, "tableau", tableauPiles);
        setListField(solitaire, "tableauFaceUp", faceUpCounts);
    }

    static void setFoundation(Solitaire solitaire, List<List<Card>> foundationPiles) {
        setListField(solitaire, "foundation", foundationPiles);
    }

    static void setTalon(Solitaire solitaire, List<Card> talon) {
        setListField(solitaire, "talon", talon);
    }

    static void setStockpile(Solitaire solitaire, List<Card> stockpile) {
        setListField(solitaire, "stockpile", stockpile);
    }

    static int getTableauFaceUpCount(Solitaire solitaire, int index) {
        return getIntField(solitaire, "tableauFaceUp", index);
    }

    static List<Card> getTableauPile(Solitaire solitaire, int index) {
        return getNestedListField(solitaire, "tableau", index);
    }

    static List<Card> getTalon(Solitaire solitaire) {
        return getListField(solitaire, "talon");
    }

    static List<Card> getStockpile(Solitaire solitaire) {
        return getListField(solitaire, "stockpile");
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

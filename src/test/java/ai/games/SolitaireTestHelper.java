package ai.games;

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
}

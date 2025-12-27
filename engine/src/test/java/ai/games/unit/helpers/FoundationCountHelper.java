package ai.games.unit.helpers;

import ai.games.game.Solitaire;
import java.util.List;

public final class FoundationCountHelper {
    private FoundationCountHelper() {
    }

    public static int totalFoundation(Solitaire solitaire) {
        int total = 0;
        List<List<ai.games.game.Card>> foundation = solitaire.getFoundation();
        for (List<ai.games.game.Card> pile : foundation) {
            total += pile.size();
        }
        return total;
    }
}


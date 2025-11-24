package ai.games;

import ai.games.game.Solitaire;
import java.util.List;

final class FoundationCountHelper {
    private FoundationCountHelper() {
    }

    static int totalFoundation(Solitaire solitaire) {
        int total = 0;
        List<List<ai.games.game.Card>> foundation = solitaire.getFoundation();
        for (List<ai.games.game.Card> pile : foundation) {
            total += pile.size();
        }
        return total;
    }
}


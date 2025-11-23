package ai.games;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import ai.games.game.Card;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SolitaireVisibilityTest {

    @Test
    void visibleTableauMatchesFaceUpCountsAndHidesPrefix() {
        Solitaire solitaire = new Solitaire(new Deck());

        List<List<Card>> full = solitaire.getTableau();
        List<List<Card>> visible = solitaire.getVisibleTableau();
        List<Integer> faceUpCounts = solitaire.getTableauFaceUpCounts();

        for (int i = 0; i < full.size(); i++) {
            List<Card> pile = full.get(i);
            List<Card> vis = visible.get(i);
            int faceUp = faceUpCounts.get(i);

            assertEquals(faceUp, vis.size(), "visible size must equal face-up count");

            int hiddenSize = Math.max(0, pile.size() - faceUp);
            Set<Card> hidden = new HashSet<>(pile.subList(0, hiddenSize));
            for (Card c : vis) {
                assertFalse(hidden.contains(c), "visible pile must not expose hidden prefix card");
            }
        }
    }
}


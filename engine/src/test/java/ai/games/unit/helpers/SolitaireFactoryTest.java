package ai.games.unit.helpers;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SolitaireFactoryTest {

    @Test
    void oneMoveFromWin_isActuallyOneMoveFromWin() {
        Solitaire s = SolitaireFactory.oneMoveFromWin();

        // Move the last king to complete the hearts foundation.
        assertTrue(s.moveCard("T1", null, "F4"));

        int totalFoundation = 0;
        for (java.util.List<Card> pile : s.getFoundation()) {
            totalFoundation += pile.size();
        }
        assertEquals(52, totalFoundation);
        assertEquals(0, s.getStockpile().size());
        assertEquals(0, s.getTalon().size());
    }

    @Test
    void flipAfterMovingLastFaceUpCard_flipsNextCard() {
        Solitaire s = SolitaireFactory.flipAfterMovingLastFaceUpCard();

        assertTrue(s.moveCard("T1", "5♠", "T2"));

        // After the move, K♦ should flip to face-up.
        assertEquals(1, s.getTableauFaceUpCounts().get(0));
        assertEquals(new Card(Rank.KING, Suit.DIAMONDS), s.getVisibleTableau().get(0).get(0));
    }
}

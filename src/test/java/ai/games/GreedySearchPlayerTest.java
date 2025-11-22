package ai.games;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.Player;
import ai.games.player.ai.GreedySearchPlayer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Simple completion test for the greedy AI: seeds a near-won state and expects the last move.
 */
class GreedySearchPlayerTest {

    @Test
    void greedyAiFinishesGame() {
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new GreedySearchPlayer();

        runSingleMoveCompletion(solitaire, ai);

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void greedyAiWinsKnownMidGameState() {
        // Seed a winnable mid-game so the greedy AI has to make several choices.
        Solitaire solitaire = seedMidGameWinnable();
        Player ai = new GreedySearchPlayer();

        for (int i = 0; i < 50 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire);
            applyCommand(solitaire, command);
        }

        assertTrue(isWon(solitaire), "Greedy AI should win the seeded mid-game state");
    }

    private void runSingleMoveCompletion(Solitaire solitaire, Player ai) {
        for (int i = 0; i < 5 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire);
            applyCommand(solitaire, command);
        }
    }

    private void applyCommand(Solitaire solitaire, String command) {
        if (command == null) {
            return;
        }
        String trimmed = command.trim();
        if (trimmed.equalsIgnoreCase("turn")) {
            solitaire.turnThree();
            return;
        }
        String[] parts = trimmed.split("\\s+");
        if (parts.length >= 3 && parts[0].equalsIgnoreCase("move")) {
            if (parts.length == 4) {
                solitaire.moveCard(parts[1], parts[2], parts[3]);
            } else {
                solitaire.moveCard(parts[1], null, parts[2]);
            }
        }
    }

    private Solitaire seedNearlyWonGame() {
        Solitaire solitaire = new Solitaire(new Deck());

        // Tableau: final card K♥ to play; others empty.
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(new Card(Rank.KING, Suit.HEARTS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 0, 0, 0, 0, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        // Foundation: 51 cards already placed; F1 top is Q♥ so K♥ is legal.
        List<List<Card>> foundation = Arrays.asList(
                pileWithTop(new Card(Rank.QUEEN, Suit.HEARTS), 12),
                fillerPile(13),
                fillerPile(13),
                fillerPile(12)
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        SolitaireTestHelper.setTalon(solitaire, Collections.emptyList());
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        return solitaire;
    }

    private List<Card> pileWithTop(Card top, int fillersBelow) {
        List<Card> pile = fillerPile(fillersBelow);
        pile.add(top);
        return pile;
    }

    private Solitaire seedMidGameWinnable() {
        Solitaire solitaire = new Solitaire(new Deck());

        // Tableau with a mix of face-up cards that can progress to foundation.
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(new Card(Rank.THREE, Suit.SPADES)),
                SolitaireTestHelper.pile(new Card(Rank.TWO, Suit.SPADES)),
                SolitaireTestHelper.pile(new Card(Rank.KING, Suit.HEARTS)),
                SolitaireTestHelper.pile(new Card(Rank.JACK, Suit.CLUBS)),
                SolitaireTestHelper.pile(new Card(Rank.QUEEN, Suit.DIAMONDS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 1, 1, 1, 1, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        // Foundation has A♠ to allow progression of spades.
        List<List<Card>> foundation = Arrays.asList(
                SolitaireTestHelper.pile(new Card(Rank.ACE, Suit.SPADES)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        // Talon gives the next spade.
        SolitaireTestHelper.setTalon(solitaire, SolitaireTestHelper.pile(new Card(Rank.FOUR, Suit.SPADES)));
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        return solitaire;
    }

    private List<Card> fillerPile(int count) {
        List<Card> pile = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            pile.add(new Card(Rank.ACE, Suit.CLUBS));
        }
        return pile;
    }

    private int totalFoundation(Solitaire solitaire) {
        int total = 0;
        for (List<Card> pile : solitaire.getFoundation()) {
            total += pile.size();
        }
        return total;
    }

    private boolean isWon(Solitaire solitaire) {
        return totalFoundation(solitaire) == 52;
    }
}

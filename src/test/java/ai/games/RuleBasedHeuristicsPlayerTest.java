package ai.games;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.Player;
import ai.games.player.ai.SimpleRuleBasedHeuristicsPlayer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Basic end-to-end test for the rule-based AI: seeds a near-complete game and expects the AI
 * to issue the final move to finish.
 */
class SimpleRuleBasedHeuristicsPlayerTest {

    @Test
    void ruleBasedAiFinishesGame() {
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new SimpleRuleBasedHeuristicsPlayer();

        runSingleMoveCompletion(solitaire, ai);

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void ruleBasedAiWinsKnownMidGameState() {
        // Seed a winnable mid-game state: AI should push toward finishing given deterministic setup.
        Solitaire solitaire = seedMidGameWinnable();
        Player ai = new SimpleRuleBasedHeuristicsPlayer();

        // Allow several moves to reach completion.
        for (int i = 0; i < 50 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "");
            applyCommand(solitaire, command);
        }

        assertTrue(isWon(solitaire), "AI should win the seeded mid-game state");
    }

    private void runSingleMoveCompletion(Solitaire solitaire, Player ai) {
        // Allow a few iterations in case AI chooses a turn first; but this scenario needs one move.
        for (int i = 0; i < 5 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "");
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

        // Tableau: T1 holds the final King♥ face up; others empty.
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

        // Foundation: pre-fill 51 cards; F1 top is Q♥ so K♥ is legal.
        List<List<Card>> foundation = Arrays.asList(
                pileWithTop(new Card(Rank.QUEEN, Suit.HEARTS), 12),
                fillerPile(13),
                fillerPile(13),
                fillerPile(12)
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        // Empty stock/talon to force the move.
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

        // Tableau with a mix of face-up cards enabling foundation moves.
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(new Card(Rank.THREE, Suit.HEARTS)),
                SolitaireTestHelper.pile(new Card(Rank.TWO, Suit.HEARTS)),
                SolitaireTestHelper.pile(new Card(Rank.KING, Suit.SPADES)),
                SolitaireTestHelper.pile(new Card(Rank.JACK, Suit.DIAMONDS)),
                SolitaireTestHelper.pile(new Card(Rank.QUEEN, Suit.CLUBS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 1, 1, 1, 1, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        // Foundation has A♥ so 2♥/3♥ can move.
        List<List<Card>> foundation = Arrays.asList(
                SolitaireTestHelper.pile(new Card(Rank.ACE, Suit.HEARTS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        // Talon provides supportive cards.
        SolitaireTestHelper.setTalon(solitaire, SolitaireTestHelper.pile(new Card(Rank.FOUR, Suit.HEARTS)));
        // Stockpile empty to force using available moves.
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

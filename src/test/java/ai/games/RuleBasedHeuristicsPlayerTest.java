package ai.games;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.Player;
import ai.games.player.ai.RuleBasedHeuristicsPlayer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Basic end-to-end test for the rule-based AI: seeds a near-complete game and expects the AI
 * to issue the final move to finish.
 */
class RuleBasedHeuristicsPlayerTest {
    private static final int MAX_TEST_STEPS = 1000;

    @Test
    void ruleBasedAiFinishesGame() {
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new RuleBasedHeuristicsPlayer();

        runSingleMoveCompletion(solitaire, ai);

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void ruleBasedAiWinsKnownMidGameState() {
        // Seed a winnable mid-game state: AI should push toward finishing given deterministic setup.
        Solitaire solitaire = seedMidGameWinnable();
        Player ai = new RuleBasedHeuristicsPlayer();

        // Allow several moves to reach completion.
        for (int i = 0; i < MAX_TEST_STEPS && !isWon(solitaire); i++) {
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

        List<Card> deck = SolitaireTestHelper.fullDeck();

        // Tableau: T1 holds the final King♥ face up; others empty.
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

        // Foundation: hearts A..Q on F1, remainder split across F2–F4.
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

        // Empty stock/talon to force the move.
        SolitaireTestHelper.setTalon(solitaire, Collections.emptyList());
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        SolitaireTestHelper.assertFullDeckState(solitaire);
        return solitaire;
    }

    private Solitaire seedMidGameWinnable() {
        Solitaire solitaire = new Solitaire(new Deck());

        List<Card> deck = SolitaireTestHelper.fullDeck();

        // Tableau with a mix of face-up cards enabling foundation moves.
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.THREE, Suit.HEARTS)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.TWO, Suit.HEARTS)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.KING, Suit.SPADES)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.JACK, Suit.DIAMONDS)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.QUEEN, Suit.CLUBS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 1, 1, 1, 1, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        // Foundation has A♥ so 2♥/3♥ can move.
        List<List<Card>> foundation = Arrays.asList(
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.ACE, Suit.HEARTS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        // Talon provides supportive cards.
        SolitaireTestHelper.setTalon(solitaire, SolitaireTestHelper.pile(
                SolitaireTestHelper.takeCard(deck, Rank.FOUR, Suit.HEARTS)));
        // Remaining cards into stockpile.
        SolitaireTestHelper.setStockpile(solitaire, new ArrayList<>(deck));

        SolitaireTestHelper.assertFullDeckState(solitaire);
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

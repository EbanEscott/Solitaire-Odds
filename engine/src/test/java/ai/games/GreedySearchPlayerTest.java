package ai.games;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.Player;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.GreedySearchPlayer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple completion test for the greedy AI: seeds a near-won state and expects the last move.
 */
class GreedySearchPlayerTest {
    private static final Logger log = LoggerFactory.getLogger(GreedySearchPlayerTest.class);
    private static final int MAX_TEST_STEPS = 1000;

    @Test
    void greedyAiFinishesGame() {
        logTestHeader("greedyAiFinishesGame");
        Solitaire solitaire = seedNearlyWonGame();
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (nearly-won test):\n{}", stripAnsi(solitaire.toString()));
        }

        runSingleMoveCompletion(solitaire, ai);

        if (log.isDebugEnabled()) {
            log.debug("Final board (nearly-won test):\n{}", stripAnsi(solitaire.toString()));
        }

        assertEquals(52, totalFoundation(solitaire), "AI should finish the game");
        assertTrue(isWon(solitaire));
    }

    @Test
    void greedyAiWinsKnownMidGameState() {
        logTestHeader("greedyAiWinsKnownMidGameState");
        // Seed a winnable mid-game so the greedy AI has to make several choices.
        Solitaire solitaire = seedMidGameWinnable();
        GreedySearchPlayer.resetForNewGame();
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (mid-game test):\n{}", stripAnsi(solitaire.toString()));
        }

        for (int i = 0; i < MAX_TEST_STEPS && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiWinsKnownMidGameState: step {} command={}", i, command);
            }
            if (command == null) {
                break;
            }
            String trimmed = command.trim();
            if ("quit".equalsIgnoreCase(trimmed)) {
                break;
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
        }

        assertTrue(isWon(solitaire), "Greedy AI should win the seeded mid-game state");
    }

    @Test
    void greedyAiDoesNotQuitWithMovesAvailableInRandomGame() {
        logTestHeader("greedyAiDoesNotQuitWithMovesAvailableInRandomGame");
        GreedySearchPlayer.resetForNewGame();
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new GreedySearchPlayer();

        if (log.isDebugEnabled()) {
            log.debug("Initial board (random game test):\n{}", stripAnsi(solitaire.toString()));
        }

        for (int step = 0; step < MAX_TEST_STEPS && !isWon(solitaire); step++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiDoesNotQuitWithMovesAvailableInRandomGame: step {} command={}", step, command);
            }
            if (command == null) {
                break;
            }
            String trimmed = command.trim();
            if ("quit".equalsIgnoreCase(trimmed)) {
                List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
                boolean hasNonQuitMove = legal.stream().anyMatch(m -> !"quit".equalsIgnoreCase(m.trim()));
                assertTrue(!hasNonQuitMove, "Greedy quit while non-quit moves were still legal: " + legal);
                break;
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
        }
    }

    private void runSingleMoveCompletion(Solitaire solitaire, Player ai) {
        for (int i = 0; i < 5 && !isWon(solitaire); i++) {
            String command = ai.nextCommand(solitaire, "", "");
            if (log.isDebugEnabled()) {
                log.debug("greedyAiFinishesGame: step {} command={}", i, command);
            }
            applyCommand(solitaire, command);
            if (log.isDebugEnabled()) {
                log.debug("Board after command:\n{}", stripAnsi(solitaire.toString()));
            }
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

        // Build a full 52-card deck and allocate specific cards to tableau/foundation.
        List<Card> deck = SolitaireTestHelper.fullDeck();

        // Tableau: final card K♥ to play; others empty.
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

        // Foundation:
        // - F1 has hearts A..Q (K♥ reserved on tableau).
        // - F2–F4 take the remaining cards split evenly.
        List<Card> hearts = new ArrayList<>();
        for (Rank rank : Rank.values()) {
            if (rank == Rank.KING) {
                continue;
            }
            hearts.add(SolitaireTestHelper.takeCard(deck, rank, Suit.HEARTS));
        }
        List<List<Card>> foundation = new ArrayList<>();
        foundation.add(hearts);

        // Distribute remaining cards into three foundation piles.
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

        // No stock or talon: all cards allocated to tableau/foundation.
        SolitaireTestHelper.setTalon(solitaire, Collections.emptyList());
        SolitaireTestHelper.setStockpile(solitaire, Collections.emptyList());

        SolitaireTestHelper.assertFullDeckState(solitaire);
        return solitaire;
    }

    private Solitaire seedMidGameWinnable() {
        Solitaire solitaire = new Solitaire(new Deck());

        // Start from a full deck and carve out a specific mid-game layout.
        List<Card> deck = SolitaireTestHelper.fullDeck();

        // Tableau with a mix of face-up cards that can progress to foundation.
        List<List<Card>> tableau = Arrays.asList(
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.THREE, Suit.SPADES)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.TWO, Suit.SPADES)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.KING, Suit.HEARTS)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.JACK, Suit.CLUBS)),
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.QUEEN, Suit.DIAMONDS)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        List<Integer> faceUp = Arrays.asList(1, 1, 1, 1, 1, 0, 0);
        SolitaireTestHelper.setTableau(solitaire, tableau, faceUp);

        // Foundation has A♠ to allow progression of spades.
        List<List<Card>> foundation = Arrays.asList(
                SolitaireTestHelper.pile(SolitaireTestHelper.takeCard(deck, Rank.ACE, Suit.SPADES)),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile(),
                SolitaireTestHelper.emptyPile()
        );
        SolitaireTestHelper.setFoundation(solitaire, foundation);

        // Talon gives the next spade.
        SolitaireTestHelper.setTalon(solitaire, SolitaireTestHelper.pile(
                SolitaireTestHelper.takeCard(deck, Rank.FOUR, Suit.SPADES)));
        // Remaining cards go to stockpile.
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

    private static String stripAnsi(String input) {
        return input.replaceAll("\\u001B\\[[;\\d]*m", "");
    }

    private static void logTestHeader(String testName) {
        if (!log.isDebugEnabled()) {
            return;
        }
        String banner = """
============================================================
   ____                 _           _        _             
  / ___| ___ _ __   ___| |__   __ _| |_ _ __(_)_ __   __ _ 
 | |  _ / _ \\ '_ \\ / __| '_ \\ / _` | __| '__| | '_ \\ / _` |
 | |_| |  __/ | | | (__| | | | (_| | |_| |  | | | | | (_| |
  \\____|\\___|_| |_|\\___|_| |_|\\__,_|\\__|_|  |_|_| |_|\\__, |
                                                      |___/ 
============================================================
GREEDY TEST START: %s
============================================================""".formatted(testName);
        log.debug("\n{}", banner);
    }
}

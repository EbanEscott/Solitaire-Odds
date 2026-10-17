package ai.games.unit.player;

import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.ai.GreedySearchPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Deterministic regressions for greedy one-ply scoring and per-game history. */
class GreedyRulesTest {

    /** Moving the complete visible run must outrank moving only its exposed top card. */
    @Test
    void prefersActualHiddenCardReveal() {
        /** Both run heads have legal destinations, but only the seven exposes the hidden card. */
        Solitaire board = SolitaireBuilder.newGame()
                .tableau("T1", 2, "9♠", "7♣", "6♥")
                .tableau("T2", "8♥").tableau("T3", "7♠").build();
        assertEquals("move T1 7♣ T2", new GreedySearchPlayer().nextCommand(board, "", ""));
    }

    /** Merely relocating a king to another empty column must not earn an empty-column bonus. */
    @Test
    void prefersWasteDeploymentOverKingRelocation() {
        /** Waste deployment has immediate utility while the king move preserves the empty count. */
        Solitaire board = SolitaireBuilder.newGame().tableau("T1", "K♠")
                .tableau("T2", "8♥").waste("7♣").build();
        assertEquals("move W T2", new GreedySearchPlayer().nextCommand(board, "", ""));
    }

    /** A legal but unsupported foundation push should not outrank a real tableau reveal. */
    @Test
    void foundationSafetyUsesOtherSuits() {
        /** The club predecessor makes the push legal, but does not make it conservatively safe. */
        Solitaire board = SolitaireBuilder.newGame().foundation("F1", "A♣", "2♣")
                .tableau("T1", "3♣").tableau("T2", 1, "9♠", "7♣")
                .tableau("T3", "8♥").build();
        assertEquals("move T2 7♣ T3", new GreedySearchPlayer().nextCommand(board, "", ""));
    }

    /** A second player must not inherit the first player's observed transitions or visited states. */
    @Test
    void historiesAreIndependentAndResettable() {
        /** Visit a successor with one player, then replay the initial board with another. */
        Solitaire initial = SolitaireBuilder.newGame().tableau("T1", "A♣").build();
        GreedySearchPlayer first = new GreedySearchPlayer();
        String expected = first.nextCommand(initial, "", "");
        Solitaire successor = initial.copy();
        Move move = Move.parse(expected);
        assertTrue(successor.moveCard(move.from().toCode(), move.card().shortName(), move.to().toCode()));
        first.nextCommand(successor, "", "");
        assertEquals(expected, new GreedySearchPlayer().nextCommand(initial, "", ""));
        /** Explicit lifecycle reset restores the original deterministic choice. */
        first.resetGame();
        assertEquals(expected, first.nextCommand(initial, "", ""));
    }

    /** An exhausted stock cycle terminates even though recycling remains legally available. */
    @Test
    void terminatesRepeatedStockObservations() {
        /** Bury every other card beneath an immovable ten, leaving three unusable stock cards. */
        java.util.List<String> buried = new java.util.ArrayList<>();
        for (ai.games.game.Suit suit : ai.games.game.Suit.values()) {
            /** Unknown is a planning placeholder, not a physical deck suit. */
            if (suit == ai.games.game.Suit.UNKNOWN) {
                continue;
            }
            /** Enumerate physical ranks for the current suit. */
            for (ai.games.game.Rank rank : ai.games.game.Rank.values()) {
                /** Reserve the tableau top and stock cards while retaining all 52 identities. */
                if (rank != ai.games.game.Rank.UNKNOWN) {
                    String card = new ai.games.game.Card(rank, suit).shortName();
                    /** Keep reserved cards out of the buried prefix. */
                    if (!java.util.Set.of("5♣", "6♣", "7♣", "10♣").contains(card)) {
                        buried.add(card);
                    }
                }
            }
        }
        buried.add("10♣");
        Solitaire board = SolitaireBuilder.newGame().tableau("T1", 1, buried.toArray(String[]::new))
                .stock("5♣", "6♣", "7♣").build();
        GreedySearchPlayer player = new GreedySearchPlayer();
        int turns = 0;
        /** Observe actual draws and recycles until history proves that no new position remains. */
        while (turns < 10) {
            String command = player.nextCommand(board, "", "");
            /** Stop when every remaining successor repeats a known position. */
            if (command.equals("quit")) {
                break;
            }
            assertEquals("turn", command);
            assertTrue(board.turnStock());
            turns++;
        }
        assertTrue(turns >= 1 && turns < 10, "Draw the stock, then terminate before repeating an observed position");
        assertTrue(board.canTurnStock(), "Cycle termination is distinct from a legal stock-pass limit");
    }

}

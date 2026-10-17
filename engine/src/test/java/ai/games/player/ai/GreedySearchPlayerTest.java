package ai.games.player.ai;

import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Functional coverage for deterministic greedy completion and legal cycle termination. */
class GreedySearchPlayerTest {

    /** A nearly completed deal must finish with its remaining legal foundation move. */
    @Test
    void greedyAiFinishesGame() {
        /** Run the fixed endgame without relying on random deals or shared player history. */
        assertCompletes(SolitaireFactory.oneMoveFromWin(), 5);
    }

    /** The established mid-game fixture remains solvable with one-ply greedy decisions. */
    @Test
    void greedyAiWinsKnownMidGameState() {
        /** Preserve the existing complete-deck fixture and its completion expectation. */
        Solitaire board = SolitaireBuilder.newGame().foundation("F1", "A♠")
                .tableau("T1", "3♠").tableau("T2", "2♠").tableau("T3", "K♥")
                .tableau("T4", "J♣").tableau("T5", "Q♦").waste("4♠").build();
        assertCompletes(board, 1000);
    }

    /** Checks legal execution and completion while allowing quit when all choices repeat history. */
    private void assertCompletes(Solitaire board, int limit) {
        /** Use a fresh player and validate every selected action against the real board. */
        GreedySearchPlayer player = new GreedySearchPlayer();
        for (int step = 0; step < limit && !won(board); step++) {
            String command = player.nextCommand(board, "", "");
            assertTrue(LegalMovesHelper.listLegalMoves(board).contains(command), command);
            Move move = Move.parse(command);
            /** A quit is legal but cannot satisfy a completion fixture prematurely. */
            if (move.isQuit()) {
                break;
            }
            /** Apply configured turns or structured card moves and require successful execution. */
            if (move.isTurn()) {
                assertTrue(board.turnStock());
            } else {
                assertTrue(board.moveCard(move.from().toCode(),
                        move.card() == null ? null : move.card().shortName(), move.to().toCode()));
            }
        }
        assertTrue(won(board), "Greedy should complete this fixed winnable fixture");
    }

    /** Identifies completion using all four foundation sizes. */
    private boolean won(Solitaire board) {
        /** All 52 cards must have reached foundations. */
        return board.getFoundation().stream().mapToInt(java.util.List::size).sum() == 52;
    }
}

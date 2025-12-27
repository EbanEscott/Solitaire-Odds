package ai.games.unit.game;

import ai.games.game.Deck;
import ai.games.game.Solitaire;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for training mode undo functionality.
 * Tests move history tracking and replay-based undo.
 */
class TrainingModeTest {

    @Test
    void testCanUndoReturnsFalseWhenNoMoves() {
        Solitaire solitaire = new Solitaire(new Deck());
        assertFalse(solitaire.canUndo());
    }

    @Test
    void testCanUndoReturnsTrueAfterMove() {
        Solitaire solitaire = new Solitaire(new Deck());
        // Seed a move by attempting one
        Solitaire.MoveResult result = solitaire.attemptMove("W", null, "T1");
        // Result may succeed or fail depending on game state, but move should be recorded if successful
        if (result.success) {
            assertTrue(solitaire.canUndo());
        }
    }

    @Test
    void testMoveHistorySizeIncrementsOnSuccessfulMove() {
        Solitaire solitaire = new Solitaire(new Deck());
        int historyBefore = solitaire.getMoveHistorySize();
        Solitaire.MoveResult result = solitaire.attemptMove("W", null, "T1");
        if (result.success) {
            assertEquals(historyBefore + 1, solitaire.getMoveHistorySize());
        }
    }

    @Test
    void testTurnActionIsRecorded() {
        Solitaire solitaire = new Solitaire(new Deck());
        int historyBefore = solitaire.getMoveHistorySize();
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        assertEquals(historyBefore + 1, solitaire.getMoveHistorySize());
    }

    @Test
    void testUndoLastMoveFalseWhenNoMoves() {
        Solitaire solitaire = new Solitaire(new Deck());
        assertFalse(solitaire.undoLastMove());
    }

    @Test
    void testUndoLastMoveReducesHistorySize() {
        Solitaire solitaire = new Solitaire(new Deck());
        // Perform a turn (always succeeds)
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        int historyAfterTurn = solitaire.getMoveHistorySize();
        
        // Now undo
        boolean undoResult = solitaire.undoLastMove();
        assertTrue(undoResult);
        assertEquals(historyAfterTurn - 1, solitaire.getMoveHistorySize());
    }

    @Test
    void testUndoRestoresGameState() {
        Solitaire solitaire = new Solitaire(new Deck());
        // Turn three cards to set up a known state
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        
        int stockpileAfterTurn = solitaire.getStockpile().size();
        int talonAfterTurn = solitaire.getTalon().size();
        
        // Now undo
        solitaire.undoLastMove();
        
        // After undo, we should be back to initial state
        int stockpileAfterUndo = solitaire.getStockpile().size();
        int talonAfterUndo = solitaire.getTalon().size();
        
        // Talon should be empty (as in initial state)
        assertEquals(0, talonAfterUndo);
        // Stockpile should have all 24 cards (none turned yet)
        assertEquals(24, stockpileAfterUndo);
    }

    @Test
    void testActionRecordCreatesValidMoveAction() {
        Solitaire.Action moveAction = Solitaire.Action.move("W", "K♣", "T1");
        assertEquals("move", moveAction.type());
        assertEquals("W", moveAction.from());
        assertEquals("K♣", moveAction.cardCode());
        assertEquals("T1", moveAction.to());
    }

    @Test
    void testActionRecordCreateValidTurnAction() {
        Solitaire.Action turnAction = Solitaire.Action.turn();
        assertEquals("turn", turnAction.type());
        assertNull(turnAction.from());
        assertNull(turnAction.cardCode());
        assertNull(turnAction.to());
    }

    @Test
    void testMultipleTurnsAndUndo() {
        // Test scenario: turn 3 times, undo one, verify state
        Solitaire solitaire = new Solitaire(new Deck());
        
        // Turn 1
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        assertEquals(1, solitaire.getMoveHistorySize());
        
        // Turn 2
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        assertEquals(2, solitaire.getMoveHistorySize());
        
        // Turn 3
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        assertEquals(3, solitaire.getMoveHistorySize());
        
        // Undo last turn
        assertTrue(solitaire.undoLastMove());
        assertEquals(2, solitaire.getMoveHistorySize());
        
        // Verify we can undo again
        assertTrue(solitaire.undoLastMove());
        assertEquals(1, solitaire.getMoveHistorySize());
    }

    @Test
    void testMoveReplayAfterUndo() {
        // Test that moves are correctly replayed after undo
        Solitaire solitaire = new Solitaire(new Deck());
        
        // Record an initial turn
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        int stockpileAfterFirstTurn = solitaire.getStockpile().size();
        
        // Record another turn
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        int stockpileAfterSecondTurn = solitaire.getStockpile().size();
        
        // Undo the second turn
        solitaire.undoLastMove();
        
        // Should be back to state after first turn
        assertEquals(stockpileAfterFirstTurn, solitaire.getStockpile().size());
        assertEquals(1, solitaire.getMoveHistorySize());
    }

    @Test
    void testUndoAllTheWayBackToStart() {
        // Test that we can undo all actions and get back to initial state
        Solitaire solitaire = new Solitaire(new Deck());
        
        // Perform 3 actions
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        solitaire.turnThree();
        solitaire.recordAction(Solitaire.Action.turn());
        
        assertEquals(3, solitaire.getMoveHistorySize());
        
        // Undo all
        assertTrue(solitaire.undoLastMove());
        assertTrue(solitaire.undoLastMove());
        assertTrue(solitaire.undoLastMove());
        
        // Should be at start
        assertEquals(0, solitaire.getMoveHistorySize());
        assertFalse(solitaire.canUndo());
        assertEquals(0, solitaire.getTalon().size());
        assertEquals(24, solitaire.getStockpile().size());
    }
}

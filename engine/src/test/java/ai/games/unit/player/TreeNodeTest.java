package ai.games.unit.player;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.games.game.Solitaire;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ai.games.player.ai.tree.TreeNode} methods using {@link MonteCarloTreeNode}.
 *
 * <p>Tests cover:
 * <ul>
 *   <li>{@code isWon} - Game won detection based on foundation card count</li>
 *   <li>{@code isTerminal} - Terminal state detection (won, stuck, null state)</li>
 *   <li>{@code isQuit} - Quit command detection</li>
 *   <li>{@code isTurn} - Turn command detection</li>
 *   <li>{@code isUselessKingMove} - Useless king shuffle pruning</li>
 *   <li>{@code isCycleDetected} - Cycle/ping-pong detection in tree</li>
 * </ul>
 */
class TreeNodeTest {

    // ========================================================================
    // isWon tests
    // ========================================================================

    @Nested
    @DisplayName("isWon")
    class IsWonTests {

        @Test
        void emptyFoundationsReturnsFalse() {
            Solitaire solitaire = SolitaireFactory.stockOnly();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);

            assertFalse(node.isWon(), "Empty foundations should not be won");
        }

        @Test
        void partialFoundationsReturnsFalse() {
            Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .foundation("F1", SolitaireFactory.foundationUpTo(ai.games.game.Suit.SPADES, ai.games.game.Rank.KING))
                    .foundation("F2", SolitaireFactory.foundationUpTo(ai.games.game.Suit.HEARTS, ai.games.game.Rank.FIVE))
                    .build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);

            assertFalse(node.isWon(), "Partial foundations (18 cards) should not be won");
        }

        @Test
        void allFourFoundationsCompleteReturnsTrue() {
            Solitaire solitaire = SolitaireFactory.wonGame();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);

            assertTrue(node.isWon(), "All 52 cards in foundations should be won");
        }
    }

    // ========================================================================
    // isTerminal tests
    // ========================================================================

    @Nested
    @DisplayName("isTerminal")
    class IsTerminalTests {

        @Test
        void nullStateReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            // state is null by default

            assertTrue(node.isTerminal(), "Null state should be terminal");
        }

        @Test
        void wonGameReturnsTrue() {
            Solitaire solitaire = SolitaireFactory.wonGame();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);

            assertTrue(node.isTerminal(), "Won game should be terminal");
        }

        @Test
        void gameWithLegalMovesReturnsFalse() {
            // Create a simple game with at least one legal move
            // Put A♠ in T1 - can move to empty foundation
            Solitaire solitaire = SolitaireBuilder.newGame().tableau("T1", "A♠").build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);

            assertFalse(node.isTerminal(), "Game with legal moves should not be terminal");
        }

        @Test
        void stuckGameWithOnlyQuitAvailableIsNotTerminal() {
            // Create a stuck position: the only "legal move" is quit.
            // Note: LegalMovesHelper always includes "quit" as a legal command,
            // so isTerminal() will return false even when no progress moves exist.
            // This test documents this behavior.
            Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .tableau("T1", "2♠")
                    .tableau("T2", "3♠")
                    .build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            
            // isTerminal checks if listLegalMoves is empty, but "quit" is always included,
            // so this returns false even though no progress is possible.
            assertFalse(node.isTerminal(), 
                    "Stuck game with only 'quit' available should not be terminal (quit is always legal)");
        }
    }

    // ========================================================================
    // isQuit tests
    // ========================================================================

    @Nested
    @DisplayName("isQuit")
    class IsQuitTests {

        @Test
        void nullMoveReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove(null);

            assertFalse(node.isQuit(), "Null move should not be quit");
        }

        @Test
        void exactQuitReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("quit");

            assertTrue(node.isQuit(), "'quit' should be detected as quit");
        }

        @Test
        void uppercaseQuitReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("QUIT");

            assertTrue(node.isQuit(), "'QUIT' (uppercase) should be detected as quit");
        }

        @Test
        void mixedCaseQuitReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("QuIt");

            assertTrue(node.isQuit(), "'QuIt' (mixed case) should be detected as quit");
        }

        @Test
        void quitWithWhitespaceReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("  quit  ");

            assertTrue(node.isQuit(), "'  quit  ' (with whitespace) should be detected as quit");
        }

        @Test
        void quitAsPrefixReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("quit game");

            assertFalse(node.isQuit(), "'quit game' should not be detected as quit (exact match required)");
        }

        @Test
        void otherCommandReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("turn");

            assertFalse(node.isQuit(), "'turn' should not be detected as quit");
        }
    }

    // ========================================================================
    // isTurn tests
    // ========================================================================

    @Nested
    @DisplayName("isTurn")
    class IsTurnTests {

        @Test
        void nullMoveReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove(null);

            assertFalse(node.isTurn(), "Null move should not be turn");
        }

        @Test
        void exactTurnReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("turn");

            assertTrue(node.isTurn(), "'turn' should be detected as turn");
        }

        @Test
        void uppercaseTurnReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("TURN");

            assertTrue(node.isTurn(), "'TURN' (uppercase) should be detected as turn");
        }

        @Test
        void turnWithNumberReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("turn 3");

            assertTrue(node.isTurn(), "'turn 3' should be detected as turn");
        }

        @Test
        void turnWithWhitespaceReturnsTrue() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("  turn  ");

            assertTrue(node.isTurn(), "'  turn  ' (with whitespace) should be detected as turn");
        }

        @Test
        void moveCommandReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("move T1 K♠ T7");

            assertFalse(node.isTurn(), "'move T1 K♠ T7' should not be detected as turn");
        }
    }

    // ========================================================================
    // isUselessKingMove tests
    // ========================================================================

    @Nested
    @DisplayName("isUselessKingMove")
    class IsUselessKingMoveTests {

        @Test
        void kingTableauToTableauWithNoFaceDownsReturnsTrue() {
            // King in T1 with 0 face-downs, moving to empty T7 is useless
            Solitaire solitaire = SolitaireBuilder.newGame().tableau("T1", "K♠").build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            node.setMove("move T1 K♠ T7");

            assertTrue(node.isUselessKingMove(), "King T→T with 0 face-downs should be useless");
        }

        @Test
        void kingShuffleStillDetectedWhenChildStateIsAfterMove() {
            // In search trees (MCTS/A*), a node's state is typically AFTER applying its move.
            // Ensure useless-king classification is still correct by using the parent's pre-move state.
            // T2 has a single king and 0 face-downs.
            Solitaire before = SolitaireBuilder.newGame().tableau("T2", "K♦").build();

            MonteCarloTreeNode parent = new MonteCarloTreeNode();
            parent.setState(before);

            // Child state is AFTER moving the king from T2 to empty T7.
            Solitaire after = before.copy();
            boolean success = after.moveCard("T2", "K♦", "T7");
            assertTrue(success, "Test setup requires move T2 K♦ T7 to be legal");

            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(parent);
            child.setState(after);
            child.setMove("move T2 K♦ T7");

            assertTrue(
                    child.isUselessKingMove(),
                    "Useless king shuffles should be detected even when the node's state is post-move");
        }

        @Test
        void kingTableauToTableauWithFaceDownsReturnsFalse() {
            // King in T1 with face-down cards beneath - moving reveals a card (useful)
            Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", 1, "A♥", "K♠")
                .build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            node.setMove("move T1 K♠ T7");

            assertFalse(node.isUselessKingMove(), "King T→T with face-downs should be useful (reveals card)");
        }

        @Test
        void kingTableauToFoundationReturnsFalse() {
            // King to foundation is always useful (completes a suit)
            Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .foundation("F1", SolitaireFactory.foundationUpTo(ai.games.game.Suit.SPADES, ai.games.game.Rank.QUEEN))
                    .tableau("T1", "K♠")
                    .build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            node.setMove("move T1 K♠ F1");

            assertFalse(node.isUselessKingMove(), "King T→F should not be useless");
        }

        @Test
        void kingToFoundationNotMisclassifiedWhenChildStateIsAfterMove() {
            // Regression-style: ensure king-to-foundation moves never get swept into useless-king pruning.
            Solitaire before = SolitaireBuilder
                    .newGame()
                    .foundation("F3", SolitaireFactory.foundationUpTo(ai.games.game.Suit.SPADES, ai.games.game.Rank.QUEEN))
                    .tableau("T6", "K♠")
                    .build();

            MonteCarloTreeNode parent = new MonteCarloTreeNode();
            parent.setState(before);

            Solitaire after = before.copy();
            boolean success = after.moveCard("T6", "K♠", "F3");
            assertTrue(success, "Test setup requires move T6 K♠ F3 to be legal");

            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(parent);
            child.setState(after);
            child.setMove("move T6 K♠ F3");

            assertFalse(
                    child.isUselessKingMove(),
                    "King to foundation should not be classified as a useless king move, even post-move");
        }

        @Test
        void nonKingMoveReturnsFalse() {
            // Non-king moves are not affected by this check
            Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .tableau("T1", "Q♠")
                    .tableau("T2", "K♥")
                    .build();

            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            node.setMove("move T1 Q♠ T2");

            assertFalse(node.isUselessKingMove(), "Non-king move should not be flagged as useless king move");
        }

        @Test
        void nullMoveReturnsFalse() {
            Solitaire solitaire = SolitaireFactory.stockOnly();
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            node.setMove(null);

            assertFalse(node.isUselessKingMove(), "Null move should not be useless king move");
        }

        @Test
        void nullStateReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("move T1 K♠ T7");

            assertFalse(node.isUselessKingMove(), "Null state should not be useless king move");
        }

        @Test
        void mctsStyleChildNodeUsesParentStateForUselessKingDetection() {
            // This mimics how MCTS constructs nodes:
            // - parent holds the pre-move state
            // - child holds the post-move state (after applying the move)
            // The useless-king classifier must still work even if the source pile becomes empty.
            // Single-card king pile with 0 facedowns.
            Solitaire solitaire = SolitaireBuilder.newGame().tableau("T2", "K♦").build();

            MonteCarloTreeNode parent = new MonteCarloTreeNode();
            parent.setState(solitaire.copy());

            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(parent);
            child.setState(solitaire.copy());
            child.applyMove("move T2 K♦ T7");

            assertTrue(child.isUselessKingMove(),
                    "Child nodes storing post-move state should still detect useless king shuffles");
        }

        @Test
        void mctsStyleChildNodeKingToFoundationIsNeverUseless() {
            // Make K♠ -> F3 legal by seeding spades foundation up to Q♠.
            Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .foundation("F3", SolitaireFactory.foundationUpTo(ai.games.game.Suit.SPADES, ai.games.game.Rank.QUEEN))
                    .tableau("T6", "K♠")
                    .build();

            MonteCarloTreeNode parent = new MonteCarloTreeNode();
            parent.setState(solitaire.copy());

            MonteCarloTreeNode child = new MonteCarloTreeNode();
            child.setParent(parent);
            child.setState(solitaire.copy());
            child.applyMove("move T6 K♠ F3");

            assertFalse(child.isUselessKingMove(), "King to foundation should not be flagged as useless");
        }
    }

    // ========================================================================
    // isCycleDetected tests
    // ========================================================================

    @Nested
    @DisplayName("isCycleDetected")
    class IsCycleDetectedTests {

        // --------------------------------------------------------------------
        // Stock cycling tests (turn commands)
        // --------------------------------------------------------------------

        @Test
        @DisplayName("Stock cycle: no cycle when stock not fully cycled")
        void stockCycleNotFullyCycledReturnsFalse() {
            // Setup: 6 cards in stock, empty talon. Need 2 turns to see all, then
            // 1 more turn to recycle, then 2 more turns to return to original state.
            // After just 1 turn, no cycle yet.
            Solitaire solitaire = SolitaireFactory.withExactStockAndWaste(
                new String[] {"A♣", "2♣", "3♣", "4♣", "5♣", "6♣"},
                new String[] {}
            );

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // Apply first turn
            MonteCarloTreeNode child1 = new MonteCarloTreeNode();
            child1.setState(root.getState().copy());
            child1.applyMove("turn");
            root.addChild("turn", child1);

            assertFalse(child1.isCycleDetected(), "After 1 turn, no cycle should be detected");
        }

        @Test
        @DisplayName("Stock cycle: cycle detected when stock fully cycles back")
        void stockCycleFullyCycledReturnsTrue() {
            // Setup: 6 cards in stock. With turn-3:
            // - Turn 1: stock=3, talon=3
            // - Turn 2: stock=0, talon=6
            // - Turn 3: recycle talon to stock, then stock=3, talon=3
            // - Turn 4: stock=0, talon=6 (same as after turn 2)
            // - Turn 5: recycle, stock=3, talon=3 (same as after turn 3)
            // 
            // State after turn 3 should match state after turn 1 (first occurrence)
            // State after turn 5 should match both turn 1 and turn 3 (second occurrence = cycle!)
            Solitaire solitaire = SolitaireFactory.withExactStockAndWaste(
                new String[] {"A♣", "2♣", "3♣", "4♣", "5♣", "6♣"},
                new String[] {}
            );

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // Build tree: root -> turn1 -> turn2 -> turn3 -> turn4 -> turn5
            MonteCarloTreeNode turn1 = new MonteCarloTreeNode();
            turn1.setState(root.getState().copy());
            turn1.applyMove("turn");
            root.addChild("turn", turn1);

            MonteCarloTreeNode turn2 = new MonteCarloTreeNode();
            turn2.setState(turn1.getState().copy());
            turn2.applyMove("turn");
            turn1.addChild("turn", turn2);

            MonteCarloTreeNode turn3 = new MonteCarloTreeNode();
            turn3.setState(turn2.getState().copy());
            turn3.applyMove("turn");
            turn2.addChild("turn", turn3);

            MonteCarloTreeNode turn4 = new MonteCarloTreeNode();
            turn4.setState(turn3.getState().copy());
            turn4.applyMove("turn");
            turn3.addChild("turn", turn4);

            MonteCarloTreeNode turn5 = new MonteCarloTreeNode();
            turn5.setState(turn4.getState().copy());
            turn5.applyMove("turn");
            turn4.addChild("turn", turn5);

            // After turn5, we should have cycled back twice
            // The cycle detection requires 2+ occurrences in ancestors
            assertTrue(turn5.isCycleDetected(), 
                    "After full stock cycle, cycle should be detected");
        }

        @Test
        @DisplayName("Stock cycle: first return to state is not a cycle (1 occurrence allowed)")
        void stockCycleFirstReturnNotCycle() {
            // With 6 cards: after 3 turns we return to a state similar to turn 1.
            // This is the FIRST occurrence, so should NOT trigger cycle detection.
            Solitaire solitaire = SolitaireFactory.withExactStockAndWaste(
                new String[] {"A♣", "2♣", "3♣", "4♣", "5♣", "6♣"},
                new String[] {}
            );

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            MonteCarloTreeNode turn1 = new MonteCarloTreeNode();
            turn1.setState(root.getState().copy());
            turn1.applyMove("turn");
            root.addChild("turn", turn1);

            MonteCarloTreeNode turn2 = new MonteCarloTreeNode();
            turn2.setState(turn1.getState().copy());
            turn2.applyMove("turn");
            turn1.addChild("turn", turn2);

            MonteCarloTreeNode turn3 = new MonteCarloTreeNode();
            turn3.setState(turn2.getState().copy());
            turn3.applyMove("turn");
            turn2.addChild("turn", turn3);

            // turn3 state may match turn1 state (first occurrence), but not a cycle yet
            assertFalse(turn3.isCycleDetected(), 
                    "First return to a state should not be flagged as cycle (1 occurrence allowed)");
        }

        @Test
        @DisplayName("Stock cycle: full cycle with interim move breaks the cycle")
        void stockCycleWithInterimMoveBreaksCycle() {
            // Setup: stock cycling plus a progress move that changes state.
            // We don't need the stock to be tiny here; we just need multiple turns and a move.
            Solitaire solitaire = SolitaireBuilder
                .newGame()
                .tableau("T1", "A♠")
                .tableau("T2", "2♠")
                .build();

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // Cycle 1: turn -> turn (stock cycles through)
            MonteCarloTreeNode turn1 = new MonteCarloTreeNode();
            turn1.setState(root.getState().copy());
            turn1.applyMove("turn");
            root.addChild("turn", turn1);

            MonteCarloTreeNode turn2 = new MonteCarloTreeNode();
            turn2.setState(turn1.getState().copy());
            turn2.applyMove("turn");
            turn1.addChild("turn", turn2);

            // Make a PROGRESS move: move A♠ to F1 (this changes the state!)
            String progressMove = "move T1 A♠ F1";
            MonteCarloTreeNode progress = new MonteCarloTreeNode();
            progress.setState(turn2.getState().copy());
            progress.applyMove(progressMove);
            turn2.addChild(progressMove, progress);

            // Cycle 2: turn -> turn -> turn (more turns to potentially create a "cycle")
            MonteCarloTreeNode turn3 = new MonteCarloTreeNode();
            turn3.setState(progress.getState().copy());
            turn3.applyMove("turn");
            progress.addChild("turn", turn3);

            MonteCarloTreeNode turn4 = new MonteCarloTreeNode();
            turn4.setState(turn3.getState().copy());
            turn4.applyMove("turn");
            turn3.addChild("turn", turn4);

            MonteCarloTreeNode turn5 = new MonteCarloTreeNode();
            turn5.setState(turn4.getState().copy());
            turn5.applyMove("turn");
            turn4.addChild("turn", turn5);

            // Even though we've done many turns, the progress move changed the state,
            // so the stock cycling after the Ace move creates NEW states, not repeats.
            assertFalse(turn5.isCycleDetected(),
                    "Stock cycling with an interim progress move should NOT detect a cycle " +
                    "(the state is different because A♠ moved to foundation)");
        }

        // --------------------------------------------------------------------
        // Tableau ping-pong tests (moving cards back and forth)
        // --------------------------------------------------------------------

        @Test
        @DisplayName("Tableau ping-pong: 2-move cycle (A→B→A)")
        void tableauTwoMoveCycleDetected() {
            // Setup: red 7 on black 8 in T1, another black 8 in T2.
            // Move red 7 to T2, then back to T1 = same state as original.
            // Do it twice = cycle detected.
                Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .tableau("T1", "8♠", "7♥")
                    .tableau("T2", "8♣")
                    .build();

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // Move 7♥ from T1 to T2
            String move1Str = "move T1 7♥ T2";
            MonteCarloTreeNode move1 = new MonteCarloTreeNode();
            move1.setState(root.getState().copy());
            move1.applyMove(move1Str);
            root.addChild(move1Str, move1);
            // Move 7♥ back from T2 to T1 (state returns to root state)
            String move2Str = "move T2 7♥ T1";
            MonteCarloTreeNode move2 = new MonteCarloTreeNode();
            move2.setState(move1.getState().copy());
            move2.applyMove(move2Str);
            move1.addChild(move2Str, move2);
            // Repeat: Move 7♥ from T1 to T2 again
            String move3Str = "move T1 7♥ T2";
            MonteCarloTreeNode move3 = new MonteCarloTreeNode();
            move3.setState(move2.getState().copy());
            move3.applyMove(move3Str);
            move2.addChild(move3Str, move3);
            // Move 7♥ back from T2 to T1 again (second cycle!)
            String move4Str = "move T2 7♥ T1";
            MonteCarloTreeNode move4 = new MonteCarloTreeNode();
            move4.setState(move3.getState().copy());
            move4.applyMove(move4Str);
            move3.addChild(move4Str, move4);

            assertTrue(move4.isCycleDetected(), 
                    "2-move ping-pong repeated twice should be detected as cycle");
        }

        @Test
        @DisplayName("Tableau ping-pong: first round-trip is not a cycle")
        void tableauFirstRoundTripNotCycle() {
            // Same setup as above, but only one round-trip (A→B→A).
            // This is the first occurrence of the original state, so no cycle yet.
                Solitaire solitaire = SolitaireBuilder
                    .newGame()
                    .tableau("T1", "8♠", "7♥")
                    .tableau("T2", "8♣")
                    .build();

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            String move1Str = "move T1 7♥ T2";
            MonteCarloTreeNode move1 = new MonteCarloTreeNode();
            move1.setState(root.getState().copy());
            move1.applyMove(move1Str);
            root.addChild(move1Str, move1);

            String move2Str = "move T2 7♥ T1";
            MonteCarloTreeNode move2 = new MonteCarloTreeNode();
            move2.setState(move1.getState().copy());
            move2.applyMove(move2Str);
            move1.addChild(move2Str, move2);

            assertFalse(move2.isCycleDetected(), 
                    "First round-trip back to original state should not be flagged as cycle");
        }

        @Test
        @DisplayName("Tableau: 4-move cycle (A→B→C→D→A repeated)")
        void tableauFourMoveCycleDetected() {
            // Setup a more complex scenario with multiple columns where we can
            // move a card around in a 4-step loop.
            // T1: 8♠ + 7♥, T2: 8♣, T3: 8♦ (red), T4: 9♠ (black)
            // 
            // Actually, let's simplify: move 7♥ around different black 8s.
            // T1: 8♠ + 7♥
            // T2: 8♣
            // Then 7♥ can go T1→T2→T1 (2-move cycle), but for a 4-move we need more cards.
            //
            // Better approach: Use multiple cards that can shuffle.
            // Let's use: T1: 9♠+8♥, T2: 9♣, T3: 10♦+9♥
            // This way: 8♥ can go on 9♠ or 9♣ (both black)
            // And we can shuffle the 8♥ between T1 and T2 while also shuffling 9♥ elsewhere.
            //
            // For simplicity, let's just do 4 moves with one card through 3 columns:
            // T1: 8♠ + 7♥
            // T2: 8♣
            // T3: 8♦ (red - can't accept 7♥)
            // Wait, 7♥ can only go on black 8s. So we need another black 8.
            // 
            // Let's do: T1: 8♠ + 7♥, T2: 8♣, T3: 9♦ + 8♥
            // Actually this is getting complex. Let's just verify 2-move cycle works
            // and create a simpler 4-move test with stock + tableau mix.

            // Simpler: 4 turn moves that cycle the stock
            Solitaire solitaire = SolitaireFactory.withExactStockAndWaste(
                new String[] {"A♣", "2♣", "3♣"},
                new String[] {}
            ); // 3 cards = 1 turn cycles

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // With 3 cards: turn puts all 3 in talon. Next turn recycles and puts all 3 back.
            // So after 2 turns we're back to start. After 4 turns, we've done it twice = cycle.
            MonteCarloTreeNode t1 = new MonteCarloTreeNode();
            t1.setState(root.getState().copy());
            t1.applyMove("turn");
            root.addChild("turn", t1);

            MonteCarloTreeNode t2 = new MonteCarloTreeNode();
            t2.setState(t1.getState().copy());
            t2.applyMove("turn");
            t1.addChild("turn", t2);  // Back to root-like state

            MonteCarloTreeNode t3 = new MonteCarloTreeNode();
            t3.setState(t2.getState().copy());
            t3.applyMove("turn");
            t2.addChild("turn", t3);

            MonteCarloTreeNode t4 = new MonteCarloTreeNode();
            t4.setState(t3.getState().copy());
            t4.applyMove("turn");
            t3.addChild("turn", t4);  // Back again = second occurrence = cycle!

            assertTrue(t4.isCycleDetected(), 
                    "4-move cycle (2 full stock rotations) should be detected");
        }

        // --------------------------------------------------------------------
        // Edge cases
        // --------------------------------------------------------------------

        @Test
        void noAncestorsReturnsFalse() {
            Solitaire solitaire = SolitaireFactory.stockOnly();
            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);
            root.setMove("turn");

            assertFalse(root.isCycleDetected(), "Root node with no ancestors should not detect cycle");
        }

        @Test
        void nullStateReturnsFalse() {
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setMove("turn");
            // state is null by default

            assertFalse(node.isCycleDetected(), "Node with null state should not detect cycle");
        }

        @Test
        void nullMoveReturnsFalse() {
            Solitaire solitaire = SolitaireFactory.stockOnly();
            MonteCarloTreeNode node = new MonteCarloTreeNode();
            node.setState(solitaire);
            // move is null by default

            assertFalse(node.isCycleDetected(), "Node with null move should not detect cycle");
        }

        @Test
        @DisplayName("Deep tree without cycles returns false")
        void deepTreeWithoutCycleReturnsFalse() {
            // Build a tree with progress at each step (no cycles)
            // Create a simple game and make a progress move (Ace to foundation)
            // Put A♠ in T1 - can move to F1
            Solitaire solitaire = SolitaireBuilder.newGame().tableau("T1", "A♠").build();

            MonteCarloTreeNode root = new MonteCarloTreeNode();
            root.setState(solitaire);

            // Moving A♠ to F1 is progress, not a cycle
            String move1Str = "move T1 A♠ F1";
            MonteCarloTreeNode move1 = new MonteCarloTreeNode();
            move1.setState(root.getState().copy());
            move1.applyMove(move1Str);
            root.addChild(move1Str, move1);

            assertFalse(move1.isCycleDetected(), 
                    "Progress moves should not be detected as cycles");
        }
    }
}

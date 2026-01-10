package ai.games.player.ai;

import static org.junit.jupiter.api.Assertions.*;

import ai.games.game.Deck;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.Player;
import ai.games.player.ai.mcts.MonteCarloPlayer;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.FoundationCountHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import ai.games.unit.helpers.SolitaireFactory;
import ai.games.player.LegalMovesHelper;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

/**
 * Basic behaviour tests for {@link MonteCarloPlayer}:
 * - improves simple nearly-won setups
 * - does not loop indefinitely on random games.
 */
class MonteCarloPlayerTest {

    private static final int MAX_TEST_STEPS = 2000;

    @Test
    void monteCarloAdvancesOnSimpleNearlyWonGame() {
        Solitaire solitaire = SolitaireFactory.oneMoveFromWin();
        Player ai = new MonteCarloPlayer();

        int startFoundation = FoundationCountHelper.totalFoundation(solitaire);

        for (int i = 0; i < 10 && FoundationCountHelper.totalFoundation(solitaire) < 52; i++) {
            String command = ai.nextCommand(solitaire, "", "");
            applyCommand(solitaire, command);
        }

        int endFoundation = FoundationCountHelper.totalFoundation(solitaire);
        assertTrue(endFoundation > startFoundation, "Monte Carlo should improve foundation count on simple setup");
    }

    @Test
    @Disabled("Disabled: can be long-running/non-deterministic; keep unit tests fast")
    void monteCarloDoesNotLoopForeverOnRandomGame() {
        Solitaire solitaire = new Solitaire(new Deck());
        Player ai = new MonteCarloPlayer();

        int steps = 0;
        while (!isTerminal(solitaire) && steps < MAX_TEST_STEPS) {
            String command = ai.nextCommand(solitaire, "", "");
            assertNotNull(command, "Monte Carlo player should always return a command until game exits");
            if ("quit".equalsIgnoreCase(command.trim())) {
                break;
            }
            applyCommand(solitaire, command);
            steps++;
        }

        assertTrue(steps < MAX_TEST_STEPS, "Monte Carlo player should not run indefinitely on a random game");
    }

    @Test
    void reusesTreeAndAdvancesCurrentWhileRootStaysFixed() throws Exception {
        Solitaire solitaire = SolitaireFactory.twoMovesFromWin();
        MonteCarloPlayer ai = new MonteCarloPlayer();

        String firstMove = ai.nextCommand(solitaire, "", "");
        assertNotNull(firstMove);
        applyCommand(solitaire, firstMove);

        NodeSnapshot snapAfterFirst = snapshot(ai);

        String secondMove = ai.nextCommand(solitaire, "", "");
        assertNotNull(secondMove);

        NodeSnapshot snapAfterSecond = snapshot(ai);

        assertSame(snapAfterFirst.root, snapAfterSecond.root, "Root should remain anchored to initial state");
        assertNotSame(snapAfterFirst.current, snapAfterSecond.current, "Current should advance after a move");
        assertEquals(firstMove.trim(), snapAfterFirst.currentMove, "Current node should reflect the chosen move");
    }

    @Test
    void throwsWhenMoveNotAppliedBetweenTurns() {
        Solitaire solitaire = SolitaireFactory.twoMovesFromWin();
        MonteCarloPlayer ai = new MonteCarloPlayer();

        String firstMove = ai.nextCommand(solitaire, "", "");
        assertNotNull(firstMove);

        assertThrows(IllegalStateException.class, () -> ai.nextCommand(solitaire, "", ""),
                "Expected state drift when previous move is not applied");
    }

    @Test
    void throwsWhenDifferentMoveAppliedBetweenTurns() {
        Solitaire solitaire = SolitaireFactory.twoMovesFromWin();
        MonteCarloPlayer ai = new MonteCarloPlayer();

        String chosenMove = ai.nextCommand(solitaire, "", "");
        assertNotNull(chosenMove);

        List<String> legal = LegalMovesHelper.listLegalMoves(solitaire);
        legal.removeIf(m -> m.equalsIgnoreCase("quit") || m.equalsIgnoreCase(chosenMove));
        if (legal.isEmpty()) {
            fail("Test setup requires an alternative legal move to trigger drift");
        }
        String differentMove = legal.get(0);

        applyCommand(solitaire, differentMove);

        assertThrows(IllegalStateException.class, () -> ai.nextCommand(solitaire, "", ""),
                "Applying a different move should trigger state drift detection");
    }

    @Test
    void acceptsPlanModeAndClearsExpectedKeyOnValidation() {
        Solitaire solitaire = SolitaireFactory.twoMovesFromWin();
        solitaire.setMode(Solitaire.GameMode.PLAN);
        MonteCarloPlayer ai = new MonteCarloPlayer();

        String firstMove = assertDoesNotThrow(() -> ai.nextCommand(solitaire, "", ""));
        assertNotNull(firstMove);
        applyCommand(solitaire, firstMove);

        assertDoesNotThrow(() -> ai.nextCommand(solitaire, "", ""),
                "Second call after applying move should validate and clear expected key");
    }

    @Test
        void kingToFoundationIsNotUselessInLoggedGameState() throws Exception {
        Solitaire solitaire = seedLoggedGameStateMove249();

        MonteCarloTreeNode parent = new MonteCarloTreeNode();
        parent.setState(solitaire.copy());

        MonteCarloTreeNode kingToFoundation = new MonteCarloTreeNode();
        kingToFoundation.setParent(parent);
        kingToFoundation.setState(solitaire.copy());
        kingToFoundation.applyMove("move T6 K♠ F3");

        assertFalse(
            kingToFoundation.isUselessKingMove(),
            "King to foundation should never be classified as a useless king move");

        // And verify MonteCarloPlayer.simulate() does not apply the -1 sentinel.
        MonteCarloPlayer ai = new MonteCarloPlayer();
        Method simulate = MonteCarloPlayer.class.getDeclaredMethod("simulate", MonteCarloTreeNode.class);
        simulate.setAccessible(true);

        double reward = (double) simulate.invoke(ai, kingToFoundation);
        assertNotEquals(
            -1.0,
            reward,
            "King to foundation should not get the -1 sentinel penalty");
        }

        @Test
        void kingShuffleTableauToTableauIsUselessInLoggedGameState() throws Exception {
        Solitaire solitaire = seedLoggedGameStateMove249();

        MonteCarloTreeNode parent = new MonteCarloTreeNode();
        parent.setState(solitaire.copy());

        // Useless king move: moving K♦ from a 0-facedown, single-card pile to an empty tableau.
        MonteCarloTreeNode uselessKingShuffle = new MonteCarloTreeNode();
        uselessKingShuffle.setParent(parent);
        uselessKingShuffle.setState(solitaire.copy());
        uselessKingShuffle.applyMove("move T2 K♦ T7");

        assertTrue(
            uselessKingShuffle.isUselessKingMove(),
            "King shuffle T→T with 0 facedowns (even when source becomes empty) should be classified as useless");

        // And verify MonteCarloPlayer.simulate() applies the -1 sentinel.
        MonteCarloPlayer ai = new MonteCarloPlayer();
        Method simulate = MonteCarloPlayer.class.getDeclaredMethod("simulate", MonteCarloTreeNode.class);
        simulate.setAccessible(true);

        double reward = (double) simulate.invoke(ai, uselessKingShuffle);
        assertEquals(
            -1.0,
            reward,
            0.0,
            "Useless king moves should be hard-penalised to -1 sentinel");
        }

    /**
     * Recreates the board position from the provided log (GAME 1 MOVE 249).
     * Foundations: F1=Q♦, F2=K♣, F3=Q♠, F4=K♥.
     * Tableau: T2 has K♦, T6 has K♠, all other piles empty. Stock/talon empty.
     */
    private Solitaire seedLoggedGameStateMove249() {
        return SolitaireBuilder
                .newGame()
                .foundation("F1", SolitaireFactory.foundationUpTo(Suit.DIAMONDS, Rank.QUEEN))
                .foundation("F2", SolitaireFactory.foundationUpTo(Suit.CLUBS, Rank.KING))
                .foundation("F3", SolitaireFactory.foundationUpTo(Suit.SPADES, Rank.QUEEN))
                .foundation("F4", SolitaireFactory.foundationUpTo(Suit.HEARTS, Rank.KING))
                .tableau("T2", "K♦")
                .tableau("T6", "K♠")
                .stock()
                .waste()
                .build();
    }

    private boolean isTerminal(Solitaire solitaire) {
        return FoundationCountHelper.totalFoundation(solitaire) == 52
                || (solitaire.getStockpile().isEmpty()
                && solitaire.getTalon().isEmpty());
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

    private NodeSnapshot snapshot(MonteCarloPlayer ai) throws Exception {
        Field rootField = MonteCarloPlayer.class.getDeclaredField("root");
        rootField.setAccessible(true);
        Object root = rootField.get(ai);

        Field currentField = MonteCarloPlayer.class.getDeclaredField("current");
        currentField.setAccessible(true);
        Object current = currentField.get(ai);

        Field moveField = current.getClass().getSuperclass().getDeclaredField("move");
        moveField.setAccessible(true);
        String move = (String) moveField.get(current);

        return new NodeSnapshot(root, current, move);
    }

    private record NodeSnapshot(Object root, Object current, String currentMove) {}
}

package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.alpha.AlphaSolitaireRequest;
import ai.games.player.ai.mcts.MonteCarloTreeNode;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class StructuredMovesCompatibilityTest {
    // Covers all five move categories, turn/quit, and implicit waste/foundation cards.
    private static final List<String> COMMANDS = List.of(
            "move T1 A♣ F4", "move T5 3♦ F1",
            "move T2 4♣ T3", "move T4 K♠ T6", "move T4 K♠ T7", "move T5 3♦ T2",
            "move W F3", "move W T5", "move F2 T2", "turn", "quit");

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void structuredAndTextApisPreserveCommandsOrderAndMutability(Solitaire.GameMode mode) {
        Solitaire state = position(mode);
        List<String> commands = LegalMovesHelper.listLegalMoves(state);
        List<Move> moves = LegalMovesHelper.listMoves(state);

        assertEquals(COMMANDS, commands);
        assertEquals(COMMANDS, moves.stream().map(Move::toCommandString).toList());
        assertEquals(COMMANDS.stream().map(Move::parse).toList(), moves);
        // Existing search players remove quit or reorder the returned list.
        commands.remove("quit");
        moves.removeIf(Move::isQuit);
        assertEquals(10, commands.size());
        assertEquals(10, moves.size());
        assertEquals(COMMANDS, LegalMovesHelper.listLegalMoves(state));
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void structuredApplicationMatchesStringApplication(Solitaire.GameMode mode) {
        Solitaire state = position(mode);
        for (Move move : LegalMovesHelper.listMoves(state)) {
            MonteCarloTreeNode textNode = new MonteCarloTreeNode(state.copy());
            MonteCarloTreeNode structuredNode = new MonteCarloTreeNode(state.copy());

            textNode.applyMove(move.toCommandString());
            structuredNode.applyMove(move);

            assertSame(move, structuredNode.getMove());
            assertEquals(textNode.getMove(), structuredNode.getMove());
            assertEquals(textNode.getStateKey(), structuredNode.getStateKey());
            assertEquals(textNode.getState().getTableau(), structuredNode.getState().getTableau());
            assertEquals(textNode.getState().getTableauFaceUpCounts(), structuredNode.getState().getTableauFaceUpCounts());
            assertEquals(textNode.getState().getFoundation(), structuredNode.getState().getFoundation());
            assertEquals(textNode.getState().getTalon(), structuredNode.getState().getTalon());
            assertEquals(textNode.getState().getStockpile(), structuredNode.getState().getStockpile());
        }
    }

    @Test
    void textApplicationKeepsPermissiveParsingAndInvalidCommandBehaviour() {
        Solitaire state = position(Solitaire.GameMode.GAME);
        MonteCarloTreeNode node = new MonteCarloTreeNode(state);
        node.applyMove("  MOVE   t1   ac   f4  ");
        assertEquals(List.of(new Card(Rank.ACE, Suit.CLUBS)), state.getFoundation().get(3));
        long key = node.getStateKey();

        node.applyMove("not a command");
        assertNull(node.getMove());
        assertEquals(key, node.getStateKey());
        assertEquals(key, state.getStateKey());
        assertThrows(IllegalArgumentException.class, () -> node.applyMove((String) null));
        assertThrows(IllegalArgumentException.class, () -> node.applyMove((Move) null));
    }

    @Test
    void unresolvedUnknownSourcesAreNotOfferedAsPlayableCards() {
        Solitaire state = spy(position(Solitaire.GameMode.PLAN));
        List<List<Card>> tableau = new ArrayList<>(state.getVisibleTableau());
        tableau.set(0, List.of(new Card(Rank.UNKNOWN, Suit.UNKNOWN)));
        tableau.set(1, List.of(new Card(Rank.ACE, Suit.HEARTS)));
        doReturn(tableau).when(state).getVisibleTableau();
        doReturn(tableau).when(state).getTableau();

        assertFalse(LegalMovesHelper.listLegalMoves(state).contains("move T1 ?? T2"));
        assertTrue(LegalMovesHelper.listMoves(state).stream()
                .noneMatch(move -> move.card() != null && move.card().rank() == Rank.UNKNOWN));
    }

    @Test
    void alphaServiceRequestStillContainsTextCommands() {
        AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(position(Solitaire.GameMode.GAME));
        assertEquals(COMMANDS, request.getLegalMoves());
    }

    @Test
    void nullStateKeepsEmptyLists() {
        assertTrue(LegalMovesHelper.listLegalMoves(null).isEmpty());
        assertTrue(LegalMovesHelper.listMoves(null).isEmpty());
    }

    private Solitaire position(Solitaire.GameMode mode) {
        Solitaire state = SolitaireBuilder.newGame()
                .tableau("T1", "A♣").tableau("T2", "4♣").tableau("T3", "5♥")
                .tableau("T4", "K♠").tableau("T5", "3♦")
                .foundation("F1", "A♦", "2♦")
                .foundation("F2", "A♥", "2♥", "3♥")
                .foundation("F3", "A♠").waste("2♠").build();
        if (mode == Solitaire.GameMode.PLAN) state = state.planningCopy();
        return state;
    }
}

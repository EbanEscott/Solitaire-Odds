package ai.games.unit.game;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import ai.games.game.Card;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.game.UnknownCardGuess;
import ai.games.game.moves.GameMovesHelper;
import ai.games.game.moves.MovesHelper;
import ai.games.game.moves.PlanningMovesHelper;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class TableauMoveGenerationTest {
    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void visibleStackMovesKeepSourceCardAndDestinationOrder(Solitaire.GameMode mode) {
        Solitaire state = SolitaireBuilder.newGame()
                .tableau("T1", 3, "3♣", "8♥", "7♣", "6♦")
                .tableau("T2", "9♠").tableau("T3", "9♣")
                .tableau("T4", "8♦").tableau("T5", "7♠").build();
        state.setMode(mode);

        assertEquals(List.of(
                "move T1 8♥ T2", "move T1 8♥ T3",
                "move T1 7♣ T4", "move T1 6♦ T5",
                "move T4 8♦ T2", "move T4 8♦ T3",
                "move T5 7♠ T4", "turn", "quit"), LegalMovesHelper.listLegalMoves(state));
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void emptyDestinationsAcceptBothKingColoursOnly(Solitaire.GameMode mode) {
        Solitaire state = kingsAndQueen();
        state.setMode(mode);

        assertEquals(List.of(
                "move T1 K♥ T5", "move T1 K♥ T6", "move T1 K♥ T7",
                "move T2 K♠ T5", "move T2 K♠ T6", "move T2 K♠ T7",
                "move T3 Q♦ T2", "move T4 J♣ T3", "turn", "quit"),
                LegalMovesHelper.listLegalMoves(state));
    }

    @ParameterizedTest
    @EnumSource(Solitaire.GameMode.class)
    void reusedHelperReadsChangedDestinationsOnNextCall(Solitaire.GameMode mode) {
        Solitaire state = kingsAndQueen();
        state.setMode(mode);
        MovesHelper helper = mode == Solitaire.GameMode.PLAN
                ? new PlanningMovesHelper() : new GameMovesHelper();
        assertTrue(helper.listLegalMoves(state).contains("move T4 J♣ T3"));

        assertTrue(state.moveCard("T3", "Q♦", "T2"));

        List<String> moves = helper.listLegalMoves(state);
        assertTrue(moves.contains("move T4 J♣ T2"));
        assertFalse(moves.contains("move T4 J♣ T3"));
        assertTrue(moves.contains("move T1 K♥ T3"));
    }

    @Test
    void planModeKeepsUnknownGuessesSeparateAndAfterVisibleDestinations() {
        Solitaire state = spy(SolitaireBuilder.newGame()
                .tableau("T1", "4♥").tableau("T2", "8♣").tableau("T3", "5♠").build());
        state.setMode(Solitaire.GameMode.PLAN);
        Card unknown = new Card(Rank.UNKNOWN, Suit.UNKNOWN);
        // Expose an UNKNOWN top without changing the normal piles used by the builder.
        List<List<Card>> tableau = new ArrayList<>(state.getVisibleTableau());
        tableau.set(1, List.of(unknown));
        doReturn(tableau).when(state).getVisibleTableau();
        doReturn(tableau).when(state).getTableau();
        UnknownCardGuess guess = new UnknownCardGuess(unknown, List.of(new Card(Rank.FIVE, Suit.CLUBS)));
        state.getUnknownCardGuesses().put(unknown, guess);

        assertEquals(List.of("move T1 4♥ T3", "move T1 4♥ T2", "turn", "quit"),
                LegalMovesHelper.listLegalMoves(state));
        assertSame(guess, state.getUnknownCardGuesses().get(unknown));
    }

    private Solitaire kingsAndQueen() {
        return SolitaireBuilder.newGame().tableau("T1", "K♥").tableau("T2", "K♠")
                .tableau("T3", "Q♦").tableau("T4", "J♣").build();
    }
}

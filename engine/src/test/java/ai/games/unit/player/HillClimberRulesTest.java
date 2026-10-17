package ai.games.unit.player;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.HillClimberPlayer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Controlled immediate neighborhoods verify local selection without random game outcomes. */
class HillClimberRulesTest {
    /** Every legal neighbor is copied once, and the strongest improvement wins. */
    @Test
    void evaluatesEachNeighborOnce() {
        /** Face-up counts provide distinct scores without changing the production evaluator. */
        try (Neighborhood fixture = new Neighborhood()) {
            Solitaire root = board(1, 1, 0, 0);
            fixture.connect(root, List.of("move T1 F1", "move T2 F1"),
                    board(2, 2, 0, 0), board(3, 3, 0, 0));
            assertEquals("move T2 F1", new HillClimberPlayer(42).nextCommand(root, "", ""));
            verify(root, times(2)).copy();
        }
    }

    /** The first plateau step must prefer an equal-score neighbor over a downhill move. */
    @Test
    void prefersSidewaysBeforeDownhillPerturbation() {
        /** A strict plateau preference makes this independent of the random seed. */
        try (Neighborhood fixture = new Neighborhood()) {
            Solitaire root = board(1, 2, 0, 0);
            fixture.connect(root, List.of("move T1 T2", "move F1 T3"),
                    board(2, 2, 0, 0), board(3, 1, 0, 0));
            assertEquals("move T1 T2", new HillClimberPlayer(42).nextCommand(root, "", ""));
        }
    }

    /** A valid downhill escape must not be missed among many rejected self-loop actions. */
    @Test
    void checksAllNeighborsBeforeQuitting() {
        /** Twelve rejected actions would make bounded sampling with replacement unreliable. */
        try (Neighborhood fixture = new Neighborhood()) {
            Solitaire root = board(1, 2, 0, 0);
            List<String> commands = new ArrayList<>();
            List<Solitaire> successors = new ArrayList<>();
            for (int source = 1; source <= 6; source++) {
                /** Two distinct legal command shapes lead back to the same observed key. */
                for (int destination = 1; destination <= 2; destination++) {
                    commands.add("move T" + source + " F" + destination);
                    successors.add(board(1, 2, 0, 0));
                }
            }
            commands.add("turn");
            successors.add(board(2, 1, 0, 0));
            fixture.connect(root, commands, successors.toArray(Solitaire[]::new));
            assertEquals("turn", new HillClimberPlayer(42).nextCommand(root, "", ""));
            verify(root, times(13)).copy();
        }
    }

    /** Drawing and recycling are neutral when only stock/waste allocation changes. */
    @Test
    void stockRelocationDoesNotCreateArtificialImprovement() {
        /** Reset identical seeded players: moving six cards either direction must preserve tie behavior. */
        try (Neighborhood fixture = new Neighborhood()) {
            Solitaire draw = board(1, 1, 6, 0);
            Solitaire recycle = board(4, 1, 0, 6);
            fixture.connect(draw, List.of("turn", "move T1 T2"),
                    board(2, 1, 0, 6), board(3, 1, 6, 0));
            fixture.connect(recycle, List.of("turn", "move T1 T2"),
                    board(5, 1, 6, 0), board(6, 1, 0, 6));
            /** Seed variation exercises reproducible stochastic choices. */
            for (int seed = 0; seed < 20; seed++) {
                assertEquals(new HillClimberPlayer(seed).nextCommand(draw, "", ""),
                        new HillClimberPlayer(seed).nextCommand(recycle, "", ""));
            }
        }
    }

    /** Repeated real positions are rejected, and explicit reset starts a fresh exploration. */
    @Test
    void tracksRealHistoryAndResetsIndependently() {
        /** A one-edge reversal is legal but returns to an observation already visited. */
        try (Neighborhood fixture = new Neighborhood()) {
            Solitaire root = board(1, 1, 0, 0);
            Solitaire next = board(2, 2, 0, 0);
            fixture.connect(root, List.of("move T1 T2"), next);
            fixture.connect(next, List.of("move T2 T1"), root);
            HillClimberPlayer player = new HillClimberPlayer(42);
            assertEquals("move T1 T2", player.nextCommand(root, "", ""));
            assertEquals("quit", player.nextCommand(next, "", ""));
            assertEquals("move T1 T2", new HillClimberPlayer(42).nextCommand(root, "", ""));
            player.resetGame();
            assertEquals("move T1 T2", player.nextCommand(root, "", ""));
        }
    }

    /** A long stock-like plateau must continue past eight sideways moves rather than falsely quit. */
    @Test
    void perturbationContinuesWhenOnlyNeutralNeighborsExist() {
        /** Unique observations model discovering cards during a draw-one stock pass. */
        try (Neighborhood fixture = new Neighborhood()) {
            HillClimberPlayer player = new HillClimberPlayer(42);
            /** Long neutral passages must remain eligible within the global action budget. */
            for (int step = 1; step <= 50; step++) {
                Solitaire root = board(step, 1, 6, 0);
                fixture.connect(root, List.of("turn"), board(step + 1, 1, 3, 3));
                assertEquals("turn", player.nextCommand(root, "", ""));
            }
        }
    }

    /** After eight preferred sideways steps the escape pool includes downhill neighbors. */
    @Test
    void sidewaysPreferenceIsBounded() {
        /** Across fixed seeds, at least one ninth choice must use the downhill escape. */
        boolean escapedDownhill = false;
        try (Neighborhood fixture = new Neighborhood()) {
            /** Seed variation exercises reproducible stochastic choices. */
            for (int seed = 0; seed < 10; seed++) {
                HillClimberPlayer player = new HillClimberPlayer(seed);
                /** Model successive neutral real observations before the escape choice. */
                for (int step = 1; step <= 9; step++) {
                    Solitaire root = board(step, 2, 0, 0);
                    fixture.connect(root, List.of("move T1 T2", "move F1 T3"),
                            board(step + 1, 2, 0, 0), board(100 + step, 1, 0, 0));
                    String command = player.nextCommand(root, "", "");
                    /** Earlier plateau steps must remain neutral; the ninth may perturb downhill. */
                    if (step <= 8) {
                        assertEquals("move T1 T2", command);
                    } else {
                        escapedDownhill |= command.equals("move F1 T3");
                    }
                }
            }
        }
        assertTrue(escapedDownhill, "The sideways budget must eventually permit downhill exploration");
    }

    /** Creates a score-controlled planning board without simulating physical card movement. */
    private Solitaire board(long key, int visible, int stock, int waste) {
        /** Keys distinguish observed positions while counts provide the real evaluator's inputs. */
        Solitaire board = mock(Solitaire.class);
        when(board.planningCopy()).thenReturn(board);
        when(board.getStateKey()).thenReturn(key);
        when(board.getFoundation()).thenReturn(List.of());
        when(board.getTableauFaceUpCounts()).thenReturn(List.of(visible));
        when(board.getTableauFaceDownCounts()).thenReturn(List.of(0));
        Card unknown = new Card(Rank.UNKNOWN, Suit.UNKNOWN);
        when(board.getStockpile()).thenReturn(Collections.nCopies(stock, unknown));
        when(board.getTalon()).thenReturn(Collections.nCopies(waste, unknown));
        return board;
    }

    /** Supplies fixed immediate neighbors and restores normal legal generation after each test. */
    private static final class Neighborhood implements AutoCloseable {
        /** Thread-scoped override for controlled legal move lists. */
        private final MockedStatic<LegalMovesHelper> legal = mockStatic(LegalMovesHelper.class);

        /** Connects ordered commands to precomputed copied successors without deeper expansion. */
        private void connect(Solitaire root, List<String> commands, Solitaire... successors) {
            /** Cycle copy responses so repeated seeded decisions see the same neighborhood. */
            AtomicInteger index = new AtomicInteger();
            when(root.copy()).thenAnswer(ignored -> successors[index.getAndIncrement() % successors.length]);
            legal.when(() -> LegalMovesHelper.listMoves(root))
                    .thenReturn(commands.stream().map(Move::parse).toList());
        }

        /** Releases the scoped override even if a test assertion fails. */
        @Override
        public void close() {
            legal.close();
        }
    }
}

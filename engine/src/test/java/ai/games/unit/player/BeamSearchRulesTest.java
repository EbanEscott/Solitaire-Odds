package ai.games.unit.player;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.player.ai.BeamSearchPlayer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.mockito.MockedStatic;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Synthetic search graphs isolate beam semantics from Solitaire heuristic tuning. */
class BeamSearchRulesTest {
    /** Beam width must not truncate legal generation before successors have been scored. */
    @Test
    void ranksEverySuccessorBeforePruning() {
        /** With width one the old two-successor cutoff missed the third, best action. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            root.edge("move T1 F1", graph.vertex(101));
            root.edge("move T2 F1", graph.vertex(102));
            root.edge("move W F1", graph.vertex(130));
            assertEquals("move W F1", new BeamSearchPlayer(1, 1, 42).nextCommand(graph.board(root), "", ""));
        }
    }

    /** Below-root scores still select the best searched path, never a random fallback. */
    @Test
    void choosesBestDecliningContinuation() {
        /** Distinct scores make the answer independent of tie-breaking seeds. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            root.edge("move T1 F1", graph.vertex(99));
            root.edge("move T2 F1", graph.vertex(80));
            /** Changing tie randomness cannot change a strict score preference. */
            for (int seed = 0; seed < 10; seed++) {
                assertEquals("move T1 F1", new BeamSearchPlayer(1, 2, seed).nextCommand(graph.board(root), "", ""));
            }
        }
    }

    /** A neutral setup can lead to the best surviving depth-two result. */
    @Test
    void selectsSetupFromFinalBeam() {
        /** Both setup branches survive the first layer so their continuations can compete. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex setup = graph.vertex(98);
            Vertex distraction = graph.vertex(110);
            root.edge("move T1 T2", setup);
            root.edge("move T3 F1", distraction);
            setup.edge("move T2 F1", graph.vertex(150));
            distraction.edge("move T4 F1", graph.vertex(111));
            assertEquals("move T1 T2", new BeamSearchPlayer(2, 2, 42).nextCommand(graph.board(root), "", ""));
        }
    }

    /** Foundation withdrawals compete by the same search rules as every other legal action. */
    @Test
    void searchesWorryBack() {
        /** A withdrawal setup exposes a better second-ply continuation. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex setup = graph.vertex(80);
            root.edge("move F1 T1", setup);
            root.edge("turn", graph.vertex(101));
            setup.edge("move T2 F2", graph.vertex(150));
            assertEquals("move F1 T1", new BeamSearchPlayer(2, 2, 42).nextCommand(graph.board(root), "", ""));
        }
    }

    /** A duplicate endpoint must retain the route with lower cumulative path costs. */
    @Test
    void keepsBestDuplicatePath() {
        /** The turn branch is expanded first, but reaches the same endpoint at a higher cost. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex drawn = graph.vertex(110);
            Vertex moved = graph.vertex(100);
            Vertex shared = graph.vertex(140);
            root.edge("turn", drawn);
            root.edge("move T1 T2", moved);
            drawn.edge("move T3 F1", shared);
            moved.edge("move T4 F1", shared);
            assertEquals("move T1 T2", new BeamSearchPlayer(2, 2, 42).nextCommand(graph.board(root), "", ""));
        }
    }

    /** Generating a subsequently pruned state must not ban it from a later retained path. */
    @Test
    void prunedStatesDoNotPoisonLaterLayers() {
        /** The third root child is pruned, then reached through a surviving setup branch. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex first = graph.vertex(120);
            Vertex second = graph.vertex(110);
            Vertex pruned = graph.vertex(105);
            root.edge("move T1 T2", first);
            root.edge("move T3 T4", second);
            root.edge("move T5 T6", pruned);
            first.edge("move T2 F1", graph.vertex(0));
            second.edge("move T4 F1", pruned);
            pruned.edge("move T6 F1", graph.vertex(200));
            assertEquals("move T3 T4", new BeamSearchPlayer(3, 2, 42).nextCommand(graph.board(root), "", ""));
        }
    }

    /** Search must stop at unknown observations while retaining them as selectable leaves. */
    @Test
    void doesNotExpandUnknownObservation() {
        /** The artificial lucrative move beyond the unknown leaf must never be inspected. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex unknown = graph.vertex(100);
            unknown.unresolved = true;
            root.edge("turn", unknown);
            root.edge("move T1 T2", graph.vertex(110));
            unknown.edge("move W F1", graph.vertex(200));
            assertEquals("move T1 T2", new BeamSearchPlayer(3, 2, 42).nextCommand(graph.board(root), "", ""));
            assertEquals(0, unknown.expansions);
        }
    }

    /** Real decisions share cycle history, and reset restores an independent new game. */
    @Test
    void preventsReversalAcrossDecisions() {
        /** Depth one cannot see the loop until the next real observation arrives. */
        try (Graph graph = new Graph()) {
            Vertex root = graph.vertex(100);
            Vertex next = graph.vertex(110);
            root.edge("move T1 T2", next);
            next.edge("move T2 T1", root);
            BeamSearchPlayer player = new BeamSearchPlayer(1, 2, 42);
            assertEquals("move T1 T2", player.nextCommand(graph.board(root), "", ""));
            assertEquals("quit", player.nextCommand(graph.board(next), "", ""));
            player.resetGame();
            assertEquals("move T1 T2", player.nextCommand(graph.board(root), "", ""));
        }
    }

    /** A scored graph vertex with explicit ordered legal outgoing actions. */
    private static final class Vertex {
        /** Unique position identity used by beam history and duplicate detection. */
        private final long key;
        /** Synthetic state value encoded through stock size for the real evaluator. */
        private final int value;
        /** Outgoing actions in legal-generation order. */
        private final Map<String, Vertex> edges = new LinkedHashMap<>();
        /** Whether the vertex represents a newly exposed unknown card. */
        private boolean unresolved;
        /** Number of move-generation calls, used to enforce the observation boundary. */
        private int expansions;

        /** Creates a unique vertex with a controlled evaluation value. */
        private Vertex(long key, int value) {
            /** Vertex identity and evaluation are immutable across all board copies. */
            this.key = key;
            this.value = value;
        }

        /** Adds a synthetic legal transition without assuming physical card semantics. */
        private void edge(String command, Vertex target) {
            /** Preserve insertion order to expose generation-order cutoff regressions. */
            edges.put(command, target);
        }
    }

    /** Lightweight mocked planning boards for testing search control independently of game rules. */
    private static final class Graph implements AutoCloseable {
        /** Vertices indexed by their opaque position key. */
        private final List<Vertex> vertices = new ArrayList<>();
        /** Scoped legal-generation override, restored after each test. */
        private final MockedStatic<LegalMovesHelper> legal = mockStatic(LegalMovesHelper.class);

        /** Supplies each vertex's ordered synthetic legal moves to the production beam. */
        private Graph() {
            /** Route legal-move requests through the current copied board position. */
            legal.when(() -> LegalMovesHelper.listMoves(any(Solitaire.class))).thenAnswer(invocation -> {
                /** Count expansion so an unknown observation cannot silently be searched through. */
                Solitaire board = invocation.getArgument(0);
                Vertex vertex = vertices.get((int) board.getStateKey() - 1);
                vertex.expansions++;
                return vertex.edges.keySet().stream().map(Move::parse).toList();
            });
        }

        /** Registers a fresh position with the requested relative score. */
        private Vertex vertex(int value) {
            /** Sequential nonzero keys make lookup independent of heuristic values. */
            Vertex vertex = new Vertex(vertices.size() + 1, value);
            vertices.add(vertex);
            return vertex;
        }

        /** Returns a mutable board cursor whose copies move independently through the graph. */
        private Solitaire board(Vertex initial) {
            /** Mutable state belongs to this copy alone; planningCopy and copy create fresh cursors. */
            AtomicReference<Vertex> current = new AtomicReference<>(initial);
            Solitaire board = mock(Solitaire.class);
            when(board.planningCopy()).thenAnswer(ignored -> board(current.get()));
            when(board.copy()).thenAnswer(ignored -> board(current.get()));
            when(board.getStateKey()).thenAnswer(ignored -> current.get().key);
            when(board.getFoundation()).thenReturn(List.of());
            when(board.getTableauFaceUpCounts()).thenReturn(List.of(1));
            when(board.getTableauFaceDownCounts()).thenReturn(List.of(0));
            when(board.getStockpile()).thenAnswer(ignored -> Collections.nCopies(300 - current.get().value,
                    new Card(Rank.UNKNOWN, Suit.UNKNOWN)));
            when(board.nextUnresolvedCard()).thenAnswer(ignored -> current.get().unresolved
                    ? new Card(Rank.UNKNOWN, Suit.UNKNOWN) : null);
            when(board.turnStock()).thenAnswer(ignored -> {
                /** Only registered legal transitions may be executed. */
                current.set(current.get().edges.get("turn"));
                assertNotNull(current.get());
                return true;
            });
            when(board.moveCard(anyString(), nullable(String.class), anyString())).thenAnswer(invocation -> {
                /** Reconstruct the structured command to follow its explicit graph edge. */
                String card = invocation.getArgument(1);
                String command = "move " + invocation.getArgument(0) + (card == null ? "" : " " + card)
                        + " " + invocation.getArgument(2);
                current.set(current.get().edges.get(command));
                assertNotNull(current.get());
                return true;
            });
            return board;
        }

        /** Restores normal engine move generation after the synthetic graph test. */
        @Override
        public void close() {
            /** Release the thread-scoped static mock even when an assertion fails. */
            legal.close();
        }
    }
}

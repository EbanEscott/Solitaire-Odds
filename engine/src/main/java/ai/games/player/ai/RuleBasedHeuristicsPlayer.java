package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Deterministic ordered-rule player with no successor simulation or lookahead.
 *
 * <p>Begin with a stock observation when permitted, then prefer exposed foundation
 * moves, whole-run tableau reveals, waste deployment, and whole-run consolidation,
 * followed by a stock turn. Tableau ties scan T7 to T1; foundation destinations scan
 * F1 to F4. Moving a fully exposed king between empty columns is excluded.
 *
 * <p>Never repeat an action from the same observed position. Knowledge-aware state
 * keys allow another pass after discoveries or board changes, while unchanged stock
 * cycles eventually terminate. Finite pass limits remain the engine's responsibility.
 * Each game requires a fresh instance or an explicit {@link #resetGame()}.
 */
@Component
@Profile("ai-rule")
public class RuleBasedHeuristicsPlayer extends AIPlayer implements Player {
    /** Actions already selected at each observed position in the current game. */
    private final Map<Long, Set<String>> attemptedActions = new HashMap<>();

    /** Whether the initial stock-observation rule has been considered. */
    private boolean started;

    /** Constructs a player with independent history for one game. */
    public RuleBasedHeuristicsPlayer() {
        /** Field initializers provide empty action history and an unused opening rule. */
    }

    /** Clears instance history before reusing this player for another deal. */
    public void resetGame() {
        /** Reset the opening rule together with every remembered action. */
        attemptedActions.clear();
        started = false;
    }

    /** Selects the first applicable untried rule using observed identities only. */
    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        /** Remember actions against actual observations, never hypothetical hidden identities. */
        Solitaire board = solitaire.planningCopy();
        Set<String> attempted = attemptedActions.computeIfAbsent(board.getStateKey(), key -> new HashSet<>());
        List<Move> legal = LegalMovesHelper.listMoves(board);

        /** Preserve the baseline's opening stock observation when a turn is legal. */
        if (!started) {
            started = true;
            /** Respect finite pass limits even for the opening rule. */
            if (board.canTurnStock()) {
                attempted.add("turn");
                return "turn";
            }
        }

        /** Sort explicit rule categories, not evaluated successor scores. */
        legal.sort(Comparator.comparingInt((Move move) -> priority(board, move))
                .thenComparingInt(this::sourceOrder).thenComparingInt(this::destinationOrder));
        for (Move move : legal) {
            /** Ignore actions outside this rule policy and actions already tried at this observation. */
            if (priority(board, move) == Integer.MAX_VALUE) {
                continue;
            }
            String command = move.toCommandString();
            /** Record the selected rule before returning control to the actual game. */
            if (attempted.add(command)) {
                return command;
            }
        }
        return "quit";
    }

    /** Returns an ordered rule category, or excludes actions without a corresponding rule. */
    private int priority(Solitaire board, Move move) {
        /** Stock handling is last and uses engine legality rather than an approximate pass counter. */
        if (move.isTurn()) {
            return 4;
        }
        /** Quit and worry-back have no tactical rule in this deliberately simple baseline. */
        if (move.isQuit() || move.from().type() == Move.PileType.FOUNDATION) {
            return Integer.MAX_VALUE;
        }
        /** Only legal exposed tops appear among foundation candidates. */
        if (move.to().type() == Move.PileType.FOUNDATION) {
            return 0;
        }
        /** Deploy waste before rearranging fully visible tableau columns. */
        if (move.from().type() == Move.PileType.WASTE) {
            return 2;
        }

        /** Preserve whole-visible-run selection rather than introducing partial-run search. */
        int column = move.from().index();
        List<Card> visible = board.getVisibleTableau().get(column);
        Card head = visible.getFirst();
        if (move.card() == null || move.card().rank() != head.getRank()
                || move.card().suit() != head.getSuit()) {
            return Integer.MAX_VALUE;
        }
        /** A whole-run move with hidden cards beneath it physically exposes a new tableau card. */
        if (board.getTableauFaceDownCounts().get(column) > 0) {
            return 1;
        }
        /** Moving an exposed king to another empty column creates no additional space. */
        if (head.getRank() == Rank.KING) {
            return Integer.MAX_VALUE;
        }
        return 3;
    }

    /** Orders tableau sources T7 to T1, followed by waste and then non-card commands. */
    private int sourceOrder(Move move) {
        /** Non-card actions do not carry a source reference. */
        if (!move.isMove()) {
            return 8;
        }
        return move.from().type() == Move.PileType.TABLEAU ? 6 - move.from().index() : 7;
    }

    /** Preserves ascending foundation destinations and descending tableau destinations. */
    private int destinationOrder(Move move) {
        /** Non-card actions do not carry a destination reference. */
        if (!move.isMove()) {
            return 0;
        }
        return move.to().type() == Move.PileType.FOUNDATION ? move.to().index() : 6 - move.to().index();
    }
}

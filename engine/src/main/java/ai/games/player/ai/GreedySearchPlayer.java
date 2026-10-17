package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Move;
import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.LegalMovesHelper;
import ai.games.player.Player;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Deterministic one-ply greedy player that scores every legal immediate successor.
 *
 * <p>Only observed identities are available to simulations. Scores reward physical tableau
 * reveals, net empty columns, waste deployment, and foundation progress. Conservative
 * foundation safety affects priority rather than legality. Ties retain engine move order.
 * There is no deeper search, hidden-card sampling, or random escape.
 *
 * <p>Observed transitions and visited positions prevent cycles, including stock recycling.
 * Knowledge and remaining finite passes are part of the state key, so new observations do
 * not count as old positions. Create one player per game, or call {@link #resetGame()}.
 */
@Component
@Profile("ai-greedy")
public class GreedySearchPlayer extends AIPlayer implements Player {

    /** Observed positions in this player's current game, never shared with other players. */
    private final Set<Long> visitedStates = new HashSet<>();

    /** Actual observed successors, distinguished from predictions containing unknown cards. */
    private final Map<StateCommand, Long> transitions = new HashMap<>();

    /** Previous selected action awaiting its real observation, including stock turns. */
    private StateCommand pendingAction;

    /** Reward for exposing a physically face-down tableau card, even in thoughtful mode. */
    private static final int SCORE_REVEAL = 10;

    /** Reward per net new empty tableau column; relocating a king earns no such reward. */
    private static final int SCORE_EMPTY_COLUMN = 7;

    /** Reward for foundation progress with conservative support from the other suits. */
    private static final int SCORE_SAFE_FOUNDATION = 15;

    /** Smaller reward for legal foundation progress whose tableau usefulness is uncertain. */
    private static final int SCORE_RISKY_FOUNDATION = 3;

    /** Cost of withdrawing a foundation card, recoverable by immediate tableau benefits. */
    private static final int COST_WORRY_BACK = 5;

    /** Creates an independent player with no observed game history. */
    public GreedySearchPlayer() {
        /** Field initializers provide empty per-game history. */
    }

    /** Clears this instance's history when its owner starts a different game. */
    public void resetGame() {
        /** Discard both visited observations and the action awaiting observation. */
        visitedStates.clear();
        transitions.clear();
        pendingAction = null;
    }

    /**
     * Retains source compatibility for callers that reset before constructing a new player.
     *
     * @deprecated History is now instance-owned; construct a new player or call {@link #resetGame()}.
     */
    @Deprecated
    public static void resetForNewGame() {
        /** No shared game history remains to reset. */
    }

    /**
     * Selects the highest-scoring unvisited immediate successor, or quits when none remains.
     *
     * <p>Every selected action is recorded through the same path. Unknown predictions are
     * never stored as actual observations; the next call supplies the real successor.
     */
    @Override
    public String nextCommand(Solitaire solitaire, String recommendedMoves, String feedback) {
        /** Restrict all features and successor simulations to the player's knowledge. */
        Solitaire board = solitaire.planningCopy();
        long currentKey = board.getStateKey();
        /** Complete the previous transition from the board actually returned by the game. */
        if (pendingAction != null) {
            transitions.put(pendingAction, currentKey);
        }
        visitedStates.add(currentKey);

        /** Evaluate each legal action once, retaining the first candidate on equal scores. */
        Move best = null;
        int bestScore = Integer.MIN_VALUE;
        for (Move move : LegalMovesHelper.listMoves(board)) {
            /** Quit is a termination decision, not a scored successor. */
            if (move.isQuit()) {
                continue;
            }
            Solitaire successor = board.copy();
            apply(successor, move);
            Long observed = transitions.get(new StateCommand(currentKey, move.toCommandString()));
            long nextKey = observed != null ? observed : successor.getStateKey();
            /** Reject repeated observations, including exact reversals and exhausted stock cycles. */
            if (visitedStates.contains(nextKey)) {
                continue;
            }
            int score = score(board, successor, move);
            /** Strict comparison makes tie-breaking deterministic in engine generation order. */
            if (best == null || score > bestScore) {
                best = move;
                bestScore = score;
            }
        }

        /** Record turns and card moves identically; quitting leaves no pending action. */
        String command = best == null ? "quit" : best.toCommandString();
        pendingAction = best == null ? null : new StateCommand(currentKey, command);
        return command;
    }

    /** Applies one generated legal move to a planning copy without expanding its successors. */
    private void apply(Solitaire board, Move move) {
        /** Stock turns use the configured draw count and pass limit. */
        if (move.isTurn()) {
            board.turnStock();
        } else {
            /** Structured moves preserve explicit run heads and foundation withdrawals. */
            board.moveCard(move.from().toCode(), move.card() == null ? null : move.card().shortName(),
                    move.to().toCode());
        }
    }

    /** Scores immediate physical changes and action utility without inspecting hidden identities. */
    private int score(Solitaire before, Solitaire after, Move move) {
        /** Engine-maintained counts correctly distinguish whole-run reveals from top-card moves. */
        int reveals = hiddenCount(before) - hiddenCount(after);
        int score = SCORE_REVEAL * reveals
                + SCORE_EMPTY_COLUMN * (emptyColumns(after) - emptyColumns(before));
        /** Drawing is neutral: moving cards between stock and waste alone is not deployment. */
        if (move.isTurn()) {
            return score;
        }
        /** Foundation legality comes from the engine; safety only adjusts its local reward. */
        if (move.to().type() == Move.PileType.FOUNDATION) {
            List<Card> destination = after.getFoundation().get(move.to().index());
            Card card = destination.getLast();
            score += isConservativelySafe(card, before.getFoundation())
                    ? SCORE_SAFE_FOUNDATION : SCORE_RISKY_FOUNDATION;
        } else if (move.from().type() == Move.PileType.FOUNDATION) {
            /** Worry-back remains available, but must compete with cheaper immediate actions. */
            score -= COST_WORRY_BACK;
        } else if (move.from().type() == Move.PileType.WASTE) {
            /** Deploying waste cards changes access on later passes without assuming their identities. */
            score += 2;
        } else {
            /** Give ordinary tableau rearrangements a small deterministic preference over drawing. */
            score += 1;
        }
        return score;
    }

    /** Counts physically face-down cards independently of whether their identities are known. */
    private int hiddenCount(Solitaire board) {
        /** Sum all tableau columns; no hidden identity is consulted. */
        return board.getTableauFaceDownCounts().stream().mapToInt(Integer::intValue).sum();
    }

    /** Counts genuinely empty columns, excluding columns whose visible suffix is empty. */
    private int emptyColumns(Solitaire board) {
        /** Both visible and face-down counts must be zero for a usable empty column. */
        int empty = 0;
        for (int column = 0; column < 7; column++) {
            /** A column with hidden cards is not an empty king destination. */
            if (board.getTableauFaceUpCounts().get(column) == 0
                    && board.getTableauFaceDownCounts().get(column) == 0) {
                empty++;
            }
        }
        return empty;
    }

    /**
     * Gives priority to low cards and cards supported by progress in every other suit.
     *
     * <p>Opposite-color suits must reach rank minus one; the other same-color suit must
     * reach rank minus two. This conservative heuristic is not a proof of deal solvability.
     */
    private boolean isConservativelySafe(Card card, List<List<Card>> foundations) {
        /** Aces and twos cannot serve as essential supports for undeployed lower tableau runs. */
        int rank = card.getRank().getValue();
        if (rank <= 2) {
            return true;
        }
        /** Count supporting suits, treating an empty foundation as rank zero. */
        int supportingSuits = 0;
        for (List<Card> pile : foundations) {
            /** Only nonempty foundations identify a suit and its deployed rank. */
            if (!pile.isEmpty() && pile.getLast().getSuit() != card.getSuit()) {
                Card top = pile.getLast();
                int required = top.getSuit().isRed() == card.getSuit().isRed() ? rank - 2 : rank - 1;
                /** Each sufficiently advanced other suit contributes one support. */
                if (top.getRank().getValue() >= required) {
                    supportingSuits++;
                }
            }
        }
        return supportingSuits == 3;
    }

    /**
     * Identifies an action at an observed position for subsequent transition learning.
     *
     * @param stateKey knowledge-aware position before the selected command
     * @param command selected engine command, including stock turns
     */
    private record StateCommand(long stateKey, String command) { }
}

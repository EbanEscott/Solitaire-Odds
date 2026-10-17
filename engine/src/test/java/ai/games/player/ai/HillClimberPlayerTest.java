package ai.games.player.ai;

import ai.games.game.Card;
import ai.games.game.Deck;
import ai.games.game.Move;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.LegalMovesHelper;
import ai.games.unit.helpers.SolitaireFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Integration coverage for hill-climbing completion, legality, and history on a reproducible real deal. */
class HillClimberPlayerTest {
    /** Integration budget, not a claim that every deal can be resolved within this many commands. */
    private static final int MAX_TEST_STEPS = 1000;

    /** A known near-win must still complete through legal foundation progress. */
    @Test
    void hillClimberCompletesNearlyWonGame() {
        /** Exercise seeded local selection on a complete, validated fixture. */
        Solitaire board = SolitaireFactory.oneMoveFromWin();
        HillClimberPlayer player = new HillClimberPlayer(123);
        for (int step = 0; step < 10 && !won(board); step++) {
            String command = player.nextCommand(board, "", "");
            assertNotEquals("quit", command);
            apply(board, command);
        }
        assertTrue(won(board));
    }

    /** A fixed real deal must never repeat an action at an identical observed position. */
    @Test
    void decisionsAreLegalAndDoNotRepeatObservedActions() {
        /** Fix both deck order and tie-breaking seed so failures can be reproduced. */
        Solitaire board = seededBoard();
        HillClimberPlayer player = new HillClimberPlayer(42);
        Set<String> decisions = new HashSet<>();
        for (int step = 0; step < MAX_TEST_STEPS && !won(board); step++) {
            long observation = board.planningCopy().getStateKey();
            long trueState = board.getStateKey();
            String command = player.nextCommand(board, "", "");
            assertEquals(trueState, board.getStateKey(), "Search must not mutate the game");
            assertTrue(LegalMovesHelper.listLegalMoves(board).contains(command), command);
            /** Exhausted policy choices may terminate even when some repeating actions remain legal. */
            if (command.equals("quit")) {
                break;
            }
            assertTrue(decisions.add(observation + ":" + command), "Repeated observed action: " + command);
            apply(board, command);
        }
    }

    /** Builds an ordinary 52-card game with stable suit/rank ordering before seeded shuffling. */
    private Solitaire seededBoard() {
        /** Unknown placeholders are excluded from the physical deck. */
        List<Card> cards = new ArrayList<>();
        for (Suit suit : Suit.values()) {
            /** Skip the suit placeholder used only in planning observations. */
            if (suit == Suit.UNKNOWN) {
                continue;
            }
            /** Add each real rank exactly once for this suit. */
            for (Rank rank : Rank.values()) {
                /** Skip the rank placeholder used only in planning observations. */
                if (rank != Rank.UNKNOWN) {
                    cards.add(new Card(rank, suit));
                }
            }
        }
        Collections.shuffle(cards, new Random(42));
        return new Solitaire(new Deck(cards));
    }

    /** Applies a selected command and requires that the real engine accepts it. */
    private void apply(Solitaire board, String command) {
        /** Parse the same structured formats used by production move generation. */
        Move move = Move.parse(command);
        if (move.isTurn()) {
            assertTrue(board.turnStock());
        } else {
            /** Card moves can omit an explicit card when the source has only one playable top. */
            assertTrue(board.moveCard(move.from().toCode(),
                    move.card() == null ? null : move.card().shortName(), move.to().toCode()));
        }
    }

    /** Requires all 52 cards to reach foundations for completion. */
    private boolean won(Solitaire board) {
        /** Foundation sizes are observable and do not depend on hidden identities. */
        return board.getFoundation().stream().mapToInt(List::size).sum() == 52;
    }
}

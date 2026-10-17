package ai.games.unit.player;

import ai.games.game.Card;
import ai.games.game.GameRules;
import ai.games.game.Rank;
import ai.games.game.Solitaire;
import ai.games.game.Suit;
import ai.games.player.ai.RuleBasedHeuristicsPlayer;
import ai.games.unit.helpers.SolitaireBuilder;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Regressions for ordered rules, stock-cycle termination, and independent game histories. */
class RuleBasedRulesTest {
    /** Revealing a hidden card outranks the first otherwise eligible consolidation. */
    @Test
    void prefersRevealsBeforeConsolidation() {
        /** T7 is scanned first, but moving T1 is the only move that exposes a hidden card. */
        Solitaire board = SolitaireBuilder.newGame().tableau("T7", "5♣")
                .tableau("T6", "6♥").tableau("T1", 1, "9♦", "7♣")
                .tableau("T2", "8♥").build();
        RuleBasedHeuristicsPlayer player = openStock(board);
        assertEquals("move T1 7♣ T2", player.nextCommand(board, "", ""));
    }

    /** Stock deployments must not be starved by rearranging fully visible tableau piles. */
    @Test
    void deploysWasteBeforeConsolidation() {
        /** The stable fixture stock draws J-spades onto a visible red queen. */
        Solitaire board = SolitaireBuilder.newGame().tableau("T7", "7♣")
                .tableau("T6", "8♥").tableau("T3", "Q♥").build();
        RuleBasedHeuristicsPlayer player = openStock(board);
        assertEquals("J♠", board.getTalon().getLast().shortName());
        assertEquals("move W T3", player.nextCommand(board, "", ""));
    }

    /** Repeated positions exhaust their stock action without a hard-coded draw limit. */
    @Test
    void terminatesStockCyclesAcrossConfiguredRules() {
        /** Exercise both draw sizes, visibility policies, and finite/unlimited allowances. */
        for (int draw : List.of(1, 3)) {
            /** Observation policy must not change legal cycle handling. */
            for (GameRules.StockVisibility visibility : GameRules.StockVisibility.values()) {
                /** Include one pass, three passes, and unlimited recycling. */
                for (Integer passes : java.util.Arrays.asList(1, 3, null)) {
                    /** No stock card fits the tableau or foundations in this complete-deck fixture. */
                    Solitaire board = blockedStock(new GameRules(draw, visibility, passes));
                    RuleBasedHeuristicsPlayer player = new RuleBasedHeuristicsPlayer();
                    int turns = 0;
                    /** Bound the test so a broken termination rule fails promptly. */
                    while (turns < 30) {
                        String command = player.nextCommand(board, "", "");
                        /** Quit is expected either at the configured limit or on a repeated observation. */
                        if (command.equals("quit")) {
                            break;
                        }
                        assertEquals("turn", command);
                        assertTrue(board.turnStock());
                        turns++;
                    }
                    assertTrue(turns >= 3 / draw && turns < 30, "Must draw the first pass and terminate");
                    /** Unlimited-pass termination must come from history rather than engine illegality. */
                    if (passes == null) {
                        assertTrue(board.canTurnStock());
                    }
                }
            }
        }
    }

    /** Another player and an explicitly reset player retain their own opening observation. */
    @Test
    void resetsHistoryForAnotherGame() {
        /** Reuse an unchanged observation to exhaust its only applicable action. */
        Solitaire board = blockedStock(GameRules.DEFAULT);
        RuleBasedHeuristicsPlayer player = new RuleBasedHeuristicsPlayer();
        assertEquals("turn", player.nextCommand(board, "", ""));
        assertEquals("quit", player.nextCommand(board, "", ""));
        assertEquals("turn", new RuleBasedHeuristicsPlayer().nextCommand(board, "", ""));
        player.resetGame();
        assertEquals("turn", player.nextCommand(board, "", ""));
    }

    /** Executes the preserved opening draw before checking subsequent tactical rules. */
    private RuleBasedHeuristicsPlayer openStock(Solitaire board) {
        /** Apply the returned action so history always corresponds to an actual observation. */
        RuleBasedHeuristicsPlayer player = new RuleBasedHeuristicsPlayer();
        assertEquals("turn", player.nextCommand(board, "", ""));
        assertTrue(board.turnStock());
        return player;
    }

    /** Builds a complete deck with three unusable stock cards and an immovable tableau top. */
    private Solitaire blockedStock(GameRules rules) {
        /** Reserve three stock identities and bury all remaining cards below the ten. */
        List<String> buried = new ArrayList<>();
        Set<String> reserved = Set.of("5♣", "6♣", "7♣", "10♣");
        for (Suit suit : Suit.values()) {
            /** Unknown placeholders are not physical cards. */
            if (suit == Suit.UNKNOWN) {
                continue;
            }
            /** Retain each physical rank exactly once. */
            for (Rank rank : Rank.values()) {
                /** Skip the nonphysical rank placeholder. */
                if (rank != Rank.UNKNOWN) {
                    String card = new Card(rank, suit).shortName();
                    /** Reserved cards occupy stock or the tableau top instead. */
                    if (!reserved.contains(card)) {
                        buried.add(card);
                    }
                }
            }
        }
        buried.add("10♣");
        return SolitaireBuilder.newGame().tableau("T1", 1, buried.toArray(String[]::new))
                .stock("5♣", "6♣", "7♣").build(rules);
    }
}

package ai.games.unit.helpers;

import ai.games.game.Solitaire;

/**
 * Reusable scenario factory for tests.
 *
 * <p><strong>Why this exists</strong>
 * <ul>
 *   <li><strong>Consistency:</strong> many tests want the same handful of canonical board positions.
 *       Centralising them prevents drift and makes failures easier to reason about.</li>
 *   <li><strong>Readability:</strong> scenario names communicate intent better than repeating long
 *       builder calls throughout the suite.</li>
 *   <li><strong>Strictness:</strong> scenarios are built via {@link SolitaireBuilder}, so they are
 *       always complete (52 cards) and validated.</li>
 * </ul>
 *
 * <p><strong>Design note</strong>
 * <p>This factory intentionally contains only "common" scenarios. Tests that require unusual or
 * highly specific layouts should use {@link SolitaireBuilder} directly.
 */
public final class SolitaireFactory {

    private SolitaireFactory() {
    }

    /**
     * A strict but minimal state: all 52 cards are in the stockpile.
     *
     * <p><strong>Why:</strong> useful as a baseline for tests that only care about stock/talon
     * transitions (e.g., {@link ai.games.game.Solitaire#turnThree()}) without involving tableau.
     */
    public static Solitaire stockOnly() {
        return SolitaireBuilder.newGame().build();
    }

    /**
     * Endgame state that is exactly one legal move away from winning.
     *
     * <p>Setup:
     * <ul>
     *   <li>F1 = A♣..K♣</li>
     *   <li>F2 = A♦..K♦</li>
     *   <li>F3 = A♠..K♠</li>
     *   <li>F4 = A♥..Q♥</li>
     *   <li>T1 = K♥ (face-up)</li>
     *   <li>Stock/talon empty</li>
     * </ul>
     *
     * <p><strong>Why:</strong> ideal for deterministic "win detection" and AI regression tests.
     */
    public static Solitaire oneMoveFromWin() {
        return SolitaireBuilder
                .newGame()
                .tableau("T1", "K♥")
                .foundation("F1",
                        "A♣", "2♣", "3♣", "4♣", "5♣", "6♣", "7♣", "8♣", "9♣", "10♣", "J♣", "Q♣", "K♣")
                .foundation("F2",
                        "A♦", "2♦", "3♦", "4♦", "5♦", "6♦", "7♦", "8♦", "9♦", "10♦", "J♦", "Q♦", "K♦")
                .foundation("F3",
                        "A♠", "2♠", "3♠", "4♠", "5♠", "6♠", "7♠", "8♠", "9♠", "10♠", "J♠", "Q♠", "K♠")
                .foundation("F4",
                        "A♥", "2♥", "3♥", "4♥", "5♥", "6♥", "7♥", "8♥", "9♥", "10♥", "J♥", "Q♥")
                .stock()
                .waste()
                .build();
    }

    /**
     * Common tableau-flip scenario.
     *
     * <p>Setup:
     * <ul>
     *   <li>T1 = K♦ (face-down), 5♠ (face-up)</li>
     *   <li>T2 = 6♥ (face-up)</li>
     *   <li>T3..T7 empty</li>
     * </ul>
     *
     * <p><strong>Why:</strong> used to test that moving the last face-up card flips the next
     * face-down card as per Klondike rules.
     */
    public static Solitaire flipAfterMovingLastFaceUpCard() {
        return SolitaireBuilder
                .newGame()
                .tableau("T1", 1, "K♦", "5♠")
                .tableau("T2", "6♥")
                .stock()
                .waste()
                .build();
    }
}

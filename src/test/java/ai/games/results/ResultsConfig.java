package ai.games.results;

/**
 * Shared knobs for AI result sweeps.
 *
 * Purpose:
 * - Keep sweep sizing and safety limits consistent across all AI-vs-AI or AI-vs-baseline runs.
 * - These values directly affect statistical confidence and runtime.
 *
 * How to think about changes:
 * - More games => narrower confidence interval (less noise) but longer sweeps.
 *   The standard error shrinks ~ 1/sqrt(n). So:
 *     * 4× games ≈ 2× tighter win-rate precision
 *     * 9× games ≈ 3× tighter precision
 * - Move cap protects against stalled/looping games and runaway runtimes.
 */
public final class ResultsConfig {
    private ResultsConfig() {}

    /**
     * Number of games to run in a single sweep.
     *
     * What changing this does:
     * - Increasing GAMES reduces randomness in the measured win rate.
     * - Decreasing GAMES speeds up sweeps but makes results noisier.
     *
     * Rough 95% confidence half-widths (worst case p≈0.5):
     * - 10,000 games     -> ±1.0%
     * - 40,000 games     -> ±0.5%
     * - 250,000 games    -> ±0.2%
     * - 1,000,000 games  -> ±0.1%
     *
     * Use cases:
     * - Early tuning / broad comparisons: 25k–100k is usually enough.
     * - Final benchmarks / tiny improvements (<0.2%): 250k–1M.
     */
    public static final int GAMES = 10000;

    /**
     * How often to log progress during multi-game sweeps.
     *
     * Example: 50 means log at games 1, 50, 100, …, N.
     */
    public static final int PROGRESS_LOG_INTERVAL = 10;

    /**
     * Safety cap on the number of moves allowed per game.
     *
     * What changing this does:
     * - Increasing MAX_MOVES_PER_GAME allows very long games to finish,
     *   which may be necessary for some rule sets or weak AIs.
     * - Decreasing it makes sweeps faster and prevents "death-spiral" games,
     *   but may bias results if legitimate games often exceed the cap.
     *
     * Typical move counts (update these if your rules/AI change):
     * - Healthy / non-stalled games: often ~150–400 moves.
     * - Long but legitimate games: can run ~600–900 moves.
     * - >1000 moves is usually a loop/stall symptom.
     *
     * Guidance:
     * - Set this comfortably above the 99th percentile of move counts
     *   for stable play (log/measure once, then tune).
     * - If you see frequent caps being hit, your results are suspect:
     *   raise the cap or fix the AI/game rules that cause loops.
     */
    public static final int MAX_MOVES_PER_GAME = 1000;
}

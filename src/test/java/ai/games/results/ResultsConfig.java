package ai.games.results;

/**
 * Shared knobs for AI result sweeps. Adjust here to change runs consistently across tests.
 */
public final class ResultsConfig {
    private ResultsConfig() {}

    // Default number of games per sweep.
    public static final int GAMES = 1000000;

    // Safety cap on moves per game.
    public static final int MAX_MOVES_PER_GAME = 1000;
}

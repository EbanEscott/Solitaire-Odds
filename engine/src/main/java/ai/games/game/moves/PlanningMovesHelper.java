package ai.games.game.moves;

/**
 * Legal moves for a planning board use the same rules as a real board.
 * <p>
 * Players resolve revealed UNKNOWN slots through
 * {@link ai.games.game.Solitaire#assumeCard(ai.games.game.Card, ai.games.game.Card)}
 * before expanding them.
 * Listing moves never commits guesses or treats UNKNOWN cards as wildcards.
 */
public class PlanningMovesHelper extends GameMovesHelper {
}

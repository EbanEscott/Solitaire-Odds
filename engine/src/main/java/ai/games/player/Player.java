package ai.games.player;

import ai.games.game.Solitaire;

/**
 * Represents a player capable of providing the next command for the game loop.
 */
public interface Player {
    
    /**
     * Provide the next command for the game loop (e.g., "turn", "move T1 F1", "quit").
     *
     * @param solitaire current game state; useful for AI decisions.
     * @param moves     recommended legal moves for this turn, rendered as a human-readable
     *                  string (e.g., a bullet list).
     * @param feedback  guidance and/or error feedback for this turn.
     * @return raw command string, or null to signal the game should exit.
     */
    String nextCommand(Solitaire solitaire, String moves, String feedback);
}

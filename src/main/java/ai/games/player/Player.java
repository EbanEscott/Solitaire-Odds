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
     * @return raw command string, or null to signal the game should exit.
     */
    String nextCommand(Solitaire solitaire);
}

package ai.games.player;

import ai.games.game.Solitaire;
import ai.games.game.moves.GameMovesHelper;
import ai.games.game.moves.PlanningMovesHelper;
import java.util.Collections;
import java.util.List;

/**
 * Facade for computing legal moves in Solitaire.
 * <p>
 * Dispatches to the appropriate implementation based on game mode:
 * <ul>
 *   <li><b>GAME mode:</b> Uses {@link GameMovesHelper} (known cards)
 *   <li><b>PLAN mode:</b> Uses {@link PlanningMovesHelper} (UNKNOWN cards with plausibility)
 * </ul>
 * <p>
 * This class maintains the public API for backward compatibility while delegating
 * the actual move generation to mode-specific implementations.
 */
public final class LegalMovesHelper {
    private LegalMovesHelper() {
    }

    /**
     * Return all currently legal commands (excluding "quit").
     */
    public static List<String> listLegalMoves(Solitaire solitaire) {
        if (solitaire == null) {
            return Collections.emptyList();
        }
        
        // Dispatch to appropriate implementation based on mode
        if (solitaire.isInPlanMode()) {
            return new PlanningMovesHelper().listLegalMoves(solitaire);
        } else {
            return new GameMovesHelper().listLegalMoves(solitaire);
        }
    }
}

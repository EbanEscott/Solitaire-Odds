package ai.games.player;

import ai.games.game.Solitaire;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Placeholder AI player; will generate moves in future iterations.
 */
@Component
@Profile("ai")
public class AIPlayer implements Player {
    @Override
    public String nextCommand(Solitaire solitaire) {
        // TODO: implement AI move generation.
        return "quit";
    }
}

package ai.games.player.ai.alpha;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * AlphaSolitaire-backed AI player that delegates move selection to the
 * Python policy–value network via the HTTP service.
 *
 * Requires the Python service to be running locally, for example:
 *
 *   cd /Users/ebo/Code/solitaire
 *   source .venv/bin/activate
 *   python3 service.py --checkpoint checkpoints/policy_value_latest.pt --host 127.0.0.1 --port 8000
 */
@Component
@Profile("ai-alpha-solitaire")
public class AlphaSolitairePlayer extends AIPlayer implements Player {

    private static final Logger log = LoggerFactory.getLogger(AlphaSolitairePlayer.class);

    private final AlphaSolitaireClient client;

    public AlphaSolitairePlayer(AlphaSolitaireClient client) {
        this.client = client;
    }

    @Override
    public String nextCommand(Solitaire solitaire, String moves, String feedback) {
        if (log.isDebugEnabled()) {
            log.debug("AlphaSolitairePlayer preparing request for current board state");
        }
        AlphaSolitaireRequest request = AlphaSolitaireRequest.fromSolitaire(solitaire);
        AlphaSolitaireResponse response = client.evaluate(request);

        if (response == null || response.getChosenCommand() == null || response.getChosenCommand().isBlank()) {
            log.warn("AlphaSolitaire service returned no command; defaulting to \"quit\".");
            return "quit";
        }

        String chosen = response.getChosenCommand().trim();
        if (log.isDebugEnabled()) {
            log.debug("AlphaSolitaire chosen command: {} (winProbability={})",
                    chosen, response.getWinProbability());
        }
        return chosen;
    }
}

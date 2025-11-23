package ai.games.player.ai;

import ai.games.game.Solitaire;
import ai.games.player.AIPlayer;
import ai.games.player.Player;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Experimental rule-based player with room for richer heuristics.
 *
 * Design goals:
 * - Start from the same basic contract as {@link SimpleRuleBasedHeuristicsPlayer}:
 *   a deterministic, rules-first player with no heavy search.
 * - Provide a dedicated class where we can iteratively add:
 *   - improved “safe to foundation” heuristics,
 *   - better handling of long stock cycles,
 *   - more nuanced tableau reshuffling rules.
 *
 * For now this implementation simply delegates to {@link SimpleRuleBasedHeuristicsPlayer}
 * to ensure behaviour is correct and testable. Future iterations can gradually fork the
 * decision logic here while keeping the “simple” version stable as a baseline.
 *
 * Profile: {@code ai-rule-complex}
 */
@Component
@Profile("ai-rule-complex")
public class ComplexRuleBasedHeuristicsPlayer extends AIPlayer implements Player {

    private final SimpleRuleBasedHeuristicsPlayer delegate = new SimpleRuleBasedHeuristicsPlayer();

    @Override
    public String nextCommand(Solitaire solitaire, String feedback) {
        // TODO(evolve): replace this delegation with enhanced heuristics while
        // keeping behaviour backward-compatible with the simple player where sensible.
        return delegate.nextCommand(solitaire, feedback);
    }
}


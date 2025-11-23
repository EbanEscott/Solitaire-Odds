package ai.games.player.ai;

import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * Thin wrapper to preserve the existing rule-based behaviour under a clearer name.
 *
 * This class simply extends the original {@link RuleBasedHeuristicsPlayer} implementation
 * so that we can:
 * - keep a stable, “simple” rule-based baseline for comparison, and
 * - introduce {@code ComplexRuleBasedHeuristicsPlayer} as a separate, experimental variant.
 *
 * Profile: {@code ai-rule}
 */
@Component
@Profile("ai-rule")
public class SimpleRuleBasedHeuristicsPlayer extends RuleBasedHeuristicsPlayer {
}


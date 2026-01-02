package ai.games.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Spring Boot configuration properties for guidance mode.
 * 
 * When enabled, guidance mode provides recommended moves, feedback on illegal moves,
 * and detection of unproductive patterns (ping-pong, repeated stock passes).
 * Useful for human players and LLM-based players; can be disabled for pure AI experiments.
 * 
 * Usage:
 * {@code ./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human -Dguidance.mode=true}
 * 
 * @since 1.0
 */
@Component
@ConfigurationProperties(prefix = "guidance")
public class GuidanceModeProperties {
  private boolean mode = true;

  /**
   * Returns whether guidance mode is enabled.
   * @return true if guidance mode is active, false otherwise
   */
  public boolean isMode() {
    return mode;
  }

  /**
   * Sets guidance mode enabled/disabled.
   * @param mode true to enable guidance mode, false to disable
   */
  public void setMode(boolean mode) {
    this.mode = mode;
  }
}

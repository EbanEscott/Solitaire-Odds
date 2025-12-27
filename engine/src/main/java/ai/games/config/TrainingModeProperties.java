package ai.games.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * Spring Boot configuration properties for training mode.
 * 
 * When enabled, training mode allows players (human and AI) to undo their moves,
 * exploring alternative game paths without penalty.
 * 
 * Usage:
 * {@code ./gradlew bootRun --console=plain -Dspring.profiles.active=ai-human -Dtraining.mode=true}
 * 
 * @since 1.0
 */
@Component
@ConfigurationProperties(prefix = "training")
public class TrainingModeProperties {
  private boolean mode = false;

  /**
   * Returns whether training mode is enabled.
   * @return true if training mode is active, false otherwise
   */
  public boolean isMode() {
    return mode;
  }

  /**
   * Sets training mode enabled/disabled.
   * @param mode true to enable training mode, false to disable
   */
  public void setMode(boolean mode) {
    this.mode = mode;
  }
}

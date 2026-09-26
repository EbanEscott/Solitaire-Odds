package ai.games.player;

import java.util.Map;

/** Supplies game-scoped metadata for reproducible experiment logs. */
public interface ExperimentMetadataProvider {

    /**
     * Returns metadata that is stable for this game once the first command has been selected.
     *
     * @return JSON-serializable experiment metadata
     */
    Map<String, Object> getExperimentMetadata();
}

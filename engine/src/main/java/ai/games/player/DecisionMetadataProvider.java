package ai.games.player;

import java.util.Map;

/** Supplies metadata for the decision most recently returned by a player. */
public interface DecisionMetadataProvider {

    /**
     * Returns JSON-serializable details for the command selected on the current turn.
     *
     * @return decision metadata, or an empty map when none is available
     */
    Map<String, Object> getLastDecisionMetadata();
}

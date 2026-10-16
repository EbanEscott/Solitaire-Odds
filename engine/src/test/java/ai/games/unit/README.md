# Unit Tests

Core unit tests for game logic and helper utilities.

## Structure

- **`game/`** — Game rules and state management (legal/illegal moves, tableau move ordering and destination refresh, structured/text move compatibility, boundaries, training mode)
- **`player/`** — Search-node pruning, move parsing, human console compatibility, and Monte Carlo rollout selection, copy counts, cycle retries, and parsing avoidance
- **`helpers/`** — Test support utilities (SolitaireTestHelper, TestGameStateBuilder, FoundationCountHelper)

## Run Unit Tests

```bash
# All unit tests
./gradlew test --tests "ai.games.unit.**"

# By category
./gradlew test --tests "ai.games.unit.game.**"
./gradlew test --tests "ai.games.unit.helpers.**"
```

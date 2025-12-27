# Unit Tests

Core unit tests for game logic and helper utilities.

## Structure

- **`game/`** — Game rules and state management (legal/illegal moves, boundaries, training mode)
- **`helpers/`** — Test support utilities (SolitaireTestHelper, TestGameStateBuilder, FoundationCountHelper)

## Run Unit Tests

```bash
# All unit tests
./gradlew test --tests "ai.games.unit.**"

# By category
./gradlew test --tests "ai.games.unit.game.**"
./gradlew test --tests "ai.games.unit.helpers.**"
```

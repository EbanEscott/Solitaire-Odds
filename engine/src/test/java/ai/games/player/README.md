# Player Tests

Tests for AI player implementations and their strategies.

## Structure

- **`ai/`** — AI player functional tests
  - Tests validate each player implementation behaves correctly
  - Verify move selection strategies work as intended
  - Check integration with game engine

See `ai/README.md` for details on individual AI players.

## Run Tests

```bash
./gradlew test --tests "ai.games.player.ai.**"
```

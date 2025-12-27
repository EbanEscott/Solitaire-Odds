# Game Rule Tests

Unit tests verifying core Solitaire game logic: legal moves, illegal move rejection, boundary conditions, and undo functionality.

## Tests

| Test | Purpose |
|------|---------|
| `LegalMovesTest` | Comprehensive validation that all legal moves are correctly identified by move logic |
| `IllegalMovesTest` | Ensures the game rejects moves that violate Solitaire rules |
| `BoundaryTest` | Edge cases: empty piles, full foundations, special card positions |
| `TrainingModeTest` | Validates undo mechanism: records moves, replays game minus last move, restores deck order deterministically |

## Run Tests

```bash
./gradlew test --tests "ai.games.unit.game.**"
./gradlew test --tests "ai.games.unit.game.LegalMovesTest"
./gradlew test --tests "ai.games.unit.game.IllegalMovesTest"
./gradlew test --tests "ai.games.unit.game.BoundaryTest"
./gradlew test --tests "ai.games.unit.game.TrainingModeTest"
```

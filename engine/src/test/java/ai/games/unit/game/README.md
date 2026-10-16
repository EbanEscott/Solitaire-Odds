# Game Rule Tests

Unit tests verifying core Solitaire game logic: legal moves, illegal move rejection, boundary conditions, PLAN mode move generation with masked cards, and undo functionality.

## Tests

| Test | Purpose |
|------|---------|
| `LegalMovesTest` | Comprehensive validation that all legal moves are correctly identified by move logic |
| `IllegalMovesTest` | Ensures the game rejects moves that violate Solitaire rules |
| `BoundaryTest` | Edge cases: empty piles, full foundations, special card positions |
| `PlanningMovesTest` | PLAN mode move generation with UNKNOWN cards: guess creation, plausibility checking, game state copying, and guess consistency validation |
| `SolitaireCopyTest` | Board preservation, independent moves across simulation copies, and PLAN masking and guess copying |
| `TrainingModeTest` | Validates direct undo and rollback: card stacks, flips, stock recycling, knowledge, cached keys, independent copies, and training commands |

## Run Tests

```bash
./gradlew test --tests "ai.games.unit.game.**"
./gradlew test --tests "ai.games.unit.game.LegalMovesTest"
./gradlew test --tests "ai.games.unit.game.IllegalMovesTest"
./gradlew test --tests "ai.games.unit.game.BoundaryTest"
./gradlew test --tests "ai.games.unit.game.PlanningMovesTest"
./gradlew test --tests "ai.games.unit.game.TrainingModeTest"
```

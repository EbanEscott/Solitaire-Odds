# Test Helpers

Utility classes supporting unit tests for game logic and AI players.

## Helpers

| Class | Purpose |
|-------|---------|
| `SolitaireTestHelper` | Reflection-based utilities for seeding Solitaire with deterministic test states (piles, foundations, etc.) |
| `FoundationCountHelper` | Calculates total foundation cards for state evaluation |
| `TestGameStateBuilder` | Builds reusable test game states (e.g., nearly-won layouts) for AI player testing |

These are shared test support classes used by multiple test suites across game logic and AI player tests.

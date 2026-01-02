# Move Generation Helpers

This package contains **reusable move generation logic** that is independent of specific AI player implementations. Rather than embedding move calculation inside each player, we've centralized it here so all players can share the same game logic.

## Architecture: Shared Logic, Not Player-Specific Code

The key design principle is **separation of concerns**:
- **Player implementations** (`ai/games/player/ai/*.java`) focus on decision-making: "Given these moves, which should I choose?"
- **Move helpers** (`ai/games/player/moves/*.java`) handle mechanics: "What moves are legal given this game state?"

This approach has several benefits:

1. **Code Reuse**: Every player (Greedy, MCTS, LLM, etc.) uses the same move generation logic, avoiding duplication
2. **Maintainability**: Bug fixes and improvements to move logic benefit all players automatically
3. **Correctness**: Game rules are implemented once and tested thoroughly, not reimplemented per-player
4. **Flexibility**: Players can focus on strategy without worrying about legal move computation

## How It Works

### Game Mode Dispatch

The move generation system supports two modes:

**GAME Mode** (`GameMovesHelper`):
- Used during actual gameplay
- Only considers visible cards
- Direct, straightforward move generation

**PLAN Mode** (`PlanningMovesHelper`):
- Used for AI lookahead and search
- Masks face-down cards with `UNKNOWN` placeholders
- Maintains `UnknownCardGuess` map for plausible moves
- Prevents information leaks during tree search

### Entry Point: LegalMovesHelper

```java
List<String> moves = LegalMovesHelper.listLegalMoves(solitaire);
```

This dispatcher automatically routes to the appropriate helper based on `solitaire.getMode()`:
- GAME mode → GameMovesHelper
- PLAN mode → PlanningMovesHelper

Both return the same move string format, so players don't need to care which implementation was used.

## PLAN Mode: Masked Information for AI Search

### The Challenge

During AI lookahead (tree search), the engine creates copies of the game state and explores branches. If face-down cards were treated as completely hidden, no moves to them would be legal. But if face-down cards showed their true values during lookahead, the AI would gain information it shouldn't have.

### The Solution: UNKNOWN Placeholders

In PLAN mode, face-down cards are replaced with `Card(Rank.UNKNOWN, Suit.UNKNOWN)`. This allows:

1. **Plausible moves**: The AI can reason about moves to face-down cards
2. **Guess-based logic**: Each UNKNOWN card gets an `UnknownCardGuess` that constrains which real cards it might be
3. **Information integrity**: The AI sees the same masked information during lookahead as a human player would

### Example: Red 4 to Unknown Black Card

When a player moves Red 4 to a face-down (UNKNOWN) card:
- The card must be rank 5 (one higher)
- The card must be black (opposite color)
- Both Black 5s are plausible: 5♣ and 5♠

So `PlanningMovesHelper` creates an `UnknownCardGuess` mapping the UNKNOWN card instance to `[5♣, 5♠]`.

### Guess Consistency: validateGuesses()

After generating moves, `validateGuesses()` ensures the guess map stays consistent:

1. **Against unknownCardTracker**: Guess possibilities must be in the list of unrevealed cards
2. **No conflicts**: No real card can appear in multiple guesses
3. **Removal**: Guesses with no remaining possibilities are discarded

This is called at the end of every `listLegalMoves()` call, so every move list returned is based on a validated, consistent guess state.

## Key Classes

### GameMovesHelper
- Generates moves for actual gameplay
- Only works with visible/revealed cards
- No UNKNOWN handling needed

### PlanningMovesHelper
- Generates moves during AI search
- Creates and maintains `UnknownCardGuess` objects
- Handles moves to/from UNKNOWN cards
- Calls `validateGuesses()` before returning moves

### LegalMovesHelper (Dispatcher)
- Single entry point for both modes
- Routes to appropriate implementation
- Returns consistent move string format

## Testing

Move generation is thoroughly tested in:
- `ai.games.unit.game.LegalMovesTest` - Valid moves in standard positions
- `ai.games.unit.game.IllegalMovesTest` - Rejection of invalid moves
- `ai.games.unit.game.BoundaryTest` - Edge cases and game rules
- `ai.games.unit.game.PlanningMovesTest` - PLAN mode with UNKNOWN cards and guesses

## Real-World Design Note

In a real Solitaire game, each player/AI would independently calculate legal moves and manage their own guess assumptions. However, by centralising this logic here, we avoid reimplementing the same game rules for every player. If an LLM player, search player, and neural network player all use the same `LegalMovesHelper`, they all follow the same rules and can be reliably compared.

This is a pragmatic trade-off: slightly more abstraction in the codebase, but much cleaner player implementations and guaranteed rule consistency.

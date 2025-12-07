# Agents

This repo has three main pieces:

- `README.md` — top-level story, latest test results table, and an overview of all players.
- `engine/` — Java/Spring Boot Solitaire engine, CLI, and search/LLM players.
- `neural-network/` — Python modeling stack for AlphaSolitaire (policy–value net, training, and service).

## General guidance

When working in this repo:

- Prefer concise, direct explanations in docs (like this file and the READMEs).
- Keep code and tests in their respective modules (`engine`, `neural-network`) rather than at the root.
- Avoid adding new heavy dependencies unless they are clearly justified for the experiment.
- When you change behaviour in `engine` or `neural-network`, update the relevant module README and, if it affects player performance, the root `README.md` test results table.

## Testing and results

- Engine-side game logic and AI players are verified via Gradle tests under `engine/src/test/java`.
- The canonical win-rate table for players lives in the root `README.md`. If you regenerate results, update that table and note when the tests were last run.

### Testing and README refresh workflow

When asked to refresh the Solitaire player results end-to-end:

1. **Run the tests** found in the `engine/src/test/java/ai/games/results` package.
2. **Gather the test results** from the logging info for each test suite.
3. **Format or update the table** in the root `README.md` using the columns:
   - **Player**, **AI**, **Games Played**, **Games Won**, **Win %**, **Avg Time/Game**, **Total Time**, **Avg Moves**, **Best Win Streak**, **Notes**.
4. **Paste or adjust the table** in `README.md` and insert the date/time the tests were run (the “last test run” line).
5. **Update the LLM Notes** so that each LLM row follows the pattern:
   - “`<Provider> <model> via Ollama; see [code](...) and [model](...)`” or  
   - “`<Provider> <model> via API; see [code](...)`”
6. **Keep code links accurate** by using module-aware paths, for example:
   - `engine/src/main/java/ai/games/player/ai/GreedySearchPlayer.java`

An example table layout (values are illustrative only):

| Player                 | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |
|------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|
| Rule-based Heuristics | Search | 500          | 60        | 12%   | 0.3s          | 150s       | 95        | 3               | Deterministic rule-based baseline; see [code](engine/src/main/java/ai/games/player/ai/RuleBasedHeuristicsPlayer.java). |
| Greedy Search         | Search | 500          | 90        | 18%   | 0.5s          | 250s       | 105       | 4               | Greedy one-step lookahead using heuristic scoring; see [code](engine/src/main/java/ai/games/player/ai/GreedySearchPlayer.java). |
| OpenAI                | LLM    | 500          | TBD       | TBD   | TBD           | TBD        | TBD       | TBD             | OpenAI gpt-5-mini via API; see [code](engine/src/main/java/ai/games/player/ai/OpenAIPlayer.java). |
| Alibaba               | LLM    | 500          | TBD       | TBD   | TBD           | TBD        | TBD       | TBD             | Alibaba qwen3-coder:30b via Ollama; see [code](engine/src/main/java/ai/games/player/ai/OllamaPlayer.java) and [model](https://ollama.com/library/qwen3-coder). |

For module-specific instructions, see:

- `engine/AGENTS.md` — how to run result sweeps and refresh the win-rate table.
- `neural-network/AGENTS.md` — AlphaSolitaire modeling plan and Python guidelines.

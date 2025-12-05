# Agents

## Testing and README refresh workflow

When asked complete these tasks:

1. **Run the Tests** found in the `cards/src/test/java/ai/games/results` package.
2. **Gather the Test Results** from the logging info for each test suite.
3. **Format a Table** using the results and the example table below, including the **AI** column to indicate whether the method is an LLM or a search-based algorithm, and a **Notes** column for clickable links or extra context.
4. **Paste the Table** into the `cards/README.md` and insert the date/time the tests were run.

| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |
|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|
| Rule-based Heuristics        | Search | 500          | 60        | 12%   | 0.3s          | 150s       | 95        | 3               | Deterministic rule-based baseline; see [code](src/main/java/ai/games/player/ai/RuleBasedHeuristicsPlayer.java). |
| Greedy Search                | Search | 500          | 90        | 18%   | 0.5s          | 250s       | 105       | 4               | Greedy one-step lookahead using heuristic scoring; see [code](src/main/java/ai/games/player/ai/GreedySearchPlayer.java). |
| OpenAI                       | LLM    | 500          | TBD       | TBD   | TBD           | TBD        | TBD       | TBD             | Open-weight gpt-oss:120b via Ollama; see [code](src/main/java/ai/games/player/ai/OllamaPlayer.java) and [model](https://ollama.com/library/gpt-oss). |
| Meta                         | LLM    | 500          | TBD       | TBD   | TBD           | TBD        | TBD       | TBD             | Meta llama4:scout via Ollama; see [code](src/main/java/ai/games/player/ai/OllamaPlayer.java) and [model](https://ollama.com/library/llama4). |

* **Player** Name of the decision or optimisation method or LLM-backed player being tested.
* **AI** Whether the method is an `LLM` (e.g., Ollama) or a search-based algorithm (e.g., A*, beam search, greedy).
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Games Won** Count of games successfully completed.
* **Win %** Percentage of games successfully completed (foundations fully built).
* **Avg Time/Game** Mean time taken to finish or fail a game.
* **Total Time** Sum of all time spent playing the batch of games.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Best Win Streak** Longest run of consecutive wins within the batch.
* **Notes** Optional free-form details, such as links to the implementing classes or external model documentation.

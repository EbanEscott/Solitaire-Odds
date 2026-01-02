# Solitaire Odds

Have you ever wondered what the odds of winnng a game of Solitaire are? This project was built to demonstrate how an AI and Human can vibe to find the probability of winning a Solitaire game.

A well-shuffled 52-card deck has *52! permutations (about 8.1 × 10^67)*, so many that it dwarfs the *roughly 10^20 grains of sand on Earth*. In other words, almost every Solitaire deal you have ever seen is effectively a one-off in cosmic terms. Even at *one deal per second*, brute-forcing every deck order would take *around 2.6 × 10^60 years*, a timespan so huge the age of the universe does not even register on the same scale.

This means testing every deck permutation is impossible. Instead, we lean on AI and solid engineering to run repeatable regression test suites over large batches of randomly shuffled games, so we can measure performance statistically rather than brute-forcing every possible deal. The goal is not to “solve” all of Solitaire, but to apply a range of AI algorithms that reliably solve as many deals as possible and, in doing so, reveal the true probability of winning under real rules.

## Test Results

The last test run was performed at Jan 2, 2026 7:50 PM AEST.

| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |
|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|
| Rule-based Heuristics        | Search | 10000 | 418 | 4.18% ± 0.39% | 0.001s | 7.187s | 733.35 | 2 | Deterministic rule-based baseline; see [code](engine/src/main/java/ai/games/player/ai/RuleBasedHeuristicsPlayer.java). |
| Greedy Search                | Search | 10000 | 651 | 6.51% ± 1.21% | 0.003s | 31.046s | 242.42 | 3 | Greedy one-step lookahead using heuristic scoring; see [code](engine/src/main/java/ai/games/player/ai/GreedySearchPlayer.java). |
| Hill-climbing Search         | Search | 10000 | 1301 | 13.01% ± 0.66% | 0.002s | 17.181s | 96.20 | 5 | Local hill-climbing with restarts over hashed game states; see [code](engine/src/main/java/ai/games/player/ai/HillClimberPlayer.java). |
| Beam Search                  | Search | 10000 | 1022 | 10.22% ± 0.59% | 0.037s | 372.615s | 915.89 | 4 | Fixed-width beam search over move sequences; see [code](engine/src/main/java/ai/games/player/ai/BeamSearchPlayer.java). |
| Monte Carlo Search           | Search | 10000 | 1742 | 17.42% ± 0.74% | 1.782s | 17817.718s | 846.24 | 4 | Monte Carlo search running random playouts per decision; see [code](engine/src/main/java/ai/games/player/ai/MonteCarloPlayer.java). |
| A* Search                    | Search | 10000 | 1535 | 15.35% ± 0.71% | 1.881s | 18812.678s | 843.26 | 4 | A* search guided by a heuristic evaluation; see [code](engine/src/main/java/ai/games/player/ai/AStarPlayer.java). |
| OpenAI                       | LLM    | 100 | 13 | 13.00% ± 6.59% | 124.992s | 12499.187s | 168.69 | 2 | OpenAI gpt-5-mini via API; see [code](engine/src/main/java/ai/games/player/ai/OpenAIPlayer.java). |
| Alibaba                      | LLM    | 10 | 0 | 0.00% ± 0.00% | 235.863s | 2358.627s | 311.60 | 0 | Alibaba qwen3-coder:30b via Ollama; see [code](engine/src/main/java/ai/games/player/ai/OllamaPlayer.java) and [model](https://ollama.com/library/qwen3-coder). |

* **Player** Name of the decision or optimisation method or LLM-backed player being tested.
* **AI** Whether the method is an `LLM` (e.g., Ollama) or a search-based algorithm (e.g., A*, beam search, greedy).
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Games Won** Count of games successfully completed.
* **Win %** Percentage of games successfully completed (foundations fully built), reported as `win% ± 95% confidence interval` so that small improvements are statistically meaningful. The half-width shrinks roughly with `1/sqrt(games)` (e.g., ~±1.0% at 10k games, ~±0.5% at 40k games).
* **Avg Time/Game** Mean time taken to finish or fail a game.
* **Total Time** Sum of all time spent playing the batch of games.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Avg Score** Mean score based on whatever scoring system you’re using (e.g., Vegas, Microsoft, or custom).
* **Best Win Streak** Longest run of consecutive wins within the batch.
* **Notes** Free-form notes and clickable links to the implementing classes or external model pages.

> Why do search-based AI far out perform LLM's at games like Solitaire? In short: LLMs don't maintain or reason over complete card-game states, they don't do efficient tree-search or simulation, and so they can't reliably choose optimal moves in a structured card-game like Solitaire.
>
> LLMs can describe good play. They cannot compute good play.

## Players

In this project, a **player** is any strategy that chooses moves given a Solitaire game state. We group them into three families:

- **Search-based players (Engine)** — Run entirely inside the Java engine by exploring the game tree:
  - **Rule-based Heuristics**: Deterministic baseline using hand-crafted Solitaire rules; never calls an LLM.
  - **Greedy Search**: One-step lookahead that evaluates immediate moves with a heuristic score.
  - **Hill-climbing Search**: Local search that walks the state space, accepting only moves that improve a heuristic value (with restarts).
  - **Beam Search**: Multi-step search that keeps only the best `k` states at each depth to control branching.
  - **Monte Carlo Search**: Runs many random playouts from each state to estimate which moves lead to more wins.
  - **A\* Search**: Treats Solitaire as a shortest-path problem and uses an admissible-ish heuristic to guide exploration toward winning states.

- **LLM-backed players** — Use language models to propose moves:
  - **OpenAI**: Sends the current state and move options to an OpenAI chat model (e.g., `gpt-5-mini`) over HTTP and executes the model’s chosen move.
  - **Alibaba (Ollama)**: Uses the `qwen3-coder:30b` model via a local Ollama server; the engine prompts the model with a structured description of the board and legal moves and follows its recommendation.

> LLM-backed players performed very poorly because they do not keep any internal state of the game play. It was not until the prompts were significantly refined that they began to win any games. At that point, it felt like the prompts were encoding game rules rather than relying on the model's reasoning.

- **Neural MCTS player (AlphaSolitaire)** — Hybrid search + learned evaluation:
  - **AlphaSolitaire (MCTS + NN)**: Uses Monte Carlo Tree Search guided by a neural policy–value network trained in the `neural-network` module. The Java engine calls the Python service to evaluate states and choose statistically strong moves.

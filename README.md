# Solitaire Odds

Have you ever wondered what the odds of winnng a game of Solitaire are? This project was built to demonstrate how an AI and Human can vibe to find the probability of winning a Solitaire game.

A well-shuffled 52-card deck has *52! permutations (about 8.1 × 10^67)*, so many that it dwarfs the *roughly 10^20 grains of sand on Earth*. In other words, almost every Solitaire deal you have ever seen is effectively a one-off in cosmic terms. Even at *one deal per second*, brute-forcing every deck order would take *around 2.6 × 10^60 years*, a timespan so huge the age of the universe does not even register on the same scale.

This means testing every deck permutation is impossible. Instead, we lean on AI and solid engineering to run repeatable regression test suites over large batches of randomly shuffled games, so we can measure performance statistically rather than brute-forcing every possible deal. The goal is not to “solve” all of Solitaire, but to apply a range of AI algorithms that reliably solve as many deals as possible and, in doing so, reveal the true probability of winning under real rules.

The current best player is *A\* search* with a win rate of 36.87% ± 0.95% across 10,000 games, with a performance at 4.009s per game and an 8-game win streak.

## Test Results

The latest test run completed on Sep 28, 2026 at 4:33 PM AEST.

* **Player** Name of the decision or optimisation method or model-backed player being tested.
* **AI** Whether the method is an `LLM`, typed `Decision` model, or search-based algorithm.
* **Policy** Prompt and strategy condition used by a model-backed player; this is an experimental variable, not merely implementation detail.
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Games Won** Count of games successfully completed.
* **Win %** Percentage of games successfully completed (foundations fully built), reported as `win% ± 95% confidence interval` so that small improvements are statistically meaningful. The half-width shrinks roughly with `1/sqrt(games)` (e.g., ~±1.0% at 10k games, ~±0.5% at 40k games).
* **Avg Time/Game** Mean time taken to finish or fail a game, shown with adaptive units for readability.
* **Total Time** Sum of all time spent playing the batch, rounded to an appropriate combination of days, hours, minutes, or seconds.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Avg Score** Mean score based on whatever scoring system you’re using (e.g., Vegas, Microsoft, or custom).
* **Best Win Streak** Longest run of consecutive wins within the batch.
* **Notes** Free-form notes and clickable links to the implementing classes or external model pages.

### Search-based players

These search results were last run on Jan 26, 2026 at 9:44 PM AEST.

| Player                        | AI     | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |
|------------------------------|--------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|
| Rule-based Heuristics        | Search | 10000 | 418 | 4.18% ± 0.39% | 1ms | 7.187s | 733.35 | 2 | Deterministic rule-based baseline; see [code](engine/src/main/java/ai/games/player/ai/RuleBasedHeuristicsPlayer.java). |
| Greedy Search                | Search | 10000 | 651 | 6.51% ± 1.21% | 3ms | 31.046s | 242.42 | 3 | Greedy one-step lookahead using heuristic scoring; see [code](engine/src/main/java/ai/games/player/ai/GreedySearchPlayer.java). |
| Hill-climbing Search         | Search | 10000 | 1301 | 13.01% ± 0.66% | 2ms | 17.181s | 96.20 | 5 | Local hill-climbing with restarts over hashed game states; see [code](engine/src/main/java/ai/games/player/ai/HillClimberPlayer.java). |
| Beam Search                  | Search | 10000 | 1022 | 10.22% ± 0.59% | 37ms | 6m 13s | 915.89 | 4 | Fixed-width beam search over move sequences; see [code](engine/src/main/java/ai/games/player/ai/BeamSearchPlayer.java). |
| Monte Carlo Search           | Search | 10000 | 1742 | 17.42% ± 0.74% | 1.782s | 4h 57m | 846.24 | 4 | Monte Carlo search running random playouts per decision; see [code](engine/src/main/java/ai/games/player/ai/MonteCarloPlayer.java). |
| A* Search                    | Search | 10000 | 3687 | 36.87% ± 0.95% | 4.009s | 11h 8m | 92.53 | 8 | A* search guided by a heuristic evaluation; see [code](engine/src/main/java/ai/games/player/ai/AStarPlayer.java). |

### Model-backed players

The main evaluations use 100 independent random deals per configuration. Jev's ten-game P0 row is included as a diagnostic control so its supplied-policy result is not mistaken for an unguided comparison. The Codex CLI baselines completed across Sep 24-26, 2026; Haiku and TypeSafe AI Jev completed across Sep 27-28, 2026.

| Player                        | AI       | Policy           | Games Played | Games Won | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak | Notes |
|-------------------------------|----------|------------------|--------------|-----------|-------|---------------|------------|-----------|-----------------|-------|
| OpenAI Codex CLI (Sol)        | LLM      | P0 self-authored | 100 | 18 | 18.00% ± 7.53% | 7m 55s | 13h 11m | 59.68 | 2 | OpenAI `gpt-5.6-sol` via Codex CLI with medium reasoning, one persistent session per game, and engine guidance disabled; see [code](engine/src/main/java/ai/games/player/ai/CodexCliPlayer.java) and [notebook](experiments/notebooks/llm_player_research.ipynb). |
| OpenAI Codex CLI (Luna)       | LLM      | P0 self-authored | 100 | 5 | 5.00% ± 4.27% | 14m 51s | 1d 44m | 130.78 | 1 | OpenAI `gpt-5.6-luna` via Codex CLI with medium reasoning, one persistent session per game, engine guidance disabled, and a 200-move cap; see [code](engine/src/main/java/ai/games/player/ai/CodexCliPlayer.java) and [notebook](experiments/notebooks/llm_player_research.ipynb). |
| GitHub Copilot CLI (Haiku)    | LLM      | P0 self-authored | 100 | 2 | 2.00% ± 2.74% | 16m 35s | 1d 3h 38m | 80.59 | 1 | Anthropic `claude-haiku-4.5` via GitHub Copilot CLI with default reasoning, one persistent session per game, engine guidance disabled, and a 200-move cap; see [code](engine/src/main/java/ai/games/player/ai/CopilotCliPlayer.java) and [notebook](experiments/notebooks/llm_player_research.ipynb). |
| TypeSafe AI (Jev control)     | Decision | P0 unguided      | 10 | 0 | 0.00% (95% CI 0.0%-27.8%) | 1m 9s | 11m 30s | 200.00 | 0 | Diagnostic control: all ten games reached the move cap and 1,971 of 2,000 choices were stock turns; see [notebook](experiments/notebooks/llm_player_research.ipynb). |
| TypeSafe AI (Jev detailed)    | Decision | P1.5 supplied    | 100 | 10 | 10.00% ± 5.88% | 1m 6s | 1h 49m | 193.52 | 1 | TypeSafe AI `jev-1.13.0` via System One Choice API with compact observed-board history, engine guidance disabled, and a 200-move cap; see [code](engine/src/main/java/ai/games/player/ai/TypeSafePlayer.java) and [notebook](experiments/notebooks/llm_player_research.ipynb). |

> Why does A* search still outperform model-backed play at Solitaire? Search can explore and compare complete game states directly. Persistent-session LLMs and stateful decision models can retain enough context to win games, but they do not perform the same efficient tree search.
>
> Persistent context let Sol, Luna, and Haiku turn their own strategic descriptions into wins at 18%, 5%, and 2%. Jev reached 10% with the supplied P1.5 policy, after winning 0 of 10 games under the unguided P0 control. Deliberate search remains stronger.

## Players

In this project, a **player** is any strategy that chooses moves given a Solitaire game state. We group them into three families:

- **Search-based players (Engine)** — Run entirely inside the Java engine by exploring the game tree:
  - **Rule-based Heuristics**: Deterministic baseline using hand-crafted Solitaire rules; never calls an LLM.
  - **Greedy Search**: One-step lookahead that evaluates immediate moves with a heuristic score.
  - **Hill-climbing Search**: Local search that walks the state space, accepting only moves that improve a heuristic value (with restarts).
  - **Beam Search**: Multi-step search that keeps only the best `k` states at each depth to control branching.
  - **Monte Carlo Search**: Runs many random playouts from each state to estimate which moves lead to more wins.
  - **A\* Search**: Treats Solitaire as a shortest-path problem and uses an admissible-ish heuristic to guide exploration toward winning states.

- **Model-backed players** — Use conversational or typed models to choose moves:
  - **OpenAI Codex CLI (Sol)**: Asks the model to state its existing strategy before the deal, then keeps the strategy, every board, and every selected move in one persistent session for the game. The engine supplies legal moves but no strategic guidance.
  - **OpenAI Codex CLI (Luna)**: Uses the same persistent-session P0 protocol as Sol, with a separate session for each random deal and no engine-supplied strategy or guidance.
  - **Anthropic (GitHub Copilot CLI)**: Runs selectable Claude models through a Copilot subscription using the same P0 prompt and one persistent session per random deal. The engine supplies legal moves but no strategy or guidance.
  - **OpenAI**: Sends the current state and move options to an OpenAI chat model (e.g., `gpt-5-mini`) over HTTP and executes the model’s chosen move.
  - **Alibaba (Ollama)**: Uses the `qwen3-coder:30b` model via a local Ollama server; the engine prompts the model with a structured description of the board and legal moves and follows its recommendation.
  - **TypeSafe AI Jev**: Sends the current visible board, compact observations from every earlier turn, and legal commands to a System One Choice question. Jev returns a typed option with probabilities and confidence; see [code](engine/src/main/java/ai/games/player/ai/TypeSafePlayer.java).

> The current LLM benchmark asks the model to state the strategy it already knows, then retains that strategy, earlier boards, commands, and feedback in one persistent session per game. Older stateless API and Ollama results remain available in Git history.
>
> Model-backed experiments can use `-Dgame.prompt.profile=p0|detailed`. The default `p0` preserves the self-authored-strategy baseline above; `detailed` supplies the engine's rules, decision priorities, progress requirements, and expanded interface as the shared P1.5 profile. The older `llm.prompt.profile` property remains accepted for reproducibility. Result logs record both profile and version.

- **Neural MCTS player (AlphaSolitaire)** — Hybrid search + learned evaluation:
  - **AlphaSolitaire (MCTS + NN)**: Uses Monte Carlo Tree Search guided by a neural policy–value network trained in the `neural-network` module. The Java engine calls the Python service to evaluate states and choose statistically strong moves.

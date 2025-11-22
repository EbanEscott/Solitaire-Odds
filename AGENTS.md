# Agents

## Testing and README refresh workflow

When asked complete these tasks:

1. **Run the Tests** found in the `cards/src/test/java/ai/games/results` package.
2. **Gather the Test Results** from the logging info for each test suite.
3. **Format a Table** using the results and the example table below.
4. **Paste the Table** into the `cards/README.md` and insert the date/time the tests were run.

| Algorithm                     | Games Played | Win % | Avg Time/Game | Total Time | Avg Moves | Best Win Streak |
|------------------------------|--------------|-------|----------------|-------------|-----------|-----------|
| Rule-based Heuristics        | 500          | 12%   | 0.3s           | 150s        | 95        | 3                |
| Greedy Search                | 500          | 18%   | 0.5s           | 250s        | 105       | 4                |

* **Algorithm** Name of the decision or optimisation method being tested.
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Win %** Percentage of games successfully completed (foundations fully built).
* **Avg Time/Game** Mean time taken to finish or fail a game.
* **Total Time** Sum of all time spent playing the batch of games.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Best Win Streak** Longest run of consecutive wins within the batch.

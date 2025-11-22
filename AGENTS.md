# Agents

## README refresh workflow

When asked complete these tasks:

1. **Run the Tests**
2. **Gather the Test Results**
3. **Format a Table** Use the Results and the example table below.
4. 

| Algorithm                     | Games Played | Win % | Avg Time/Game | Total Time | Avg Moves | Avg Score | Best Win Streak | Avg Search Depth |
|------------------------------|--------------|-------|----------------|-------------|-----------|-----------|------------------|-------------------|
| Rule-based Heuristics        | 500          | 12%   | 0.3s           | 150s        | 95        | 4200      | 3                | 0                 |
| Greedy Search                | 500          | 18%   | 0.5s           | 250s        | 105       | 4800      | 4                | 1                 |
| Hill Climbing                | 500          | 22%   | 0.7s           | 350s        | 110       | 5000      | 5                | 4                 |
| Simulated Annealing          | 500          | 26%   | 1.1s           | 550s        | 115       | 5100      | 6                | 6                 |
| Genetic Algorithm            | 500          | 31%   | 3.5s           | 1750s       | 120       | 5600      | 8                | —                 |
| Minimax                      | 500          | 35%   | 12s            | 6000s       | 140       | 6200      | 10               | 8                 |
| Alpha-Beta Pruning           | 500          | 38%   | 6s             | 3000s       | 140       | 6300      | 12               | 10                |
| Monte Carlo Simulation       | 500          | 41%   | 2s             | 1000s       | 130       | 6400      | 12               | —                 |
| MCTS                         | 500          | 55%   | 4s             | 2000s       | 150       | 6800      | 16               | 30                |
| Tabular RL                   | 500          | 52%   | 1.5s           | 750s        | 135       | 6700      | 18               | —                 |
| Deep RL                      | 500          | 63%   | 0.1s           | 50s         | 125       | 7200      | 21               | —                 |
| Model-based RL               | 500          | 68%   | 0.08s          | 40s         | 120       | 7400      | 25               | —                 |
| Evolution Strategies         | 500          | 60%   | 0.5s           | 250s        | 130       | 7000      | 15               | —                 |
| Multi-agent RL               | 500          | 70%   | 0.2s           | 100s        | 140       | 7600      | 28               | —                 |

* **Algorithm** Name of the decision or optimisation method being tested.
* **Games Played** Total number of solitaire games the algorithm attempted.
* **Win %** Percentage of games successfully completed (foundations fully built).
* **Avg Time/Game** Mean time taken to finish or fail a game.
* **Total Time** Sum of all time spent playing the batch of games.
* **Avg Moves** Average number of moves (legal actions) the algorithm performed per game.
* **Avg Score** Mean score based on whatever scoring system you’re using (e.g., Vegas, Microsoft, or custom).
* **Best Win Streak** Longest run of consecutive wins within the batch.
* **Avg Search Depth** Typical depth of lookahead or tree expansion during decision-making (applies to algorithms that search).

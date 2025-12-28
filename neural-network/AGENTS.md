The goal of this experiment is to create a solitaire model that is trained on game states. We want to evetually use this model in *AlphaSolitaire*.

## Agent guidance (neural-network module)

Python code in this `neural-network` directory should:

- Keep modules importable from the project root using `python -m src.<module>` or direct `python` invocations.
- Prefer relative imports within the `src` package once introduced.
- Avoid adding heavy dependencies beyond those listed in `requirements.txt` without discussion.
- Keep training scripts, services, and notebooks focused on the Solitaire modeling experiment (no unrelated utilities).

For the high-level AlphaSolitaire architecture and integration plan, see:

- `engine/src/main/java/ai/games/player/ai/alpha/README.md`

---

## Next session: TODO checklist

**Progress Update (Dec 27, 2025):**

### ✅ Completed
1. **Review current training setup** — Examined training infrastructure and identified encoding matches
2. **Regenerate training data** — Generated 44k+ samples from A* player, separated episode.log from game.log
3. **Debug and strengthen neural training** — Trained policy-value network: policy acc 75%, value acc 99%

### 🔄 In Progress: Step 4 - Investigate MCTS Behaviour

**Current AlphaSolitaire Results (10 games):**
- Win rate: **0% (0/10)**
- Average moves: **1000** (all games hit the move cap)
- Policy accuracy on A* moves: **75%** (model learned 3/4 of moves)
- Value prediction: **99%** (excellent at predicting win/loss)

**Observation:** Despite strong policy and value accuracy on the training data, AlphaSolitaire achieves 0% wins with:
- 32 MCTS simulations per move
- 12-move max depth per simulation
- 1.5 PUCT exploration constant

**Hypothesis:** The policy guidance may not be sharp enough to reduce the search space effectively, or MCTS parameters need tuning.

**Action Items for Step 4:**
- Add detailed MCTS logging to record:
  - Simulation count per move
  - Visit distribution over actions (top-3 moves and their visit counts)
  - Root value estimate and chosen move quality metrics
  - Number of times search hits max depth or gets stuck
- Run with increased simulation budget (e.g., 100+ sims vs 32)
- Analyze logs to understand if the problem is:
  - Policy distribution too flat (all moves equally likely)
  - Value estimates not informative (too close to 0.5)
  - Search getting trapped in loops

When you next pick this up:

1. **Review current training setup** ✅
   - Examined training infrastructure: 290-dim state, dynamic action space from logs
   - Confirmed encoding matches Java engine

2. **Regenerate or expand training data** ✅
   - Refactored episode logging (new `EpisodeLogger.java`)
   - Generated 3,664 episode steps (44,238 samples) from 10 A* games
   - Updated `engine/README.md` with episode generation instructions

3. **Debug and strengthen neural training** ✅
   - Trained on 44k samples: policy acc 75%, value acc 99%
   - Model converges smoothly (no overfitting)
   - Checkpoint saved and verified with `train_stub.py`
   - Updated `neural-network/README.md` with training output and metric explanations

4. **Investigate MCTS behaviour** (IN PROGRESS)
   - Evaluated AlphaSolitaire: 0% win rate (0/10), avg 1000 moves (move cap hit)
   - Policy accuracy on test data: 75% (learned 3/4 of A* moves)
   - Value prediction accuracy: 99% (excellent win/loss classification)
   - Next: Add MCTS logging to diagnose search quality (visit distributions, depth analysis)
   - Try: Increase simulation budget and tune PUCT constant
   - Analyze: Whether policy priors are sharp enough or search is getting trapped

5. **Re-run evaluation of AlphaSolitaire** (TODO)
   - Run 100–500 game evaluation with diagnostic MCTS logs
   - Record win rate, move distribution, and failure modes
   - Update `engine/src/main/java/ai/games/player/ai/alpha/README.md` with findings

6. **Self-Play Training Loop** (TODO)
   - Implement RL loop: MCTS selection → self-play → network update
   - Add checkpointing and model comparison
   - Close the loop for continuous improvement
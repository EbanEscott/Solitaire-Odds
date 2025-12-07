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

Context: the current AlphaSolitaire player ran 100 games, capped at 1000 moves per game, and achieved a 0% win rate. The next working session should focus on improving the neural network training and validating the MCTS loop.

When you next pick this up:

1. **Review current training setup**
   - Inspect `src/train_policy_value.py` and any existing checkpoints in `checkpoints/`.
   - Confirm how many episodes and which logs were used to train the current model.
   - Check that the action-space encoding and state encoding still match the Java engine.

2. **Regenerate or expand training data**
   - Use the Java engine to log a fresh batch of episodes (tune seeds and baseline players if needed).
   - Verify that logs are being written where the Python data pipeline expects them.

3. **Debug and strengthen neural training**
   - Re-run training with more data and/or tweaked hyperparameters (learning rate, epochs, model size).
   - Track basic metrics: training/validation loss, policy accuracy, and calibration of the value head.
   - Save a new checkpoint and note its config (hyperparameters, data size).

4. **Investigate MCTS behaviour**
   - Add logging around the MCTS loop to record:
     - Number of simulations per move.
     - Distribution of visit counts over actions.
     - Value estimates for chosen vs rejected moves.
   - Confirm that the search respects the 1000-move cap and terminates sensibly when stuck.

5. **Re-run evaluation of AlphaSolitaire**
   - Run a controlled 100–500 game evaluation using the updated model and MCTS.
   - Record win rate, average moves, and any failure modes (e.g., loops, repeated states).
   - Capture key findings in `engine/src/main/java/ai/games/player/ai/alpha/README.md` or a short notes file.

6. **Self-Play Training Loop**
   - Implement an RL training script that:
     - Runs many self-play games using MCTS + current network.
     - Records (state, improved policy from MCTS, final outcome) tuples.
     - Periodically updates the network on this data.
   - Add mechanisms for:
     - Checkpointing models.
     - Evaluating new models against older versions or baselines.
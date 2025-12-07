The goal of this experiment is to create a solitaire model that is trained on game states. We want to evetually use this model in *AlphaSolitaire*.

## Agent guidance (neural-network module)

Python code in this `neural-network` directory should:

- Keep modules importable from the project root using `python -m src.<module>` or direct `python` invocations.
- Prefer relative imports within the `src` package once introduced.
- Avoid adding heavy dependencies beyond those listed in `requirements.txt` without discussion.
- Keep training scripts, services, and notebooks focused on the Solitaire modeling experiment (no unrelated utilities).

---

Here is a concise, direct introduction and architectural summary suitable for design docs, technical papers, or engineering briefs.

# **AlphaSolitaire: Introduction**

AlphaSolitaire is a self-learning game engine for Klondike Solitaire inspired by the architectural principles of AlphaGo and AlphaZero.
Solitaire is a single-agent, partially observable combinatorial puzzle with deep tactical dependencies and long decision chains. Traditional rule-based and search-based methods excel at constrained optimisation but require extensive hand-crafted heuristics and cannot generalise effectively to unseen configurations.

AlphaSolitaire addresses this by combining a **neural policy/value network** with **Monte Carlo Tree Search (MCTS)**. The neural network learns to evaluate Solitaire states and prioritise promising actions, while MCTS performs structured look-ahead to identify the most robust continuation. Through large-scale self-play and reinforcement learning, AlphaSolitaire progressively improves its ability to reason under uncertainty and navigate complex, multi-step sequences that emerge in practical play.

The result is an adaptive engine capable of discovering high-performance strategies without manual heuristics, retaining the strengths of classical search while introducing the flexibility of learned evaluation.

---

# **AlphaSolitaire Architecture Summary**

AlphaSolitaire consists of three tightly integrated components:

## **1. State Encoder**

The game state (tableau, foundations, waste, stock, and hidden-card structure) is mapped into a fixed-size tensor representation. This encoding preserves rank, suit, visibility, column structure, movable groups, and remaining deck composition. The encoder provides a complete, machine-interpretable description of the current board.

## **2. Policy–Value Neural Network**

A compact residual network processes the encoded state and outputs:

* **Policy vector**: A probability distribution over all legal Solitaire moves, used as prior guidance for tree search.
* **Value estimate**: A scalar prediction of long-term success (win probability or shaped reward).

The network does not search; it provides heuristics that shape the search process.

## **3. Monte Carlo Tree Search (MCTS)**

MCTS is the core decision engine. For each move:

* The policy prior biases expansion toward promising actions.
* The value estimate evaluates leaf nodes.
* Node visit statistics are backpropagated to refine action confidence.
* After N simulations, the action with the highest visit count is selected.

MCTS enables multi-step reasoning, backtracking, and forward planning—capabilities the neural network alone cannot achieve.

## **4. Self-Play Reinforcement Learning**

AlphaSolitaire trains by repeatedly playing Solitaire from scratch:

* MCTS selects actions during training episodes.
* Each step produces (state, improved policy, outcome) tuples.
* The neural network is updated to match MCTS’s improved move distributions and the final game result.
* Over time, the network converges toward heuristics that reduce branching and improve long-term success.

This closed learning loop gradually produces a strong, domain-general Solitaire agent without hand-crafted rules.

---

# **Experiment Plan: Solitaire Game-State Model**

## **Goal**

Create a Solitaire policy–value model trained directly on game states, suitable for integration into the AlphaSolitaire engine as its neural evaluation core.

We will follow **Option 2 / Option 3**:

- Treat the existing Java engine in `/Users/ebo/Code/solitaire/engine/src/main/java/ai/games` as the **authoritative Solitaire simulator and evaluation harness**.
- Use this Python repo (`/Users/ebo/Code/solitaire`) as the **modeling stack** (state encoding, PyTorch networks, training, analysis).
- Start by generating data from the Java engine via logging, and only consider a pure-Python environment later if needed for research convenience.

## **Milestones**

1. **Define State Representation (Java ↔ Tensor)**
   - Specify a concrete tensor encoding for Klondike Solitaire (tableau, foundations, waste, stock, hidden cards).
   - Map this encoding onto the existing Java state representation in `ai.games` so that every logged state can be deterministically converted to a tensor.
   - Document tensor shapes, channel semantics, and normalisation in a design note (e.g., `docs/state_encoding.md` in this repo).

2. **Java-Side Episode Logging**
   - Extend or configure the Java engine to log full game episodes using existing loggers:
     - Per-step fields: raw game state, list of legal moves, chosen move (from a baseline solver), step index.
     - Per-episode fields: deal seed, outcome (win/loss/stuck), total moves, solver identity.
   - Choose a stable on-disk format (e.g., JSONL or compact binary) and output location (e.g., `logs/solitaire_episodes/` under the engine repo).
   - Add a simple CLI or test harness in the Java project to generate batches of episodes on demand.

3. **Python Data Pipeline for Logged Game States**
   - In this repo, implement a data loader that:
     - Reads the Java-generated logs.
     - Applies the agreed `encode_state(...) -> tensor` mapping.
     - Produces `(state_tensor, policy_target, value_target)` samples suitable for PyTorch training.
   - Decide on a caching/serialization strategy for large datasets (e.g., saving pre-encoded tensors as `.pt` shards).

4. **Policy–Value Network Prototype**
   - Implement a first PyTorch model class (e.g., a small residual CNN) that:
     - Accepts encoded state tensors.
     - Outputs a policy vector over a fixed action space.
     - Outputs a scalar value in [-1, 1] representing expected outcome.
   - Start with a small configuration for fast iteration, with hooks for scaling depth/width later.

5. **Supervised Pretraining (Optional but Recommended)**
   - Use trajectories from existing heuristic or search-based solvers in the Java engine to:
     - Train the policy head to imitate their move choices.
     - Train the value head on observed outcomes.
   - Establish a baseline loss curve and simple evaluation metrics (e.g., validation accuracy for move prediction, value MSE).

6. **MCTS Integration**
   - Implement a basic MCTS loop that:
     - Calls the policy–value network to expand nodes.
     - Maintains visit counts and Q-values per edge.
     - Selects actions using a PUCT-style formula.
   - Plug MCTS into the environment and verify that it plays complete games using the current network.

7. **Self-Play Training Loop**
   - Implement an RL training script that:
     - Runs many self-play games using MCTS + current network.
     - Records (state, improved policy from MCTS, final outcome) tuples.
     - Periodically updates the network on this data.
   - Add mechanisms for:
     - Checkpointing models.
     - Evaluating new models against older versions or baselines.

8. **AlphaSolitaire Integration Hooks**
   - Define a stable model interface (e.g., `evaluate_state(state) -> (policy, value)`).
   - Decide on model export format (PyTorch `state_dict`, TorchScript, or ONNX) for integration into the main AlphaSolitaire engine.
   - Document how the engine calls into this model and how versioning/updates are handled.

9. **Evaluation and Analysis**
   - Define key metrics (win rate, average moves, average reward, search time).
   - Run controlled experiments comparing:
     - Rule-based / heuristic baselines.
     - MCTS with and without the learned network.
     - Different network sizes or training regimes.
   - Capture findings in a short results summary or technical note.

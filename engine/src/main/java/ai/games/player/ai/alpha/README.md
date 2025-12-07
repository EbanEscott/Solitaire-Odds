# AlphaSolitaire Player (Java side)

This directory contains the Java integration for the AlphaSolitaire player:

- `AlphaSolitairePlayer` — a Spring profile-gated `Player` implementation that calls into the Python policy–value service.
- `AlphaSolitaireClient` — HTTP client wrapper for the Python service.
- `AlphaSolitaireRequest` / `AlphaSolitaireResponse` — DTOs describing the JSON protocol between Java and Python.

The high-level design of AlphaSolitaire is shared between the engine and the Python modeling stack and is captured below for easy reference when working on this player.

---

# AlphaSolitaire: Introduction

AlphaSolitaire is a self-learning game engine for Klondike Solitaire inspired by the architectural principles of AlphaGo and AlphaZero.
Solitaire is a single-agent, partially observable combinatorial puzzle with deep tactical dependencies and long decision chains. Traditional rule-based and search-based methods excel at constrained optimisation but require extensive hand-crafted heuristics and cannot generalise effectively to unseen configurations.

AlphaSolitaire addresses this by combining a **neural policy/value network** with **Monte Carlo Tree Search (MCTS)**. The neural network learns to evaluate Solitaire states and prioritise promising actions, while MCTS performs structured look-ahead to identify the most robust continuation. Through large-scale self-play and reinforcement learning, AlphaSolitaire progressively improves its ability to reason under uncertainty and navigate complex, multi-step sequences that emerge in practical play.

The result is an adaptive engine capable of discovering high-performance strategies without manual heuristics, retaining the strengths of classical search while introducing the flexibility of learned evaluation.

---

# AlphaSolitaire Architecture Summary

AlphaSolitaire consists of three tightly integrated components:

## 1. State Encoder

The game state (tableau, foundations, waste, stock, and hidden-card structure) is mapped into a fixed-size tensor representation. This encoding preserves rank, suit, visibility, column structure, movable groups, and remaining deck composition. The encoder provides a complete, machine-interpretable description of the current board.

## 2. Policy–Value Neural Network

A compact residual network processes the encoded state and outputs:

* **Policy vector**: A probability distribution over all legal Solitaire moves, used as prior guidance for tree search.
* **Value estimate**: A scalar prediction of long-term success (win probability or shaped reward).

The network does not search; it provides heuristics that shape the search process.

## 3. Monte Carlo Tree Search (MCTS)

MCTS is the core decision engine. For each move:

* The policy prior biases expansion toward promising actions.
* The value estimate evaluates leaf nodes.
* Node visit statistics are backpropagated to refine action confidence.
* After N simulations, the action with the highest visit count is selected.

MCTS enables multi-step reasoning, backtracking, and forward planning—capabilities the neural network alone cannot achieve.

## 4. Self-Play Reinforcement Learning

AlphaSolitaire trains by repeatedly playing Solitaire from scratch:

* MCTS selects actions during training episodes.
* Each step produces (state, improved policy, outcome) tuples.
* The neural network is updated to match MCTS’s improved move distributions and the final game result.
* Over time, the network converges toward heuristics that reduce branching and improve long-term success.

This closed learning loop gradually produces a strong, domain-general Solitaire agent without hand-crafted rules.
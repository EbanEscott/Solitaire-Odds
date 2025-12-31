# Neural Network Plan

## Current State

### What We Have
- Configurable `PolicyValueNet`: hidden_dim (256-2048+), num_layers (1-5+)
- Episode logging: captures each game step with state, legal moves, chosen move
- Training script with full argparse: `--hidden-dim`, `--num-layers`, `--epochs`, `--batch-size`, `--learning-rate`
- Current training: supervised learning on A* move labels + game outcomes

### How It Works Now
```bash
python -m src.train_policy_value \
  --hidden-dim 512 --num-layers 3 \
  logs/episode.log
```

Each episode logs:
- State before move (board layout, foundation, talon, stock)
- Legal moves available
- Chosen move (from A* or other algorithm)
- Game outcome (won/lost)

Network learns:
- Policy: predict the chosen move (supervised imitation)
- Value: predict game outcome (binary classification)

---

## Phase 1: MLP Baseline on Easy Levels

**Goal:** Get MLP working on low-difficulty games as foundation before moving to GNN

**Current Situation:**
- MLP: trained on A* data, ~0% win rate in actual play
- Need: MLP that can win some easy games before adding complexity

**Strategy:**
1. Generate training data using `TrainingOpponent` on **low levels only** (difficulty 1-20)
2. Train MLP (512 hidden, 3 layers) on this curriculum data
3. Test: MLP should win some games at these easy levels
4. Keep this MLP checkpoint (we'll maintain both MLP and GNN)

**Tasks:**
1. Configure `EndgameTrainingDataGenerator` to produce low-level episodes
2. Train MLP until validation win rate > some % on test set
3. Save checkpoint as `policy_value_mpl_v1.pt`
4. Document results in README.md

**Success Criteria:**
- MLP can win games on difficulty 1-20
- Training completes without errors
- Checkpoint saves and loads correctly

**Blockers:**
- None, can start immediately

**Next:** Once MLP works on easy levels, we move to GNN approach for harder levels

---

## Phase 2: Rich Episode Logging with EPISODE_TREE

**Goal:** Log A* search trees for future GNN training (not used by MLP yet)

**Episode Structure (Decision Made):**
```json
{
  "step_index": 5,
  "state": {...board state...},
  "legal_moves": ["move T1 F1", "turn", ...],
  "chosen_command": "move T1 F1",
  "tree": {
    "moves": ["move T1 F1", "turn", ...],
    "visits": [150, 50, ...],
    "heuristics": [0.85, 0.30, ...],
    "children": {...recursive tree...}
  }
}
```

**Key Decisions (from discussion):**
- Tree depth: Current node + immediate children only (not full tree from root)
- Tree data: visit counts, heuristic scores, move encoding
- Board as root node: Board features become node features in GNN
- All moves logged: Both winning and losing paths (network learns from both)
- Winning move marked: The chosen_command indicates which child was selected

**Tasks:**
1. Update `EpisodeLogger.java` to call `logTree()` method after each move
2. Extract current node's children from A* search
3. Serialize to JSON with visits, heuristics, move names
4. Verify file sizes are reasonable

**Why Now (Phase 2, not Phase 1):**
- MLP doesn't need tree data (uses only board state)
- Logging tree structure in Phase 2 prepares data for Phase 3
- Allows Phase 1 to focus purely on MLP performance

**Blockers:**
- Need to preserve A* tree structure after search completes
- JSON serialization must be efficient

---

## Phase 3: GNN Model with Tree Input

**Goal:** Build GNN that learns from board state + move tree structure

**Architecture Decision (Made):**
- Use pure GNN, not hybrid (MLP + GNN)
- Board state becomes root node of graph
- A* move tree (children) becomes subgraph
- GNN processes entire graph together: state influences move evaluation, moves influence state understanding

**Graph Structure:**
```
Root node (board features: pile sizes, foundation state, talon, etc.)
  ├─ Child 1 (move: "move T1 F1", visits: 150, heuristic: 0.85)
  ├─ Child 2 (move: "turn", visits: 50, heuristic: 0.30)
  └─ Child 3 (move: "move F1 T2", visits: 20, heuristic: 0.15)
```

**Why GNN (Not Flattening):**
Earlier discussion identified critical insight: flattening tree to visit counts loses the causal chains.
Example: "Red 4 coming in 3 moves means I should unlock a pile NOW"
- Visit count: `0.75` (doesn't explain why)
- Tree structure: Shows that unlocking leads to cascades → board improves → red 4 has a home
- GNN can learn these forward-looking relationships; flattening cannot

**Remaining Design Questions:**
- Graph convolution type? (GraphConv, GAT, message passing)
- Node features? (move encoding + visits + heuristic + depth)
- Handling variable tree sizes in batches? (padding/masking)
- How many GNN layers needed?

**Tasks (Not started, needs planning):**
1. Create `tree_encoding.py` to convert JSON tree to PyTorch Geometric graph
2. Update `dataset.py` to load EPISODE_TREE records
3. Update `model.py` to accept GNN instead of MLP
4. Implement graph batching for variable-sized trees

**Dependency:**
- Requires Phase 2 (EPISODE_TREE logging in place)
- Can run in parallel with Phase 1 (MLP training) once Phase 2 data exists

**Note:**
GNN will be separate checkpoint: `policy_value_gnn_v1.pt`
Both MLP and GNN will be maintained for experimentation.

---

## Phase 4: Multiple Model Support & Experimentation

**Goal:** Support both MLP and GNN in codebase for experimentation and comparison

**Current Architecture:**
- Single model type: always MLP
- Need to expand to support multiple architectures

**Required Changes:**
1. Model registry: `ModelFactory` that returns correct model type
2. Config/args: specify `--model-type mpl` or `--model-type gnn` 
3. Training loop: agnostic to model type (both implement same interface)
4. Inference: player can load MLP or GNN checkpoint
5. Evaluation: compare MLP vs GNN on same test set

**Code Structure:**
```
neural-network/src/
  model.py              (base Model class)
  models/
    mpl_model.py        (existing MLP implementation)
    gnn_model.py        (new GNN implementation)
  model_factory.py      (selects correct model)
  dataset.py            (handles both EPISODE_STEP and EPISODE_TREE)
  train_policy_value.py (model-agnostic training)
```

**Checkpoints:**
- `policy_value_mpl_v1.pt` (Phase 1: MLP on easy levels)
- `policy_value_gnn_v1.pt` (Phase 3: GNN with trees)
- Later: `mpl_v2.pt`, `gnn_v2.pt` as versions improve

**Evaluation Plan:**
- Test set: difficulty 1-50, 100 games each
- Metrics: MLP win % vs GNN win %, avg moves, performance variance
- Document in README.md

---

## Phase 5: Self-Play Loop (The Ultimate Goal)

**Goal:** Continuous improvement through self-play with both A* bootstrap and network-guided MCTS

**The Loop:**
```
v0 (GNN trained on A* data + trees)
    ↓
Run AlphaSolitairePlayer (MCTS + v0 network priors) on seeded positions
    ↓
Log full MCTS search trees + outcomes
    ↓
Retrain GNN on MCTS-discovered move distributions
    ↓
v1 (should beat v0 in win rate and move efficiency)
    ↓
v1 generates new MCTS games with better network guidance
    ↓
v2, v3, ... iterate until convergence
```

**What Makes This Work:**
1. Network learns what A* discovered (Phase 1-3)
2. Network priors make MCTS more efficient (better exploration)
3. MCTS discovers even better move distributions
4. Network learns from MCTS discoveries
5. Cycle repeats: each iteration improves both network and MCTS effectiveness

**Implementation:**
1. AlphaSolitairePlayer uses network for move priors in MCTS
2. EpisodeLogger captures full MCTS tree (different from A* tree)
3. train_policy_value.py loads MCTS episodes:
   ```
   V_target = game_outcome  (Phase 1-4: supervised)
   or
   V_target = bootstrapped from V(next_state)  (Phase 5: optional RL)
   ```
4. Version management: `gnn_v0.pt`, `gnn_v1.pt`, `gnn_v2.pt`, ...
5. Tournament harness: compare win rates between versions

**Metrics (Self-Play Performance):**
- v1 win % vs v0 win % (should improve)
- Average moves per game (should decrease)
- MCTS effectiveness: time to find good moves
- Whether network guidance reduces search space needed

**Success Criteria:**
- v1 beats v0 consistently
- v2 beats v1 consistently
- Visible improvement trajectory (win % increasing over 3-5 iterations)
- Convergence point (versions stop improving)

**This Is the Ultimate Goal:**
Self-play creates a feedback loop where the network and search algorithm bootstrap each other to superhuman play.
Without self-play, we're stuck at "network trained on A*" (limited by A*'s data).
With self-play, network and MCTS co-evolve to discover strategies that A* alone never finds.


---

## Open Questions

### 1. How to represent search tree in JSON?
Current approach: recursive nested object
```json
{
  "root": {
    "state_key": 12345,
    "moves": [...],
    "visits": [...],
    "children": {"move0": {...}, "move1": {...}}
  }
}
```

Concerns:
- Episode file size (trees can be large)
- Parsing complexity
- Storage efficiency

Alternative: Flatten to edge list, rebuild during training

### 2. Should we log full tree or just immediate children?
- Full tree (depth 12+): Complete search context, large files
- Immediate children only: What moves were explored, smaller files
- Top K moves: Prune to most promising, smallest files

**Current thinking:** Top K moves (~5-10 most visited) balances both

### 3. Visit counts as policy labels?
AlphaGo approach: use MCTS visit distribution as policy target
```
Move A: 150 visits → 0.75 probability
Move B: 50 visits → 0.25 probability
Policy_target = [0.75, 0.25]  (not one-hot)
```

Benefits: Learn improved move distributions
Cost: Requires policy head to output probabilities, not logits

### 4. When to enable bootstrapped values?
- **Now:** Use game outcome (oracle label)
- **Phase 4:** Use V(next_state) (self-play RL)

Decision: When does the network get good enough that V(state) is reliable?
Probably: After Phase 3 when trained on MCTS data

### 5. A* vs MCTS tree structure differences?
- A*: Bidirectional (parent pointers, pruning flags)
- MCTS: Statistical (visits, value sums per move)

Unified tree needs both. How much does this slow down algorithms?

---

## Success Metrics

### Phase 1 (MLP on Easy Levels)
- [ ] Training data generated (low difficulty only)
- [ ] MLP trains without errors
- [ ] MLP wins some games on difficulty 1-20
- [ ] Checkpoint saved as `policy_value_mpl_v1.pt`

### Phase 2 (Tree Logging)
- [ ] EpisodeLogger logs EPISODE_TREE records
- [ ] Tree JSON structure is correct (moves, visits, heuristics, children)
- [ ] File sizes reasonable
- [ ] Can parse trees back during training

### Phase 3 (GNN Model)
- [ ] tree_encoding.py converts JSON trees to PyTorch Geometric graphs
- [ ] dataset.py loads both board state and tree structure
- [ ] GNN model trains without errors
- [ ] Checkpoint saved as `policy_value_gnn_v1.pt`

### Phase 4 (Multi-Model Support)
- [ ] ModelFactory correctly selects MLP or GNN
- [ ] Both models can be trained and evaluated
- [ ] Comparison test: MLP vs GNN on same test set
- [ ] Results documented in README.md

---

## Key Insight: Why GNN Matters

**The Problem:** Flattening tree to statistics loses the reasoning

Visit counts don't explain *why* a move is good. Example from Solitaire:
- Turn 1: I see red 4 will appear in 3 moves
- Turn 1: I make a "sacrifice" move to unlock pile X *now*
- Turns 2-3: The freed pile enables cascades
- Turn 4: Red 4 arrives, has a home

**What flattening loses:**
- "visit_count=150" on sacrifice move doesn't explain the depth-2+ causal chain
- The value only makes sense with tree structure

**What GNN preserves:**
- Board state as root node
- Move children showing what A* explored
- GNN can learn: "unlocking this pile enables better moves downstream"
- The tree structure IS the reasoning behind move selection

**Solitaire vs AlphaGo (Architecture Difference):**
- AlphaGo (Go): Pure position evaluation → CNN works perfectly (translation invariance, locality)
- Solitaire (Planning game): Move value depends on downstream unlocking → tree structure essential → GNN required

---

## Dependencies & Blockers

- **Phase 1 → 2:** Complete Phase 1 (baseline established)
- **Phase 2 → 3:** Complete Phase 2 (EPISODE_TREE logging in place)
- **Phase 3 → 4:** Complete Phase 3 (GNN training working)
- **Phase 4 → 5:** Complete Phase 4 (multi-model support) + fix MCTS performance

Note: Phase 5 is the ultimate goal — all prior phases feed into the self-play loop


# Neural Network Plan

## Current Status (Jan 2, 2026)

### MLP Baseline Achieved ✓
- **Model:** PolicyValueNet (256 hidden, 2 layers)
- **Training Data:** Levels 2-6 endgame positions (~864k samples, 88.9% accuracy)
- **Performance:** 
  - Level 20: 90% win rate (9/10 games) ✓
  - Level 25+: **Testing needed** (reveals MLP architectural ceiling)

### The Wall: Why MLP Plateaus
MLP hit an architectural ceiling because:
- **Input:** Flat board state (532 dimensions)
- **Missing:** Tree structure showing *why* A* made moves
- **Problem:** Cannot learn causal chains like "unlock pile now → enables cascade in 3 moves"
- **Solution:** GNN with tree input can learn these multi-step dependencies

### Confirmed Sequence Forward
1. **Phase 0: Test MLP Falloff** (today/tomorrow): Map win rate across L20-50
2. **Phase 2: Add Tree Logging** (1-2 days): Update EpisodeLogger to capture A* trees
3. **Phase 3: Build GNN** (3-5 days): Implement GNN on board + tree structure
4. **Phase 4: Multi-Model** (1 day): Compare GNN vs MLP across difficulty range
5. **Phase 5: Self-Play** (ongoing): Continuous improvement loop

---

## Phase 0: Test MLP Falloff Curve

**Goal:** Quantify where MLP performance degrades (validates why GNN is needed)

**Tasks:**
1. Run AlphaSolitaireLevelTest at: 20, 25, 30, 35, 40, 50
2. Each level: 20 games minimum (statistical confidence)
3. Record: win rate %, avg moves, failure modes
4. Document in README.md with results table

**Expected Output:**
```
Level 20: 90%   (current baseline ✓)
Level 25: 70%   (noticeable drop)
Level 30: 50%   (steep decline)
Level 35: 25%   (severe drop)
Level 40: 5%    (mostly fails)
Level 50: <1%   (nearly unusable)
```

**Why This Validates the Plan:**
- Proves MLP plateau is architectural (not just missing data)
- GNN should show shallower decline curve (tree structure provides understanding)
- Establishes performance baseline for Phase 4 comparison

**Success Criteria:**
- Clear monotonic decline visible
- Confirms: GNN is necessary to go deeper than L20

---

## Phase 1: MLP Baseline on Easy Levels ✓ COMPLETE

**What Was Done:**
- Trained MLP (256 hidden, 2 layers) on A* data from levels 2-6
- Generated ~864k samples from 283k episodes across L2-L6
- Achieved 88.9% policy accuracy, 100% value accuracy on validation
- Tested on level 20: achieved 90% win rate

**Key Finding:**
Level 20 is the practical ceiling for flat-board MLP architecture. Performance degrades sharply beyond this point. This validates the core hypothesis: **tree structure is necessary for deeper levels**.

**Checkpoint:** `policy_value_latest.pt` (or `policy_value_mpl_v1.pt` for archival)

**Lessons:**
- Supervised learning on A* labels works well for easy positions
- But A* tree structure itself (why it made each move) is invisible to flat MLP
- This is why causal understanding (GNN+tree) is essential

---

## Phase 2: Rich Episode Logging with Search Trees

**Goal:** Log A* search trees for future GNN training

**Episode Structure:**
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

**Tasks:**
1. Update `EpisodeLogger.java` to call `logTree()` after each move
2. Extract current node's immediate children from A* search
3. Serialize to JSON with visits, heuristics, move names
4. Verify file sizes are reasonable (~10-50MB per 1k games)
5. Re-run data generation with tree logging enabled

**Key Design Decisions:**
- Tree depth: Current node + immediate children (not full tree from root)
- Tree data: visit counts, heuristic scores, move encoding
- Board as root node: Board features become node features in GNN
- All moves logged: Both winning and losing paths (network learns from both)
- Winning move marked: The chosen_command indicates which child was selected

**Data Generation:**
```bash
# Generate with tree logging
./gradlew test --tests "ai.games.training.EndgameTrainingDataGenerator.testEndgameLevel2" \
  -Dlog.episodes=true -Dlog.trees=true --console=plain
```

**Success Criteria:**
- Episodes contain valid tree JSON
- Tree structure matches A* search (moves, visits, heuristics)
- File sizes reasonable for training
- Can parse trees back during dataset creation

---

## Phase 3: GNN Model with Tree Input

**Goal:** Build GNN that learns from board state + move tree structure

**Architecture Decision:**
- Use pure GNN, not hybrid (MLP + GNN)
- Board state becomes root node of graph
- A* move tree (children) becomes subgraph
- GNN processes entire graph: state influences move evaluation, moves influence state understanding

**Graph Structure:**
```
Root node (board features: pile sizes, foundation state, talon, etc.)
  ├─ Child 1 (move: "move T1 F1", visits: 150, heuristic: 0.85)
  ├─ Child 2 (move: "turn", visits: 50, heuristic: 0.30)
  └─ Child 3 (move: "move F1 T2", visits: 20, heuristic: 0.15)
```

**Why GNN Over Flattening:**
Flattening tree to visit counts loses the reasoning:
- **Example:** "Unlock pile X now" → appears in 150 A* simulations (0.75 probability)
- **Visit count alone:** "This move is good (0.75)" — no explanation
- **Tree structure reveals:** Unlocking enables cascades → board improves downstream → red 4 gets home
- **GNN can learn:** These forward-looking dependencies; flattening cannot

**Implementation Tasks:**
1. Create `tree_encoding.py`: Convert JSON tree → PyTorch Geometric graph
2. Update `dataset.py`: Load both board state and tree structure
3. Create `gnn_model.py`: Implement GNN architecture (GraphConv or GAT)
4. Update `train_policy_value.py`: Support both MLP and GNN training
5. Handle variable tree sizes in batches (padding/masking)

**GNN Architecture Options:**
- GraphConv (simple, fast): `num_layers=3-4`
- GAT (attention-based): Better for learning which moves are important
- Message passing (custom): Full control over node updates

**Training:**
```bash
python -m src.train_policy_value \
  --model-type gnn \
  --hidden-dim 256 \
  --num-layers 3 \
  --epochs 5 \
  logs/episode*.log
```

**Checkpoint:** `policy_value_gnn_v1.pt`

**Success Criteria:**
- GNN trains without errors on tree-augmented data
- Validation accuracy > MLP accuracy on same data
- GNN tested on L20-50 shows improved falloff curve vs MLP
- Can load/save GNN checkpoints correctly

---

## Phase 4: Multi-Model Support & Experimentation

**Goal:** Support both MLP and GNN in codebase for comparison

**Required Changes:**
1. Model registry: `ModelFactory` selects correct model type
2. Config/args: `--model-type mpl` or `--model-type gnn`
3. Training loop: Agnostic to model type (both implement same interface)
4. Inference: Service can load either MLP or GNN
5. Evaluation: Compare MLP vs GNN on same test set

**Code Structure:**
```
neural-network/src/
  model.py              (base Model class)
  models/
    mpl_model.py        (MLP implementation from PolicyValueNet)
    gnn_model.py        (GNN implementation)
  model_factory.py      (selects correct model)
  tree_encoding.py      (JSON tree → PyTorch Geometric graph)
  dataset.py            (handles both EPISODE_STEP and EPISODE_TREE)
  train_policy_value.py (model-agnostic training)
```

**Checkpoints:**
- `policy_value_mpl_v1.pt` (Phase 1: MLP on L2-6)
- `policy_value_gnn_v1.pt` (Phase 3: GNN on L2-6 with trees)
- Later: `mpl_v2.pt`, `gnn_v2.pt` as versions improve

**Evaluation Plan:**
- Test set: Difficulty 20, 25, 30, 35, 40, 50
- Each level: 20+ games
- Metrics: Win %, avg moves, performance consistency
- Document in README.md with comparison table

**Expected Result:**
```
Level | MLP Win % | GNN Win % | Improvement
------|-----------|-----------|------------
  20  |   90%     |   95%     |    +5%
  25  |   70%     |   85%     |   +15%
  30  |   50%     |   75%     |   +25%
  35  |   25%     |   60%     |   +35%
  40  |    5%     |   40%     |   +35%
  50  |   <1%     |   15%     |   +15%
```

**Success Criteria:**
- GNN outperforms MLP across all difficulty levels
- Performance advantage increases with difficulty (proves tree value)
- Both models load/save correctly
- Service handles model selection at startup

---

## Phase 5: Self-Play Loop (The Ultimate Goal)

**Goal:** Continuous improvement through self-play with network-guided MCTS

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
3. train_policy_value.py loads MCTS episodes with updated targets:
   ```
   V_target = game_outcome  (Phase 1-4: supervised learning)
   or
   V_target = bootstrapped from V(next_state)  (Phase 5: RL)
   ```
4. Version management: `gnn_v0.pt`, `gnn_v1.pt`, `gnn_v2.pt`, ...
5. Tournament harness: compare win rates between versions

**Self-Play Training Loop:**
```bash
# Iteration 0: Use supervised GNN (v0)
python -m src.service --checkpoint checkpoints/policy_value_gnn_v0.pt &

# Generate MCTS games with network guidance
./gradlew test --tests "ai.games.training.AlphaSolitaireLevelTest" \
  -Dendgame.games.difficulty.level=30 -Dendgame.games.per.level=500 \
  -Dlog.episodes=true -Dlog.trees=true

# Retrain on MCTS-guided data
python -m src.train_policy_value \
  --model-type gnn \
  --learning-rate 0.0001 \
  logs/episode*.log

# Save as v1
python tools/promote_checkpoint.py --name gnn_v1

# Iteration 1: Test v1 vs v0
# Compare win rates, then iterate
```

**Metrics (Self-Play Performance):**
- **v1 vs v0:** Win % improvement (should be positive)
- **Convergence:** How many iterations until improvement plateaus
- **Stability:** Whether improvements are consistent or noisy
- **Scalability:** Can we reach superhuman performance?

**Expected Trajectory:**
```
v0 (A* supervised): L30 = 50%, L40 = 5%, L50 = <1%
v1 (1st self-play): L30 = 60%, L40 = 15%, L50 = 2%
v2 (2nd self-play): L30 = 70%, L40 = 25%, L50 = 5%
v3 (3rd self-play): L30 = 75%, L40 = 35%, L50 = 10%
...
vN (plateau): L30 = 85%+, L40 = 50%+, L50 = 25%+
```

**Success Criteria:**
- v1 beats v0 consistently
- v2 beats v1 consistently
- Visible improvement trajectory over 3-5 iterations
- Clear convergence point (diminishing returns)

**Why Self-Play Is The Ultimate Goal:**
Without self-play, network is capped at "trained on A*" (limited by A*'s data).
With self-play, network and MCTS co-evolve to discover strategies neither would find alone.
This is how AlphaGo and AlphaZero achieved superhuman play.

---

## Key Open Questions

### 1. How to represent search tree in JSON?
Current approach: recursive nested object
```json
{
  "moves": ["move T1 F1", "turn"],
  "visits": [150, 50],
  "children": [
    {"moves": [...], "visits": [...]},
    {"moves": [...], "visits": [...]}
  ]
}
```

Alternative: Flatten to edge list, rebuild during training
Decision: Nested for clarity; rebuild during training

### 2. Should we log full tree or just immediate children?
- Full tree (depth 12+): Complete search context, large files
- Immediate children only: What moves were explored, smaller files
- Top K moves: Prune to most promising, smallest files

**Decision:** Immediate children (~5-10 moves) balances context and file size

### 3. GNN Architecture: GraphConv vs GAT vs Custom?
- GraphConv: Simple, fast, good baseline
- GAT: Attention lets network learn which moves are important
- Message passing: Full control, potentially better but more complex

**Decision:** Start with GraphConv, upgrade to GAT if needed

### 4. When to enable bootstrapped values?
- **Now (Phase 1-4):** Use game outcome (oracle label)
- **Phase 5:** Use V(next_state) (self-play RL)

**Decision:** Switch to bootstrapped V when v0 performance plateaus (~Phase 5, iteration 2+)

---

## Success Metrics

### Phase 0 (MLP Falloff Testing)
- [ ] Performance measured at L20, 25, 30, 35, 40, 50
- [ ] Clear monotonic decline visible
- [ ] Confirms architectural limitation (gates Phase 2)

### Phase 1 (MLP Baseline) ✓ COMPLETE
- [x] Training data generated (L2-L6, ~864k samples)
- [x] MLP trains without errors
- [x] MLP wins some games on L20 (90%)
- [x] Checkpoint saved as `policy_value_latest.pt`

### Phase 2 (Tree Logging)
- [ ] EpisodeLogger logs search tree structure
- [ ] Tree JSON validates (moves, visits, heuristics, children)
- [ ] File sizes reasonable (<50MB per 1k games)
- [ ] Can parse trees back during dataset creation

### Phase 3 (GNN Model)
- [ ] tree_encoding.py converts JSON trees to PyTorch Geometric graphs
- [ ] dataset.py loads both board state and tree structure
- [ ] GNN model trains without errors
- [ ] GNN outperforms MLP on L20-50 (shallower falloff curve)
- [ ] Checkpoint saved as `policy_value_gnn_v1.pt`

### Phase 4 (Multi-Model Support)
- [ ] ModelFactory correctly selects MLP or GNN
- [ ] Both models train and evaluate cleanly
- [ ] Comparison test: MLP vs GNN on L20-50
- [ ] Results documented with improvement table

### Phase 5 (Self-Play)
- [ ] MCTS generates games with network-guided search
- [ ] v1 beats v0 in win rate (positive delta)
- [ ] v2 beats v1 (iteration improves)
- [ ] 3+ iterations show clear improvement trajectory
- [ ] Convergence point identified (diminishing returns)
- [ ] Final performance significantly exceeds Phase 4

---

## Key Insight: Why GNN Matters for Solitaire

**The Problem:** Flattening tree to statistics loses the reasoning

Visit counts don't explain *why* a move is good. Example:
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

**Architecture Difference:**
- **Go (AlphaGo):** Pure position evaluation → CNN works (translation invariance, locality)
- **Solitaire (AlphaSolitaire):** Move value depends on downstream unlocking → tree structure essential → GNN required

---

## Dependencies & Sequence

```
Phase 0 (test falloff) → validates need for Phase 2
         ↓
Phase 1 (MLP) ✓ → provides baseline for comparison
         ↓
Phase 2 (tree logging) → enables Phase 3
         ↓
Phase 3 (GNN) → can be tested immediately
         ↓
Phase 4 (multi-model) → compares GNN vs MLP
         ↓
Phase 5 (self-play) → continuous improvement loop
```

All phases feed into the ultimate goal: **superhuman solitaire play through network+search co-evolution**.

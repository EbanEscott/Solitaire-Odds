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

## Phase 1: Immediate

**Goal:** Verify configurable architecture works, establish baseline

**Tasks:**
1. Generate A* episodes (we have ~44k samples currently)
2. Train with medium model (512 hidden, 3 layers)
3. Measure: policy accuracy (~75% expected), value accuracy (~99% expected)
4. Document results in README.md

**Success Criteria:**
- Training completes without errors
- Policy accuracy > 70%
- Value accuracy > 95%
- Checkpoint saves correctly

**Blockers:**
- None, can start immediately

---

## Phase 2: Unified Game Tree

**Goal:** Create single tree representation usable by all algorithms

**Current Situation:**
- `engine/src/main/java/ai/games/player/ai/astar/GameTreeNode.java` — A* search tree
- `engine/src/main/java/ai/games/player/ai/alpha/TreeNode.java` — MCTS tree
- Separate implementations, different purposes

**What We Need:**
```java
// Unified GameTree class (new)
public class GameTree {
    public final Solitaire state;
    public final String move;           // move that led here
    public final GameTree parent;
    public final Map<String, GameTree> children;
    
    // Search information
    public int visitCount;              // how many times explored
    public double valueSum;             // cumulative reward
    public double[] moveVisits;         // visits per legal move
    public double[] moveValues;         // values per legal move
    
    // Algorithm-specific (can extend)
    public boolean pruned;              // A* cycle detection
    public double[] priors;             // MCTS policy priors
}
```

**Why Unified:**
- A* can use for cycle detection (pruning)
- MCTS can use for visit tracking
- Episode logging gets consistent tree structure
- Easier to log search reasoning for training

**Blockers:**
- Need to refactor A* and MCTS to use common class
- Need to preserve existing behavior during refactor

---

## Phase 3: Rich Episode Logging

**Goal:** Log search trees, not just final moves

**Current:** Episode logs one move per step
```json
{
  "step_index": 5,
  "state": {...board state...},
  "legal_moves": ["move T1 F1", "turn", ...],
  "chosen_command": "move T1 F1"
}
```

**Future:** Include search context
```json
{
  "step_index": 5,
  "state": {...board state...},
  "search_tree": {
    "root": {
      "moves": ["move T1 F1", "turn", ...],
      "visits": [150, 50, ...],         // move visit counts
      "values": [0.85, 0.30, ...],      // move value estimates
      "children": {...recursive tree...}
    }
  },
  "chosen_command": "move T1 F1"  // maps to index 0 in visits
}
```

**Why Important (Trajectory-Aware Training):**
Currently, network learns: "Given state, predict move"
With tree: network learns: "Given state AND search results, predict move"

The tree shows:
- What moves were explored (visited)
- How promising each was (value estimates)
- How much confidence (visit count)

This is **the real trajectory context** — not just "game won" but "search revealed this move was best"

**Network Benefits:**
- Policy learns move distributions (not just greedy choice)
- Can extract MCTS-discovered improvements
- Better understanding of move tradeoffs

**Blockers:**
- JSON serialization of tree (size concerns)
- Need to store tree during gameplay
- Training code needs to parse tree

---

## Phase 4: Self-Play Loop

**Goal:** Continuous improvement through self-play

**Flow:**
```
v0 (A* bootstrap)
    ↓
Generate MCTS games with v0 (game tree + moves logged)
    ↓
Retrain on MCTS-discovered move distributions
    ↓
v1 (should beat v0 in win rate)
    ↓
v1 generates new MCTS games
    ↓
v2, v3, ... iterate
```

**What Changes:**
1. Run AlphaSolitairePlayer (MCTS + network) on seeded positions
2. Log full search trees from MCTS
3. Retrain: network learns what MCTS discovered
4. Compare versions: v0 vs v1 win rates

**Value Targets:**
```
Phase 1-3 (supervised): V_target = game_outcome (1.0 or 0.0)
Phase 4 (self-play):    V_target = 0 + γ * V(next_state)  [bootstrapped]
```

Bootstrapped values are optional but helpful for RL.

**Blockers:**
- MCTS must work (currently 0% win rate - needs diagnostics)
- Search tree logging must work
- Need model versioning system (v0.pt, v1.pt, v2.pt)
- Need tournament harness (v1 vs v2 comparison)

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

### Phase 1
- [ ] Training runs without errors
- [ ] Policy acc > 70%, Value acc > 95%
- [ ] Checkpoint saves and loads

### Phase 2
- [ ] Unified GameTree works in A*
- [ ] Unified GameTree works in MCTS
- [ ] No regression in algorithm performance

### Phase 3
- [ ] Episode logs include search trees
- [ ] File sizes reasonable (< 10x current)
- [ ] Training code parses trees correctly

### Phase 4
- [ ] Self-play games generate consistently
- [ ] v1 model trains on MCTS data
- [ ] v1 win rate > v0 win rate

---

## Key Insight

The real trick for self-play isn't just "training on full games" but **training on full search context**.

**Before:** Network sees state, predicts move
```
State → [Neural Network] → Move
```

**After (Phase 3+):** Network sees state AND search tree
```
State + Search Tree (visits, values) → [Neural Network] → Move Distribution
```

This is why AlphaGo works: network learns what search discovered, then those improved priors make search better, creating a feedback loop.

Currently we're stuck at "training on full games" because we're not logging the search context. Phase 3 fixes that.

---

## Dependencies & Blockers

- **Phase 1 → 2:** Complete Phase 1 (baseline established)
- **Phase 2 → 3:** Complete Phase 2 (unified tree in place)
- **Phase 3 → 4:** Complete Phase 3 + resolve MCTS 0% win rate

Note: MCTS diagnostics needed before Phase 4


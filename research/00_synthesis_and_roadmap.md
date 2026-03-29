# Synthesis: Neural Network Architecture for Infinite Hexagonal Tic-Tac-Toe

**Cross-Cutting Summary and Project Roadmap**

*This document synthesizes findings from the five research documents in this series into a unified architectural specification and implementation plan.*

---

## 1. Game Definition

| Property | Value |
|----------|-------|
| Board | Infinite hexagonal grid (axial coordinates) |
| Players | 2 (deterministic, perfect information, zero-sum) |
| Moves per turn | 2 stones (except first player's first turn: 1 stone) |
| Win condition | 6 consecutive stones along any hex axis (3 axes) |
| Draw condition | Not clearly defined for infinite boards — likely none under optimal play |

This game is closely related to **Connect6** (Wu & Huang, 2006), which uses the same 2-stones-per-turn mechanic on a square grid with 6-in-a-row. The hexagonal grid adds a third axis (6 neighbors instead of 8 in square grids, but 3 line directions instead of 4), and the infinite board removes boundary effects entirely.

---

## 2. Key Findings Across Research Areas

### 2.1 From AlphaGo/AlphaZero (Doc 01)

The core AlphaZero paradigm — **dual-headed ResNet + MCTS self-play** — transfers directly. The critical architectural elements:

- **Dual-headed network** with shared ResNet backbone, policy head (move probabilities), and value head (win probability). This is the proven foundation — no reason to deviate.
- **PUCT-based MCTS** with neural network priors guides search efficiently. The policy head eliminates the need for rollouts (shown superior in Silver et al., 2017a ablation).
- **Self-play training loop**: self-play → collect (s, π, z) → train → evaluate → replace. The loss is `(z-v)² - π⊤ log p + c‖θ‖²`.
- AlphaGo Zero used 40 residual blocks / 256 filters for 19x19 Go. Our game's lower strategic complexity (no captures, no ko, no territory scoring) means a **smaller network suffices**.

### 2.2 From Post-AlphaZero Developments (Doc 02)

Five innovations are directly applicable:

1. **Gumbel MuZero search** (Danihelka et al., 2022) — sequential halving with Gumbel-Top-k. Achieved 8x simulation efficiency in 9x9 Go. Critical for our high branching factor.
2. **KataGo auxiliary targets** (Wu, 2019) — ownership prediction, score margin, threat detection. Dense spatial gradients on a large, sparse board.
3. **Playout cap randomization** (KataGo) — randomize MCTS budget per move (25% full / 75% reduced). 50x training efficiency improvement.
4. **Reanalysis** (MuZero Reanalyze / EfficientZero) — re-search past games with updated network for fresher training targets. Free sample efficiency.
5. **Hexagonal convolutions from Polygames** (Cazenave et al., 2020) — hex-aware kernels respecting 6-neighbor topology.

**Not applicable**: MuZero's learned dynamics model (our rules are known and trivial), Stochastic MuZero (deterministic game), Student of Games (perfect information only).

### 2.3 From Hex Grid & Spatial Architectures (Doc 03)

- **Axial coordinates with brick-wall storage** is the optimal representation. Standard 3x3 convolutions on the brick-wall layout capture 6 of 8 kernel positions as true hex neighbors; the network learns to suppress the 2 non-adjacent positions.
- **19x19 dynamic window** centered on stone centroid handles the infinite board. A CNN on this fixed-size window is ~0.1ms per forward pass — enabling deep MCTS.
- **D₆ 12-fold data augmentation** (6 rotations × 2 reflections) is the single most impactful technique for data efficiency on hex boards. In axial coords, 60° rotation maps (q,r) → (-r, q+r).
- **12 input feature planes** recommended, including tactical hint planes (threat-3, threat-2) that accelerate early training.
- **Sequential two-move decomposition**: the network always predicts a single move. An `IS_FIRST_MOVE_OF_TURN` input plane distinguishes first vs. second placement.

### 2.4 From MCTS Adaptations (Doc 04)

- **Interleaved sub-move tree** (Approach 3) recommended: Player A move 1 → Player A move 2 → Player B move 1 → Player B move 2. Branching factor is O(N) per edge, not O(N²).
- **Zone of Interest (ZoI) with margin 3**: only cells within 3 hexes of any existing stone are legal MCTS expansions. For 6-in-a-row, margin 3 is the minimum to guarantee all winning-relevant moves are included.
- **Lazy Zobrist hashing** on infinite coordinates with canonical move-pair ordering for transposition detection. XOR-commutative hashing means the board hash after both sub-moves is order-independent.
- **400-800 simulations/move** for self-play, yielding ~2,400 games/hour on a single GPU.
- **Backpropagation rule**: negate value only at player transitions (sub-move 2 → opponent's sub-move 1), not within a player's two sub-moves.

### 2.5 From Training Pipeline & Game Theory (Doc 05)

- **Strategy-stealing argument** proves the game cannot be a second-player win. The 1-stone first move is a balancing mechanism (analogous to Connect6). The game is likely a first-player win or draw.
- **Game complexity**: state-space ~10^100-140 (between chess and Go). Raw branching factor ~200K-250K per 2-stone turn, reduced to ~4,950 with policy guidance, and ~225 with aggressive pruning (k₁=k₂=15).
- **3-phase training pipeline**: (1) prototype on 8-block/128-filter net, 19x19 board, 200 sims, ~50-100 GPU-hours; (2) scale to 15-block/192-filter, 31x31 multi-scale, 600 sims, async self-play; (3) refine to 20-block/256-filter, architecture search, robustness testing.
- **Curriculum learning**: start with 4-in-a-row on 11x11, scale to 6-in-a-row on 31x31. Compare against direct training.
- **Risk matrix**: 9 identified risks. Highest: MCTS slowdown from branching factor (mitigate with C++/Rust MCTS + decomposed moves), opening mode collapse (mitigate with Dirichlet noise + playout cap randomization).

---

## 3. Unified Architecture Specification

### 3.1 Network Architecture

```
Input:  12 × 19×19  (Phase 1)  →  12 × 31×31  (Phase 2+)
        Axial coordinates, brick-wall layout, dynamic windowing

Input Feature Planes (12):
  1. MY_STONES          — binary, current player's stones
  2. OPP_STONES         — binary, opponent's stones
  3. MOVES_THIS_TURN    — binary, stone placed as first move of current turn
  4. IS_FIRST_MOVE      — uniform 1.0 if first move of turn, 0.0 if second
  5. COLOR_TO_PLAY      — uniform 1.0 for Player 1, 0.0 for Player 2
  6. MOVE_RECENCY       — float, per-cell decay (most recent = 1.0, ×0.9/move)
  7. Q_RELATIVE         — float, normalized axial q-coord relative to window center
  8. R_RELATIVE         — float, normalized axial r-coord relative to window center
  9. DISTANCE_TO_CENTROID — float, normalized hex distance to stone centroid
 10. MY_THREAT_5        — binary, cells completing a friendly open-5 (immediate win)
 11. OPP_THREAT_5       — binary, cells completing opponent open-5 (must-block)
 12. MY_THREAT_4        — binary, cells extending a friendly open-4

Backbone: ResNet
  ├── Initial conv: 3×3, 12 → C channels, BN, ReLU
  ├── B Residual blocks: [3×3 conv, BN, ReLU, 3×3 conv, BN] + skip, ReLU
  ├── Global pooling branch (KataGo-style): GAP + GMP → FC → broadcast
  └── Phase 1: B=8, C=128  |  Phase 2: B=15, C=192  |  Phase 3: B=20, C=256

Policy Head:
  ├── 1×1 conv, C → 2 channels, BN, ReLU
  ├── Flatten → FC → H×W logits
  └── Softmax masked to empty cells within ZoI

Value Head:
  ├── 1×1 conv, C → 1 channel, BN, ReLU
  ├── Flatten → FC(H×W, 256) → ReLU → FC(256, 1)
  └── Tanh output ∈ [-1, +1]

Auxiliary Heads (KataGo-style):
  ├── Ownership: 1×1 conv → H×W×3 (my/opp/empty probability per cell)
  ├── Threat count: FC from global pool → scalar per player
  └── Combined auxiliary loss weight: 0.20 (Phase 1) → 0.30 (Phase 2+)
```

### 3.2 MCTS Configuration

```
Search Algorithm: Gumbel MuZero-style with sequential halving
  ├── Gumbel-Top-k at root for action selection
  ├── PUCT for internal tree traversal
  └── Known-rules forward model (not learned dynamics)

Tree Structure: Interleaved sub-moves
  ├── Level pattern: P1-move1 → P1-move2 → P2-move1 → P2-move2 → ...
  ├── First turn exception: P1-move1 → P2-move1 → P2-move2 → ...
  ├── Value negation: only at player transitions
  └── Transposition DAG via Zobrist hashing

Action Space: Zone of Interest
  ├── ZoI margin: 3 hexes from any existing stone
  ├── Policy-guided pruning: top-k₁=15 for sub-move 1, top-k₂=15 for sub-move 2
  ├── Progressive widening fallback: k_expand(N) = min(|ZoI|, ⌊8·N^0.4⌋)
  └── First move: canonically (0,0) — skip search

Simulation Budget:
  ├── Phase 1 training: 200 sims/move
  ├── Phase 2 training: 600 full / 75 reduced (playout cap randomization 25/75)
  ├── Evaluation: 800-1600 sims/move
  └── Analysis: 1600-6400 sims/move

Exploration:
  ├── Dirichlet noise: α = 10/|ZoI| (~0.08-0.10), ε = 0.25
  ├── Temperature: τ=1.0 for first 20 half-moves, τ=0.5 for 21-40, τ=0.1 after
  ├── cPUCT: 2.5 (Phase 1) → 2.0 (Phase 2) → 1.5-2.0 (Phase 3)
  └── FPU reduction: 0.0 (Phase 1) → -0.2 (Phase 2+)

Parallelization:
  ├── 4 CPU threads for tree traversal
  ├── Batch size 32-64 for GPU evaluation
  ├── Virtual loss: n_vl = 3
  └── Expected: ~30K sims/sec → ~25ms/move → ~2,400 games/hour (1 GPU)
```

### 3.3 Training Pipeline

```
Self-Play Loop (AlphaZero-style):
  1. Generate self-play games with current best network + MCTS
  2. Store (state, MCTS_policy, game_outcome) in replay buffer
  3. Sample minibatches, train network on combined loss
  4. Periodically evaluate new checkpoint vs. current best (55% win threshold)
  5. Replace best network if threshold met
  6. Reanalyze: re-search buffered games with latest network (every N iterations)

Loss Function:
  L = (z - v)²                           [value: MSE with game outcome]
    - π⊤ log p                            [policy: cross-entropy with MCTS visits]
    + c‖θ‖²                               [L2 regularization, c=1e-4]
    + λ_own · L_ownership                  [auxiliary: ownership prediction]
    + λ_thr · L_threats                    [auxiliary: threat count prediction]

Replay Buffer:
  ├── Phase 1: 5,000 games (~1M positions)
  ├── Phase 2: 25,000 games (~5M positions)
  └── Sampling: 75% recency-weighted / 25% uniform

Data Augmentation:
  └── D₆ symmetry: 12-fold (6 rotations × 2 reflections)
      Applied to board state, policy target, and auxiliary targets

Optimizer: SGD with momentum 0.9
  ├── Phase 1: LR 0.02 → 0.0002 (cosine)
  ├── Phase 2: LR 0.01 → 0.001 → 0.0001 (step decay)
  └── Weight decay: 1e-4 throughout
```

---

## 4. Implementation Roadmap

### Phase 1: Prototype (Weeks 1-3, ~50-100 GPU-hours on 1× RTX 4090)

**Goal**: End-to-end pipeline validation.

| Week | Milestone |
|------|-----------|
| 1 | Game engine (hex grid, axial coords, win detection). MCTS skeleton with 2-move turns. NN forward pass works. |
| 2 | Self-play generates games. Training loop reduces loss. Network beats random >99%. |
| 3 | Network beats greedy heuristic >80%. ~500-800 Elo. Visualize learned policy/value maps. |

**Key experiments**: (a) verify decomposed 2-stone MCTS correctness, (b) with/without auxiliary targets, (c) curriculum (4-in-a-row → 6-in-a-row) vs. direct training.

**Tech stack**: PyTorch for network, Python for prototype MCTS (rewrite in C++/Rust before Phase 2).

### Phase 2: Full-Scale Training (Weeks 4-10, 2× A100 or ~12 weeks on 1× RTX 4090)

**Goal**: Strong player on full 6-in-a-row game.

| Week | Milestone |
|------|-----------|
| 4-5 | Async pipeline. MCTS in C++/Rust. Throughput targets validated. |
| 6 | ~1500 Elo. Beats 1-ply heuristic >95%. |
| 8 | ~2000 Elo. Diverse openings. Value head >70% accuracy from mid-game. |
| 10 | ~2500 Elo. Complex threat sequences. |

**Key changes from Phase 1**: 15-block/192-filter network, 31×31 multi-scale training, playout cap randomization, Gumbel search, reanalysis, 600 sims/move.

### Phase 3: Refinement (Weeks 11-16)

**Goal**: Push performance, analyze learned strategies, robustness.

- Scale to 20-block/256-filter. Architecture search (SE-ResNet, bottleneck blocks, small vision transformers).
- Adversarial testing (distant-play opponents, unseen board sizes).
- Strategy analysis: extract opening book, identify novel tactical patterns.
- Target: ~2800-3000 Elo.

---

## 5. Critical Design Decisions

### 5.1 Why AlphaZero over MuZero?

The game has known, trivial rules (place stone → check 6-in-a-row). MuZero's learned dynamics model would introduce approximation error with zero benefit. AlphaZero's exact forward model is strictly preferable for deterministic, fully-observable, known-rules games (Doc 02, §8.1).

### 5.2 Why Sequential Sub-Moves over Composite Moves?

Composite moves (treating (a,b) as a single action) create O(N²) branching factor per turn — ~200K+ actions in mid-game. Sequential decomposition reduces this to O(N) per tree level (~100-150 actions), pruned to ~15 with policy guidance. The network is simpler (single-move output), and transposition detection via Zobrist hashing automatically deduplicates (a,b) vs (b,a) orderings (Doc 04, §7.1).

### 5.3 Why Dynamic Windowing over GNNs/Transformers?

A 19×19 or 31×31 CNN window gives ~0.1ms inference — enabling 30K sims/sec and 2,400 games/hour. GNNs are ~10-50x slower per forward pass. Transformers have O(n²) attention cost. The game's locality (6-in-a-row threats are within radius 6) means the window captures all strategically relevant information. If games ever sprawl beyond 31×31 (unlikely before termination), the window can grow or shift (Doc 03, §Recommended Architecture).

### 5.4 Why Gumbel Search?

Standard PUCT spreads simulations uniformly across all root children. With 100+ legal moves and 800 simulations, that's ~8 sims per move — insufficient for reliable value estimates. Gumbel-Top-k with sequential halving concentrates simulations on ~5-10 final candidates with ~80-160 sims each. This is the single most impactful algorithmic choice from post-AlphaZero literature for high-branching-factor games (Doc 02, §8.3).

### 5.5 Why KataGo-Style Auxiliary Targets?

On a large, sparse hex board, most positions early in training have value ≈ 0 (uncertain). The binary win/loss signal is sparse and noisy. Auxiliary targets — especially **ownership prediction** (per-cell probability of being "controlled" by each player at game end) — provide dense spatial gradients that teach the network about influence, territory, and threats even when the game outcome is uncertain. KataGo achieved 50x compute efficiency over prior Go AIs partly through these auxiliary signals (Doc 02, §5; Doc 05, §3).

---

## 6. Game-Theoretic Expectations

Based on analysis in Doc 05 and comparison to Connect6:

- **First-player advantage** is mitigated by the 1-stone first move rule (identical mechanism to Connect6). Connect6 was specifically designed so the 2/6 stone-to-win ratio creates near-perfect balance (Wu & Huang, 2006).
- **Strategy-stealing** proves the game cannot be a second-player win.
- The hexagonal grid's 3-axis geometry (vs. 4-axis on square grid) may create tighter tactical patterns — fewer "quiet" directions means threats are harder to ignore.
- **Expected game length**: 40-70 turns (80-140 half-moves), based on Connect6 analogies adjusted for hex geometry.
- The game is almost certainly **not a draw** on an infinite board — the combination of unlimited space and 2-stones-per-turn should allow the first player to eventually build unstoppable threats.

---

## 7. Software Architecture Overview

```
hex-ttt-nn/
├── research/                    # This research series
├── game/                        # Game engine
│   ├── hex_grid.py              # Axial coordinate system, neighbor computation
│   ├── board.py                 # Board state, win detection (line scan along 3 axes)
│   ├── rules.py                 # Move validation, turn management (1st vs 2nd sub-move)
│   └── zobrist.py               # Lazy Zobrist hashing for infinite coordinates
├── nn/                          # Neural network
│   ├── model.py                 # Dual-headed ResNet with auxiliary heads
│   ├── features.py              # 12-plane input feature extraction
│   ├── hex_conv.py              # Brick-wall layout convolution utilities
│   └── symmetry.py              # D₆ augmentation transforms
├── mcts/                        # Monte Carlo Tree Search (rewrite in C++/Rust for Phase 2)
│   ├── search.py                # PUCT + Gumbel search
│   ├── node.py                  # Tree/DAG nodes with transposition support
│   ├── zoi.py                   # Zone of Interest computation
│   └── parallel.py              # Batched leaf evaluation, virtual loss
├── training/                    # Self-play training pipeline
│   ├── self_play.py             # Game generation with MCTS
│   ├── trainer.py               # Network training (loss, optimizer, scheduling)
│   ├── replay_buffer.py         # Replay buffer with recency weighting
│   ├── evaluator.py             # Checkpoint evaluation (Elo tracking)
│   └── reanalyze.py             # Re-search past games with updated network
├── analysis/                    # Analysis and visualization
│   ├── visualize.py             # Board state + policy/value heatmap rendering
│   └── opening_book.py          # Extract opening patterns from self-play
└── configs/                     # Hyperparameter configurations
    ├── phase1.yaml
    ├── phase2.yaml
    └── phase3.yaml
```

---

## 8. Hardware Recommendations

| Configuration | Phase 1 | Phase 2 | Phase 3 | Total Time | Estimated Cost |
|---|---|---|---|---|---|
| 1× RTX 4090 (owned) | 3 weeks | 12 weeks | 6 weeks | ~21 weeks | ~$100 (electricity) |
| 2× A100 (cloud) | 2 weeks | 6 weeks | 4 weeks | ~12 weeks | ~$12,000 |
| 4× A100 (cloud) | 1.5 weeks | 4 weeks | 3 weeks | ~8.5 weeks | ~$17,000 |

**Recommendation**: Start Phase 1 on a local GPU. Decide on cloud scaling for Phase 2 based on Phase 1 results and timeline requirements.

---

## 9. Key Risks (Prioritized)

| # | Risk | Mitigation |
|---|------|------------|
| 1 | MCTS too slow due to branching factor | Decomposed sub-moves + Gumbel search + C++/Rust MCTS + progressive widening |
| 2 | Opening mode collapse | Dirichlet noise + temperature + playout cap randomization + forced diversity |
| 3 | Value head fails to learn | Auxiliary targets (ownership, threats) + curriculum from simpler variants |
| 4 | Infinite board edge artifacts | Dynamic re-centering + 31×31 window + relative coordinate features |
| 5 | Training instability | Frequent checkpointing + conservative LR + gating (55% threshold) |
| 6 | Game is "too drawn" at high skill | Verify with long self-play; consider swap rule; shaped auxiliary rewards |

---

## 10. Open Questions for Early Experimentation

1. **Optimal ZoI margin**: Is margin=3 sufficient for 6-in-a-row, or does long-range strategic play require margin=4-5? Empirical comparison needed.
2. **Curriculum benefit**: Does starting from 4-in-a-row actually accelerate training for 6-in-a-row, or does the different tactical character (shorter threats) teach the wrong patterns?
3. **Equivariant convolutions**: At what point (if ever) does upgrading from standard 3×3 convolutions to p6-equivariant convolutions pay off? Expected: 2-5x data efficiency, but implementation complexity.
4. **Two policy heads vs. one**: Does having separate heads for first and second sub-moves improve MCTS quality? Or does a single head with the `IS_FIRST_MOVE` feature plane suffice?
5. **Game-theoretic outcome**: Is the game a first-player win under optimal play? Self-play at high strength should reveal this.

---

## References (Cross-Document)

- Antonoglou, I., et al. (2022). Planning in stochastic environments with a learned model. *ICLR 2022*.
- Cazenave, T., et al. (2020). Polygames: Improved zero learning. *ICGA Journal*, 42(4), 244-256.
- Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML 2016*.
- Danihelka, I., et al. (2022). Policy improvement by planning with Gumbel. *ICLR 2022*.
- Hoogeboom, E., et al. (2018). HexaConv. *ICLR 2018*.
- Hubert, T., et al. (2021). Learning and planning in complex action spaces. *ICML 2021*.
- Schrittwieser, J., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature*, 588, 604-609.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529, 484-489.
- Silver, D., et al. (2017a). Mastering the game of Go without human knowledge. *Nature*, 550, 354-359.
- Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go. *Science*, 362, 1140-1144.
- Wu, D. J. (2019). Accelerating self-play learning in Go. *arXiv:1902.10565*.
- Wu, I.-C., & Huang, D.-Y. (2006). A new family of k-in-a-row games. *ICGA Journal*, 29(1), 26-34.
- Ye, W., et al. (2021). Mastering Atari games with limited data. *NeurIPS 2021*.

---

*See individual research documents for comprehensive treatment of each topic:*
- *01: AlphaGo/AlphaZero Architecture*
- *02: Post-AlphaZero Developments (2019-2025)*
- *03: Hex Grids and Spatial Neural Architectures*
- *04: MCTS Adaptations for Multi-Move and Hex Games*
- *05: Training Pipeline Design and Game Theory*

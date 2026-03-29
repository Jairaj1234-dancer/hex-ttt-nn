# CLAUDE.md — Project Guide for Hex-TTT-NN

## Project Overview

AlphaZero-style self-play reinforcement learning system for **Infinite Hexagonal Tic-Tac-Toe** — a novel two-player game on an unbounded hexagonal grid where 6-in-a-row wins and players place 2 stones per turn (first player's first turn: 1 stone only).

**Repository**: `hex-ttt-nn`
**Language**: Python 3.10+
**Framework**: PyTorch
**Architecture**: Dual-headed ResNet + MCTS with KataGo-inspired auxiliary heads

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (Phase 1 prototype)
python train.py --config configs/phase1.yaml

# Train (resume from checkpoint)
python train.py --config configs/phase1.yaml --checkpoint checkpoints/latest.pt

# Play against AI
python play.py --checkpoint checkpoints/best.pt --mode human-vs-ai

# Watch AI vs AI
python play.py --checkpoint checkpoints/best.pt --mode ai-vs-ai

# Syntax check all files
for f in $(find . -name "*.py"); do python3 -m py_compile "$f"; done
```

---

## Module Architecture

```
hex-ttt-nn/
├── game/              # Game engine (zero external dependencies beyond stdlib)
│   ├── hex_grid.py    # HexCoord, axial math, D6 symmetry transforms, brick-wall layout
│   ├── board.py       # Board state (sparse dict), win detection (3-axis scan), Zobrist hash
│   ├── rules.py       # GameState: turn logic, 2-move mechanic, legal moves (ZoI)
│   └── zobrist.py     # Lazy infinite-grid Zobrist hashing (blake2b, 64-bit)
│
├── nn/                # Neural network (PyTorch)
│   ├── hex_conv.py    # HexResBlock, HexResNet (ResNet backbone + KataGo global pooling)
│   ├── features.py    # 12-plane feature extraction from GameState → tensor
│   ├── symmetry.py    # D6 augmentation (12-fold: 6 rotations × 2 reflections)
│   └── model.py       # HexTTTNet: policy + value + ownership + threat heads
│
├── mcts/              # Monte Carlo Tree Search
│   ├── zoi.py         # Zone of Interest computation (margin-based bounding box)
│   ├── node.py        # MCTSNode: PUCT, backup (negates only at player transitions)
│   ├── search.py      # MCTS: full search loop with NN evaluation
│   └── parallel.py    # BatchedEvaluator: GPU batch inference from multiple threads
│
├── training/          # Self-play training pipeline
│   ├── replay_buffer.py  # Position storage, recency-weighted sampling, D6 on-the-fly
│   ├── self_play.py      # Game generation with MCTS, playout cap randomization
│   ├── trainer.py         # SGD optimizer, 3 LR schedules, checkpointing
│   ├── evaluator.py       # Candidate vs best network match play (early termination)
│   └── reanalyze.py       # Re-search old positions with updated network
│
├── analysis/          # Visualization and analysis
│   ├── visualize.py   # Hex board rendering, training curves, Elo plots (matplotlib)
│   └── opening_book.py # Opening pattern extraction from game logs
│
├── configs/           # Training hyperparameters (YAML)
│   ├── phase1.yaml    # Prototype: 8 blocks, 128ch, 19×19, 200 sims
│   ├── phase2.yaml    # Full-scale: 15 blocks, 192ch, 31×31, 600 sims
│   └── phase3.yaml    # Refinement: 20 blocks, 256ch, 800 sims, Gumbel
│
├── research/          # 6 graduate-level research documents (~30K words)
├── train.py           # Main training entry point
├── play.py            # Interactive play entry point
└── CLAUDE.md          # This file
```

---

## Key Design Decisions

### Game Engine
- **Axial coordinates** `(q, r)` for hex grid — uniform 6-neighbor offsets, clean distance formula
- **Brick-wall layout** for NN input — standard 3×3 convolutions work (6/8 kernel positions are true hex neighbors)
- **Board.place()** returns new Board (immutable pattern) with incremental Zobrist hash update
- **GameState.apply_move()** handles sub-move counting internally — callers don't need to track turn state
- **legal_moves()** uses Zone of Interest (bounding box + margin) — not all-empty-cells

### Neural Network
- **12 input feature planes**: stones, turn state, recency, relative coordinates, threat hints
- **Threat planes** (9-11) are computed by linear scan along 3 hex axes — accelerates early training
- **HexResNet** includes KataGo-style global pooling branch (GAP + GMP → FC → broadcast + add)
- **Policy head**: flat logits over H×W grid, masked to valid moves before softmax
- **Auxiliary heads**: ownership (3-class per cell), threats (2 scalars) — dense gradient signal on sparse boards
- **Loss**: cross-entropy for policy (soft MCTS targets), MSE for value, cross-entropy for ownership, MSE for threats. L2 via optimizer weight_decay.

### MCTS
- **Interleaved sub-move tree**: P1-move1 → P1-move2 → P2-move1 → P2-move2
- **Backup negation ONLY at player transitions** — within same-player sub-moves, value sign is preserved
- **First turn exception**: `moves_remaining=1` for turn 1 — handled by GameState, transparent to MCTS
- **ZoI margin=3**: minimum to guarantee all 6-in-a-row-relevant moves are included
- **Dirichlet noise** at root only, epsilon configurable (0 for evaluation)
- **Virtual loss** for parallel MCTS thread safety

### Training Pipeline
- **Self-play → buffer → train → evaluate → replace** (standard AlphaZero loop)
- **Playout cap randomization**: randomly select full vs reduced MCTS sims per move (KataGo)
- **Recency-weighted sampling**: 75% from recent half of buffer, 25% uniform
- **D6 augmentation on-the-fly**: random symmetry per sample during batch collation
- **Gating**: candidate must win >55% of evaluation games to replace best network
- **Reanalysis**: re-search old buffer positions with latest network, blend value targets

---

## Critical Implementation Details

### Coordinate System
- `axial_to_brick(q, r, center_q, center_r, grid_size)` maps hex → grid position
- Convention: even-r offset (row 0 is not shifted)
- `brick_to_axial` is the inverse — used in MCTS to map flat policy indices back to HexCoord
- Window is centered on `stone_centroid()` rounded to nearest integer
- Coordinate planes (Q_RELATIVE, R_RELATIVE) are normalized to [-1, 1]

### Config Structure
All configs follow this schema:
```yaml
game:      {win_length, first_turn_stones, normal_turn_stones}
network:   {grid_size, num_blocks, channels, in_channels}
mcts:      {num_simulations, cpuct, dirichlet_alpha, dirichlet_epsilon, ...}
training:  {learning_rate, lr_schedule, batch_size, replay_buffer_size, ...}
evaluation: {games, win_threshold, checkpoint_interval}
playout_cap: {enabled, full_ratio, reduced_ratio}
reanalysis:  {enabled, interval, batch_size, value_blend_weight}
```

### Common Pitfalls
- `MCTS.search()` re-reads `config["num_simulations"]` each call — playout cap works by mutating the config dict before calling `get_move()`
- `ReplayBuffer.add_game()` expects a list of dicts with keys: `features`, `policy`, `value`, and optional `center`, `game_state`
- `HexTTTNet.forward()` returns a dict with keys: `policy_logits`, `policy`, `value`, `ownership`, `ownership_logits`, `threats`
- `extract_features()` returns `(tensor, (center_q, center_r))` — don't forget the center tuple
- The model expects `valid_moves_mask` as `(B, H*W)` binary tensor — 1 for legal, 0 for illegal

---

## Development Workflow

### Adding a New Feature
1. Write code in the appropriate module
2. Verify imports: `python3 -m py_compile <file>`
3. Test the cross-module integration path manually
4. Update config YAMLs if new hyperparameters are introduced

### Testing Strategy
- **No test framework yet** — validate by importing and running key paths:
  ```python
  # Quick smoke test
  python3 -c "from game import *; gs = GameState(); gs2 = gs.apply_move(HexCoord(0,0)); print(gs2)"
  python3 -c "from nn.model import HexTTTNet; import torch; m = HexTTTNet(19,2,32); print(m(torch.randn(1,12,19,19)).keys())"
  ```
- Full pipeline test: self-play → buffer → train step → checkpoint save/load (verified working)

---

## Roadmap

### Phase 1: Prototype (Target: validate pipeline end-to-end)
- [x] Game engine: hex grid, board, rules, Zobrist hashing
- [x] Neural network: ResNet backbone, dual heads, auxiliary heads
- [x] Feature extraction: 12-plane encoding with threat detection
- [x] D6 symmetry augmentation (12-fold)
- [x] MCTS: PUCT search, interleaved sub-move tree, ZoI pruning
- [x] Batched GPU evaluator for parallel MCTS
- [x] Training pipeline: replay buffer, self-play worker, trainer
- [x] Evaluation gating (candidate vs best)
- [x] Reanalysis (re-search old positions)
- [x] Training entry point (train.py) with full AlphaZero loop
- [x] Interactive play (play.py) with human-vs-AI mode
- [x] Visualization tools (hex board rendering, training curves)
- [x] 3-phase config files with all hyperparameters
- [x] Research documents (6 papers, ~30K words)
- [x] Code review: cross-module interface consistency
- [x] All 26 Python files pass syntax check
- [x] Full pipeline smoke test (self-play → buffer → train → checkpoint)
- [ ] Add unit tests for game engine (win detection, turn logic edge cases)
- [ ] Add unit tests for MCTS backup logic (value negation at player transitions)
- [ ] Add unit tests for feature extraction (coordinate mapping round-trip)
- [ ] Profile self-play bottleneck (Python MCTS vs NN inference)
- [ ] Benchmark: games/hour on single GPU with Phase 1 config
- [ ] Train Phase 1 to convergence (~100 iterations, beat random >99%)
- [ ] Verify network beats greedy heuristic >80%

### Phase 2: Full-Scale Training
- [ ] Rewrite MCTS hot loop in C++ or Rust with Python bindings
- [ ] Implement async self-play (separate process for game generation)
- [ ] Scale to 15-block/192-channel network
- [ ] Enable multi-scale training (19×19, 25×25, 31×31)
- [ ] Enable playout cap randomization
- [ ] Enable reanalysis in training loop
- [ ] Implement Elo tracking with BayesElo
- [ ] Train to ~2000 Elo (diverse openings, mid-game accuracy >70%)
- [ ] Analyze learned strategies: opening preferences, threat patterns
- [ ] Create tactical puzzle test suite for progress benchmarking

### Phase 3: Refinement
- [ ] Scale to 20-block/256-channel network
- [ ] Implement Gumbel MuZero search (sequential halving)
- [ ] Experiment with SE-ResNet (squeeze-and-excitation blocks)
- [ ] Experiment with p6-equivariant convolutions (replace D6 augmentation)
- [ ] Adversarial testing: opponents that play in distant regions
- [ ] Test generalization to unseen board sizes (37×37)
- [ ] Measure performance degradation under reduced MCTS budget
- [ ] Extract opening book from strongest checkpoint
- [ ] Investigate game-theoretic outcome (first-player win or draw?)
- [ ] Publish analysis of learned strategies and novel tactical concepts

### Infrastructure
- [ ] Add CI (GitHub Actions: syntax check + import smoke tests)
- [ ] Add proper pytest test suite
- [ ] Add type checking with mypy
- [ ] Dockerize training pipeline
- [ ] Add wandb/tensorboard integration for training monitoring
- [ ] Implement distributed self-play (Ray or similar)
- [ ] Add model export (ONNX) for deployment

### Research Extensions
- [ ] Compare single policy head vs dual policy heads (first/second sub-move)
- [ ] Experiment with Transformer-based architecture for long-range patterns
- [ ] Investigate curriculum learning (4-in-a-row → 5-in-a-row → 6-in-a-row)
- [ ] Study effect of ZoI margin on playing strength (margin 2 vs 3 vs 4)
- [ ] Analyze whether the network learns the strategy-stealing argument
- [ ] Explore transfer learning from Connect6 or Hex game models

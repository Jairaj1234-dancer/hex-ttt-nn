# Hex-TTT-NN: AlphaZero for Infinite Hexagonal Tic-Tac-Toe

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An AlphaZero-style neural network and MCTS system for playing **Infinite Hexagonal Tic-Tac-Toe** -- a strategic two-player game on an unbounded hexagonal grid where players race to form 6-in-a-row.

## Game Rules

The game is played on an **infinite flat-top hexagonal grid** using axial coordinates `(q, r)`.

- **Players** alternate turns. Player 1 goes first.
- **Stones per turn**: Each turn a player places **2 stones**, except Player 1's very first turn which is only **1 stone** (opening balance rule).
- **Win condition**: The first player to form an unbroken line of **6 stones** along any of the three hex axes wins.
- **Board**: Conceptually infinite. The search is constrained to a zone-of-interest around existing stones.

```
        -2  -1   0   1   2        <- q (axial)
         .   .   .   .   .    r=-2
          .   .   .   .   .   r=-1
         .   .   *   .   .    r=0     * = Player 1
          .   .   o   .   .   r=1     o = Player 2
         .   .   .   .   .    r=2
```

The three hex axes for line detection:
- **East**: direction `(1, 0)`
- **Northeast**: direction `(1, -1)`
- **North**: direction `(0, -1)`

## Architecture Overview

The system follows the AlphaZero paradigm: a dual-headed neural network guides Monte Carlo Tree Search, trained entirely through self-play.

```
Input Features (12 planes, H x W)
        |
  HexResNet Backbone (shared residual tower)
        |
   +----+----+----------+-----------+
   |         |           |           |
Policy    Value     Ownership    Threat
 Head      Head       Head        Head
(H*W)     [-1,1]   (3,H,W)      (2,)
```

### Module Overview

| Directory    | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `game/`     | Core game logic: hex grid math, board state, rules engine       |
| `nn/`       | Neural network: ResNet backbone, feature extraction, D6 symmetry |
| `mcts/`     | Monte Carlo Tree Search: PUCT selection, ZoI pruning            |
| `training/` | Self-play, replay buffer, trainer, evaluator, reanalyzer        |
| `analysis/` | Visualization tools, opening book extraction                    |
| `configs/`  | YAML configuration files for each training phase                |
| `research/` | Background research documents on AlphaZero and hex grids        |

## Key Features

- **KataGo auxiliary targets**: Ownership and threat prediction heads provide richer training signal from the shared backbone, accelerating learning.
- **D6 symmetry augmentation**: The full 12-element dihedral group of the hexagonal grid (6 rotations + 6 reflections) is applied on-the-fly during sampling for data-efficient training.
- **Dynamic windowing for infinite board**: A sliding feature window centred on the stone centroid converts the sparse infinite grid into fixed-size CNN inputs, enabling play on truly unbounded boards.
- **Interleaved sub-move MCTS**: The search tree correctly handles the 2-stones-per-turn mechanic, with value negation only at player transitions (not within same-player sub-moves).
- **Zone-of-Interest pruning**: MCTS action space is restricted to empty cells within a configurable hex-distance margin of existing stones, keeping the branching factor tractable.
- **Playout cap randomization**: Reduces self-play compute by using fewer MCTS simulations for a fraction of moves while maintaining training quality.
- **MuZero-style reanalysis**: Older replay buffer positions are re-searched with the latest network, keeping policy targets fresh without replaying entire games.
- **Gumbel search support** (Phase 3): Configuration support for Gumbel AlphaZero/MuZero-style search with sequential halving.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

Requirements: `torch>=2.0.0`, `numpy>=1.24.0`, `pyyaml>=6.0`, `matplotlib>=3.7.0`, `tqdm>=4.65.0`

### Train (Phase 1 -- prototype)

```bash
python train.py --config configs/phase1.yaml
```

### Resume training from checkpoint

```bash
python train.py --config configs/phase2.yaml --checkpoint checkpoints/latest.pt
```

### Play against the AI

```bash
python play.py --checkpoint checkpoints/best.pt --mode human-vs-ai
```

### Watch AI vs AI

```bash
python play.py --checkpoint checkpoints/best.pt --mode ai-vs-ai --simulations 400
```

### Human vs Human (no network needed)

```bash
python play.py --mode human-vs-human
```

## Training Phases

| Phase | Network               | MCTS Sims | Grid | Key Features                  | Timeline     |
|-------|-----------------------|-----------|------|-------------------------------|--------------|
| 1     | 8 blocks, 128 ch      | 200       | 19   | Pipeline validation           | Weeks 1--3   |
| 2     | 15 blocks, 192 ch     | 600       | 31   | Playout cap, reanalysis       | Weeks 4--10  |
| 3     | 20 blocks, 256 ch     | 800       | 31   | Gumbel search, warm restarts  | Weeks 11--16 |

Each phase builds on the previous checkpoint. Phase 1 validates that the full self-play/train/evaluate loop works end-to-end. Phase 2 scales up the network and search, adding playout cap randomization and reanalysis. Phase 3 uses the largest network with Gumbel search and aggressive reanalysis for maximum strength.

## Project Structure

```
hex-ttt-nn/
|-- train.py                  # Main training script (AlphaZero loop)
|-- play.py                   # Interactive play / demo
|-- requirements.txt
|-- pyproject.toml
|
|-- configs/
|   |-- phase1.yaml           # Prototype config (small network, 200 sims)
|   |-- phase2.yaml           # Full-scale config (larger network, 600 sims)
|   |-- phase3.yaml           # Refinement config (largest network, 800 sims)
|
|-- game/
|   |-- hex_grid.py           # Axial coordinate math, D6 symmetry transforms
|   |-- board.py              # Sparse board with Zobrist hashing
|   |-- rules.py              # GameState, move application, legal moves
|   |-- zobrist.py            # Zobrist hash table for board hashing
|
|-- nn/
|   |-- model.py              # HexTTTNet: dual-headed network with aux heads
|   |-- hex_conv.py           # HexResNet backbone (residual tower)
|   |-- features.py           # Feature extraction (12 input planes)
|   |-- symmetry.py           # D6 symmetry remap indices for augmentation
|
|-- mcts/
|   |-- search.py             # MCTS with PUCT selection and NN evaluation
|   |-- node.py               # MCTSNode with interleaved sub-move backup
|   |-- zoi.py                # Zone-of-Interest computation and masking
|   |-- parallel.py           # Parallel MCTS with virtual loss
|
|-- training/
|   |-- trainer.py            # Network trainer (optimizer, scheduler, checkpoint)
|   |-- self_play.py          # Self-play game generation
|   |-- replay_buffer.py      # Replay buffer with recency weighting and D6 augmentation
|   |-- evaluator.py          # Candidate vs best network match evaluation
|   |-- reanalyze.py          # MuZero-style reanalysis of stored positions
|
|-- analysis/
|   |-- visualize.py          # Matplotlib hex board rendering, loss curves, Elo plots
|   |-- opening_book.py       # Opening pattern extraction and analysis
|
|-- research/
|   |-- 00_synthesis_and_roadmap.md
|   |-- 01_alphago_alphazero_architecture.md
|   |-- 02_post_alphazero_developments.md
|   |-- 03_hex_grids_spatial_architectures.md
|   |-- 04_mcts_adaptations.md
|   |-- 05_training_pipeline_game_theory.md
```

## Analysis Tools

### Training visualization

```python
from analysis.visualize import plot_training_curves, plot_elo_progression

plot_training_curves("checkpoints/logs/training_log.jsonl", save_path="loss_curves.png")
plot_elo_progression("checkpoints/logs/elo_log.jsonl", save_path="elo_progress.png")
```

### Board visualization with policy heatmap

```python
from analysis.visualize import plot_hex_board

plot_hex_board(game_state, policy=policy_dict, value=0.42, save_path="board.png")
```

### Opening book extraction

```bash
python -m analysis.opening_book --games-dir game_logs/ --depth 6 --output opening_book.json
```

## References

- Silver, D. et al. (2016). *Mastering the game of Go with deep neural networks and tree search.* Nature, 529, 484--489.
- Silver, D. et al. (2017). *Mastering the game of Go without human knowledge.* Nature, 550, 354--359.
- Silver, D. et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science, 362, 1140--1144.
- Wu, D. J. (2019). *Accelerating Self-Play Learning in Go.* arXiv:1902.10565. (KataGo)
- Danihelka, I. et al. (2022). *Policy improvement by planning with Gumbel.* ICLR 2022.

## License

MIT License. See [LICENSE](LICENSE) for details.

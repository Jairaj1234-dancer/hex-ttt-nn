# Autoresearch Program: Beat Eisenstein at 6-in-a-row Hex TTT

## Goal

Maximize **win rate vs EisensteinGreedy** (metric in [0.0, 1.0]) for a ResNet neural network playing Infinite Hexagonal Tic-Tac-Toe (win_length=6, 2 stones per turn).

Current best: ~84% raw policy. Target: >90% win rate.

## The Three Files

| File | Who Edits | Purpose |
|------|-----------|---------|
| `prepare.py` | **NOBODY** (fixed) | Evaluation harness, data loading, metric measurement |
| `train.py` | **AI ONLY** | Training loop, loss functions, data generation, hyperparameters |
| `program.md` | **HUMAN ONLY** | These instructions |

## What You Can Change (in train.py)

- **Training strategy**: loss weights, learning rate, schedule, optimizer settings
- **Data generation**: adversarial game generation, position mixing, curriculum
- **Loss function**: policy weight, value weight, auxiliary losses, label smoothing
- **Regularization**: dropout, weight decay, gradient clipping
- **Data augmentation**: D6 symmetry, position weighting, hard example mining

## What You CANNOT Change

- `prepare.py` — the evaluation is sacred
- The model architecture (6 blocks, 96 channels, 13x13 grid) — it's fixed in the checkpoint
- The game rules (win_length=6, hexagonal grid, 2-move turns)
- The output format: `>>> METRIC: {value}` must appear exactly once at the end

## Constraints

- **Time budget**: Each experiment gets 10 minutes of training (wall clock)
- **Device**: Apple M-series GPU via MPS. Be mindful of memory (~8GB available)
- **Warm start**: Always train from the current best checkpoint. Never from scratch.
- **Simplicity**: Prefer clean, simple changes. One hypothesis per experiment.
- **Record your reasoning**: Add a `# EXPERIMENT: description` comment at the top of train.py

## Context

The model is a ResNet (6 blocks, 96 channels) with:
- Policy head: flat logits over 13x13 grid, masked to legal moves
- Value head: conv -> flatten -> FC -> tanh
- Auxiliary heads: ownership (3-class per cell), threats (2 scalars)

EisensteinGreedy is a 1-ply heuristic that scores moves by chain length (attack*3 + defense*2). It cannot see forks or double threats. The model already beats it ~84% with raw policy.

**Key training insight**: The model uses a transposed (column-major) policy indexing:
```python
bx, by = axial_to_brick(move.q, move.r, cq, cr, GRID_SIZE)
row = by + half   # actually col+half
col = bx + half   # actually row+half
idx = row * GRID_SIZE + col  # column-major
```
All training data and inference must use this same mapping consistently.

## Strategy Hints

Ways to push from 84% toward 90%+:
- **More adversarial data**: Generate games where NN loses, train on Eisenstein's corrections
- **Fork emphasis**: Weight fork/double-threat positions higher in training
- **Value head calibration**: Better value predictions enable MCTS search boost
- **Defensive positions**: Train on positions where NN missed a block
- **Curriculum**: Progressively harder opponent mixes
- **Loss tuning**: Adjust policy_weight vs value_weight ratio

Things NOT worth trying:
- Architecture changes (checkpoint is fixed at 6b/96ch)
- MCTS during training (too slow for 10 min budget with current value head)
- Self-play from scratch (need more time)

## How the Loop Works

1. AI reads `train.py` and `experiment_log.jsonl`
2. AI forms a hypothesis and modifies `train.py`
3. `run.py` executes `train.py` with a 10-minute training budget
4. Metric is extracted from output (win rate vs Eisenstein, 40 games)
5. If metric > current best -> ACCEPT (save model)
6. If metric <= current best -> REJECT (revert train.py)
7. Go to step 1

## Lessons Learned

1. **Adversarial training works**: Imitation (60%) -> Fork emphasis (75%) -> Adversarial refinement (84%)
2. **Don't overtrain on losses only**: Training exclusively on lost-game positions causes catastrophic forgetting
3. **Balance attack and defense**: Mix winning reinforcement with defensive corrections
4. **Value head is weak**: MSE ~0.25. Fine-tuning helps but search still underperforms raw policy.
5. **The model already handles threats implicitly**: Adding explicit 1-ply tactical overrides hurts performance

## Do NOT pause to ask the human. Run experiments continuously until stopped.

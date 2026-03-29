# Self-Play Training Pipeline Design and Game-Theoretic Considerations for AlphaZero-Style Learning on Infinite Hexagonal Tic-Tac-Toe

**Document 05 | Research Series: Hex-TTT Neural Network**
**Date: 2026-03-29**

---

## Abstract

This document presents a comprehensive analysis of the self-play training pipeline design and game-theoretic foundations required to build an AlphaZero-style system (Silver et al., 2017; Silver et al., 2018) for Infinite Hexagonal Tic-Tac-Toe -- a deterministic, perfect-information, two-player, zero-sum game played on an unbounded hexagonal grid where players place 2 stones per turn (1 on the first move), and 6-in-a-row along any hex axis wins. We analyze game complexity, first-player advantage, reward shaping, exploration strategies, and provide a concrete, actionable training plan with specific hyperparameter recommendations. Throughout, we draw heavily on precedents from KataGo (Wu, 2019), Leela Chess Zero (Pascutto & Linscott, 2019), Polygames (Cazenave et al., 2020), and the Connect6 literature (Wu & Huang, 2005; Wu & Huang, 2006).

---

## Table of Contents

1. [Game-Theoretic Analysis of Infinite Hex Tic-Tac-Toe](#1-game-theoretic-analysis-of-infinite-hex-tic-tac-toe)
2. [Game Complexity Estimation](#2-game-complexity-estimation)
3. [Self-Play Training Pipeline Design](#3-self-play-training-pipeline-design)
4. [Reward Shaping and Training Signal](#4-reward-shaping-and-training-signal)
5. [Handling the Exploration-Exploitation Tradeoff in Self-Play](#5-handling-the-exploration-exploitation-tradeoff-in-self-play)
6. [Evaluation and Progress Tracking](#6-evaluation-and-progress-tracking)
7. [Practical Engineering Considerations](#7-practical-engineering-considerations)
8. [Recommended Training Pipeline for Infinite Hex Tic-Tac-Toe](#recommended-training-pipeline-for-infinite-hex-tic-tac-toe)
9. [References](#references)

---

## 1. Game-Theoretic Analysis of Infinite Hex Tic-Tac-Toe

### 1.1 Game Definition and Formal Properties

Infinite Hexagonal Tic-Tac-Toe (hereafter **InfHexTTT**) is defined as follows:

- **Board**: An infinite hexagonal grid (equivalently, the vertices of a triangular tiling, or the cells of a hexagonal tiling with 6-connectivity).
- **Players**: Two players, Black and White, alternating turns.
- **Moves**: Each turn, a player places **2 stones** on any unoccupied cells. Exception: Black's first move is **1 stone only**.
- **Win condition**: The first player to form a contiguous line of **6 or more** of their stones along any of the **3 hex axes** wins.
- **Hex axes**: In a standard hexagonal grid, there are 3 axes of symmetry (0, 60, 120 degrees), yielding 3 directions for winning lines (compared to 4 in a square grid: horizontal, vertical, 2 diagonals).

This game is deterministic, perfect-information, two-player, and zero-sum. By the Zermelo-von Neumann theorem, it has a determined outcome under optimal play: either first-player wins, second-player wins, or it is a draw.

### 1.2 The Strategy-Stealing Argument

The classical strategy-stealing argument (Nash, 1952; Beck, 2008) applies to maker-maker games: if the second player had a winning strategy, the first player could "steal" it by making an arbitrary first move and then following the second player's strategy. An extra stone on the board can never be a disadvantage in a maker-maker game. Therefore, the second player cannot have a winning strategy, and the game is either a **first-player win** or a **draw**.

For InfHexTTT, the strategy-stealing argument applies with a subtlety: the first move is 1 stone while subsequent moves are 2 stones. If Black had a winning strategy as second player, White could steal it. However, the argument requires that having extra stones is never harmful -- which holds in maker-maker games. Therefore:

> **Theorem (informal)**: InfHexTTT is either a first-player win or a draw. It cannot be a second-player win.

### 1.3 Comparison to Connect6

InfHexTTT is structurally very similar to **Connect6** (Wu & Huang, 2005; Wu & Huang, 2006), which was deliberately designed with game-theoretic balance in mind:

| Property | Connect6 | InfHexTTT |
|---|---|---|
| Grid | Infinite square (in theory; 19x19 in practice) | Infinite hexagonal |
| Win length | 6-in-a-row | 6-in-a-row |
| Stones per turn | 2 (first move: 1) | 2 (first move: 1) |
| Directions | 4 (H, V, 2 diag) | 3 (hex axes) |
| First-player advantage | Near-balanced by design | Expected near-balanced |

Wu and Huang (2006) proved that Connect6 satisfies the strategy-stealing argument and showed empirically that the 2-stone-per-turn mechanic with the first-move restriction creates extraordinary balance. On a 19x19 board, strong Connect6 programs achieve near 50-50 win rates between Black and White, far more balanced than Gomoku (5-in-a-row, 1 stone/turn), where Black wins easily without restrictions.

The key insight from Connect6 theory: **the ratio of stones-per-turn to win-length matters**. In Gomoku (1 stone, 5 to win), the ratio is 1/5 = 0.2. In Connect6 (2 stones, 6 to win), the ratio is 2/6 = 1/3. The higher ratio means both players can advance threats at similar rates, yielding balance. InfHexTTT has the same 2/6 = 1/3 ratio, but on a hexagonal grid with only 3 directions instead of 4.

**Implication**: The reduced number of winning directions (3 vs. 4) slightly favors defense, since threats propagate in fewer directions. This suggests InfHexTTT may be **even more balanced than Connect6** or slightly more draw-prone. However, the hexagonal geometry also means each cell participates in 3 lines (vs. 4 in square grid), which slightly simplifies both attack and defense calculations.

### 1.4 Is InfHexTTT a Win or a Draw?

For infinite-board games with k-in-a-row requirements:

- **k = 1, 2, 3, 4**: First player wins trivially or near-trivially (Hales & Jewett, 1963).
- **k = 5, 1 stone/turn (Gomoku variant)**: First player wins (Allis et al., 1994).
- **k = 5, 2 stones/turn**: Likely a draw or very close to balanced.
- **k = 6, 2 stones/turn (Connect6)**: Conjectured to be a draw on an infinite board (Wu & Huang, 2006). On finite boards, empirically very close to balanced.
- **k = 9+, 1 stone/turn**: Known to be a draw by Hales-Jewett-type arguments with pairing strategies.

For InfHexTTT (k=6, 2 stones/turn, hex grid), the most likely outcome under optimal play is a **draw**, by analogy with Connect6. The hexagonal grid's 3 directions (vs. 4) make it harder to create simultaneous multi-directional threats, further supporting draw-likelihood. However, this is unproven -- it remains an open combinatorial game theory question.

**For training purposes**: We should assume the game is very close to drawn under optimal play but has rich tactical content with many opportunities for decisive wins when either player errs.

### 1.5 The "First Move = 1 Piece" Rule as a Balancing Mechanism

This rule is directly borrowed from Connect6 design principles (Wu & Huang, 2005). Without it:

- Black places 2 stones, then White places 2, alternating. Black always has a 2-stone lead. With 2 stones/turn and 6-to-win, a 2-stone lead is enormous -- it means Black always has an extra "half-move" in terms of threat creation.
- The strategy-stealing argument becomes even stronger: Black wins.

With the restriction (Black places 1 on move 1):

- After move 1: Black has 1, White has 0 (Black leads by 1).
- After move 2: Black has 1, White has 2 (White leads by 1).
- After move 3: Black has 3, White has 2 (Black leads by 1).
- Pattern: The lead alternates, with Black always ahead by exactly 1 stone after their turn. This is the minimal possible first-player advantage.

This is the same stone-count dynamic as standard Gomoku (1 stone/turn), but the doubled placement rate means both players can sustain more complex threat structures simultaneously.

### 1.6 Swap Rule / Pie Rule Considerations

Even with the 1-stone first move rule, if training reveals a persistent first-player advantage, we can implement the **swap rule** (pie rule):

1. Black places 1 stone.
2. White chooses: either (a) accept and play as White, or (b) swap colors and become Black.
3. Play continues normally.

This creates a "cake-cutting" equilibrium: Black must choose a first move that is exactly fair, because if it's too good, White will swap. Under the pie rule, the game becomes provably balanced in expected value (assuming rational players), regardless of the underlying game-theoretic value.

**For self-play training**: We should initially train **without** the swap rule to understand the true game dynamics, then optionally add it later if needed for balance. The swap rule complicates the MCTS search tree (the second player's first "move" is a binary swap decision), but it is tractable.

### 1.7 Threat Sequences on the Hexagonal Grid

On a square grid (Connect6), a winning threat sequence typically involves creating simultaneous threats along 2+ of the 4 directions. On the hexagonal grid, there are only 3 directions, which has important tactical implications:

- **Fewer forking opportunities**: On a square grid, a stone at intersection can participate in 4 lines. On a hex grid, only 3. This means fork-based tactics (simultaneous threats in multiple directions) are somewhat less frequent.
- **Denser neighbor interactions**: Each hex cell has 6 neighbors (vs. 4 in square, or 8 if counting diagonals). This means local play is more tightly coupled.
- **Broader winning lines**: 6-in-a-row on a hex grid means each winning line spans 6 cells with some lateral offset (depending on direction). The hex geometry means winning lines are not "straight" in Euclidean terms for all 3 directions simultaneously, which affects visual and strategic intuition.

**Key tactical concepts for the neural network to learn**:
- **Double threats**: Placing 2 stones that each extend or create a line, forcing the opponent to respond to both.
- **Relay threats**: Creating a chain where blocking one threat enables another.
- **Distance-based judgment**: On an infinite board, knowing which regions matter and which are "too far away" to interact is crucial. This is analogous to "influence" in Go.

---

## 2. Game Complexity Estimation

### 2.1 Effective Board Size

Although the board is infinite, practical play is bounded. In Connect6, competitive games on a 19x19 board (361 intersections) rarely fill more than ~60-80 stones before a winner emerges. InfHexTTT with 3 directions (fewer than Connect6's 4) likely produces slightly longer games but with a similar effective footprint.

**Estimation approach**: In a k-in-a-row game with 2 stones per turn, the minimum game length is:
- Minimum: 6 turns (Black: 1 + 2 + 2 = 5... no. Let's count: Move 1: Black places 1. Move 2: White places 2. Move 3: Black places 2. After move N, Black has 1 + 2*(ceil(N/2)-1) stones for N odd, White has 2*floor(N/2) stones for N even... Actually, let's just count directly.)
  - After Black's 1st turn (turn 1): B=1, W=0
  - After White's 1st turn (turn 2): B=1, W=2
  - After Black's 2nd turn (turn 3): B=3, W=2
  - After White's 2nd turn (turn 4): B=3, W=4
  - After Black's 3rd turn (turn 5): B=5, W=4
  - After White's 3rd turn (turn 6): B=5, W=6
  - White can theoretically win by turn 6 (6 stones), Black by turn 7 (7 stones, needing 6). But forcing a win in minimum moves is essentially impossible against any reasonable play.

- **Typical game length estimate**: By analogy with Connect6, typical games last **30-60 turns** (60-120 half-moves). Total stones on the board at game end: ~60-120. The effective board area is roughly a convex hull around placed stones plus a margin of ~6 cells (the win length). Estimating the effective board as roughly **15x15 to 25x25 hex cells** (in axial coordinates), giving **~170-480 effective cells** in a hexagonal grid.

For training with a finite representation, we use a **cropped window** approach. A reasonable window: **31x31 hex grid** (~750 cells after hexagonal packing), re-centered dynamically during play.

### 2.2 State-Space Complexity

State-space complexity = log10(number of legal positions).

Each cell in the effective area can be: empty, Black, or White. For an effective area of ~750 cells:
- Upper bound: 3^750 ~ 10^358. But most positions are illegal (unequal stone counts, containing a completed 6-in-a-row, etc.).
- Adjusted estimate: Comparable to Go's state-space complexity of ~10^170 (for 19x19) but on a smaller effective board. With 750 cells and typical stone counts of 40-80, the number of positions is approximately C(750, 40) * C(710, 39) ~ 10^100 to 10^140.

| Game | State-Space Complexity (log10) | Game-Tree Complexity (log10) |
|---|---|---|
| Tic-Tac-Toe | 3 | 5 |
| Chess | 47 | 123 |
| Connect6 (19x19) | ~80-100 | ~140-170 |
| Go (19x19) | 170 | 360 |
| **InfHexTTT (est.)** | **~100-140** | **~120-180** |

### 2.3 Game-Tree Complexity and Branching Factor

The **branching factor** is the critical metric for MCTS performance:

- **Single-stone games**: Branching factor = number of empty cells. For Go (19x19): ~250 average. For Chess: ~35.
- **Two-stone turns**: The branching factor for a 2-stone move is C(empty, 2) = empty*(empty-1)/2. For 700 empty cells: C(700, 2) = 244,650. For 650 empty cells: C(650, 2) = 211,025.

This is an **enormous** branching factor -- roughly **200,000-250,000** for a 2-stone move, compared to ~250 for Go. This is the fundamental computational challenge of InfHexTTT.

However, several factors mitigate this:

1. **Locality**: Strong moves are almost always near existing stones. The neural network policy head can learn to assign near-zero probability to distant cells, effectively pruning the search space to ~50-200 plausible cells, yielding a practical branching factor of C(100, 2) ~ 4,950 for the 2-stone move.
2. **Decomposition**: A 2-stone move can be decomposed into 2 sequential single-stone placements for the policy network, reducing the output space from C(N, 2) to 2*N. This is the approach used in Connect6 programs (Wu & Huang, 2006) and Polygames (Cazenave et al., 2020). The MCTS tree then treats each half-move as a separate node.
3. **Symmetry**: The hexagonal grid has 6-fold rotational symmetry and reflection symmetry (dihedral group D6, order 12). Exploiting this reduces effective state space by up to 12x.

**Recommended approach**: Decompose 2-stone turns into 2 sequential single-stone placements. This is well-established in the literature (Wu, 2001; Cazenave et al., 2020) and reduces the effective branching factor to ~100-300 per half-move (with good policy network guidance), comparable to Go.

### 2.4 Average Game Length

Using the Connect6 analogy and adjusting for hex geometry:
- Connect6 average game length: ~40-50 full turns (Wu & Huang, 2006, competitive play).
- InfHexTTT: 3 winning directions vs. Connect6's 4 means slightly longer games (fewer forced wins). Estimate: **40-70 full turns**, or **80-140 half-moves** (including decomposed 2-stone turns as 2 half-moves each, the total is ~160-280 half-move MCTS nodes per game).

**For MCTS simulation budgets**: With ~200 half-moves per game and a target of 400-800 simulations per move (reduced from AlphaZero's 800 for Go, since our effective branching factor with good policy is similar), each self-play game requires ~200 * 600 = 120,000 MCTS simulations. At ~5,000 simulations/second on a modern GPU (conservative for a lightweight network), each game takes ~24 seconds. This is tractable.

---

## 3. Self-Play Training Pipeline Design

### 3.1 The AlphaZero Training Loop

The core loop, following Silver et al. (2018) and refined by subsequent work (Wu, 2019; Pascutto & Linscott, 2019):

```
Initialize network f_θ with random weights
Initialize replay buffer B

Loop:
  1. SELF-PLAY: Generate games using f_θ + MCTS
     - For each position, run MCTS to get policy π and value v
     - Store (state, π, v, outcome z) tuples
     - Add games to replay buffer B

  2. TRAINING: Sample minibatches from B
     - Loss = L_policy(π, p_θ) + L_value(z, v_θ) + c * L_reg(θ)
     - L_policy = cross-entropy between MCTS policy π and network policy p_θ
     - L_value = MSE between game outcome z and network value v_θ
     - Update θ via SGD with momentum or Adam

  3. EVALUATION: Periodically pit new θ' against current best θ*
     - If θ' wins >55% of games, replace θ* ← θ'
     - Track Elo ratings over time

  4. CHECKPOINT: Save θ, replay buffer state, training metrics
```

### 3.2 Replay Buffer Design

The replay buffer is critical for training stability and sample efficiency.

**Size**: The buffer should hold enough games to represent diverse positions while remaining fresh enough to reflect current play strength.

- **AlphaZero (Go)**: Used the most recent 500,000 games (Silver et al., 2018).
- **KataGo**: Used a sliding window of the most recent ~250,000-1,000,000 games, with a preference for recent data (Wu, 2019).
- **For InfHexTTT**: With ~200 positions per game and a target buffer of ~2-5 million positions, we need **10,000-25,000 games**. At early training stages, a smaller buffer of 5,000 games suffices.

**Sampling strategy**:

| Strategy | Pros | Cons | Recommendation |
|---|---|---|---|
| Uniform | Simple, stable | Oversamples stale data | Baseline |
| Prioritized (by loss) | Focuses on hard examples | Can overfit to outliers; added complexity | Use sparingly |
| Recency-weighted | Tracks improving play quality | May underrepresent rare positions | **Recommended** |
| Mixed: 75% recent, 25% uniform | Balance of freshness and diversity | Slightly more complex | **Best for our case** |

**Data freshness**: Following KataGo (Wu, 2019), we use a **sliding window** with exponential decay. Position weight = exp(-age / tau), where age is measured in training steps since the position was generated, and tau is tuned so that positions older than ~3 training epochs have weight < 0.1.

**Position storage format**: Each position is stored as:
- Board state: sparse representation (list of (coordinate, color) pairs), typically 40-100 entries.
- MCTS policy: sparse distribution over legal moves (top-K with K~50-100 for the first stone of the 2-stone move, then top-K for the second).
- MCTS value estimate: float32.
- Game outcome: {-1, 0, +1} from the perspective of the player to move.
- Auxiliary targets (see Section 4): ownership map, score estimate, etc.

Estimated storage per position: ~2-4 KB (sparse). Buffer of 5 million positions: ~10-20 GB. This fits comfortably in RAM.

### 3.3 Asynchronous vs. Synchronous Self-Play and Training

**Synchronous** (AlphaZero-style):
- Generate N games with current network.
- Train on those games.
- Evaluate new network.
- Replace if better.
- Simple, stable, but GPU idle during self-play and vice versa.

**Asynchronous** (KataGo / Leela Chess Zero style):
- Self-play workers continuously generate games using the latest network.
- Training continuously samples from the replay buffer and updates the network.
- Self-play workers periodically fetch the latest network weights.
- Much higher throughput; GPU utilization near 100%.

**Recommendation for InfHexTTT**: Start with **synchronous** for simplicity and debugging. Transition to **asynchronous** once the pipeline is validated. For a single-machine setup (1-2 GPUs), the overhead of async coordination may not be worth it initially. For multi-machine setups, async is strongly preferred.

**Concrete async architecture**:
```
Self-play workers (CPU/GPU) --[games]--> Replay Buffer (shared memory / Redis)
                                              |
                                         [minibatches]
                                              |
                                         Training loop (GPU) --[weights]--> Weight server
                                              |
Self-play workers <--[latest weights]-- Weight server
```

### 3.4 Checkpointing and Model Evaluation

**Checkpointing frequency**: Every 1,000-5,000 training steps (or every ~30 minutes of wall time). Each checkpoint saves:
- Network weights (θ).
- Optimizer state (momentum buffers, Adam moments).
- Replay buffer metadata (not necessarily the full buffer, which can be reconstructed).
- Training step count, loss curves, Elo estimates.

**Model evaluation protocol** (gating):

Following AlphaZero, a new checkpoint replaces the current best if it achieves a >55% win rate in a head-to-head match. Details:

- **Number of evaluation games**: 400 games (200 as each color) provides a standard error of ~2.5% on win rate, sufficient to distinguish a 55% win rate from 50% at ~95% confidence.
- **MCTS budget for evaluation**: Same as training self-play (e.g., 600 simulations/move) or higher (e.g., 1200) for more reliable evaluation.
- **Temperature**: Use τ=0 (argmax) for evaluation games to remove stochasticity, or τ=0.1 for minimal noise.

**Elo tracking**: Maintain a running Elo rating for each checkpoint. Use BayesElo or a simple logistic model:

P(win) = 1 / (1 + 10^((R_opponent - R_player) / 400))

Update ratings using the standard Elo formula with K-factor = 32. Over the course of training, we expect Elo to increase roughly logarithmically with training compute, as observed in AlphaZero and KataGo.

**Expected Elo progression** (rough estimate for InfHexTTT):
- Random play: 0 Elo
- After 1 day of training (single GPU): ~500 Elo (learns basic tactics)
- After 1 week: ~1500 Elo (learns intermediate strategy)
- After 1 month: ~2500-3000 Elo (approaching expert level)
- Diminishing returns beyond ~3000 Elo

### 3.5 Curriculum Learning Considerations

The infinite board and 6-in-a-row requirement make InfHexTTT complex from the start. Curriculum learning can accelerate early training:

**Proposed curriculum**:

| Phase | Variant | Duration | Purpose |
|---|---|---|---|
| 0 (optional) | 4-in-a-row, 11x11 hex board, 1 stone/turn | 1-2 days | Learn basic hex geometry and line-forming |
| 1 | 5-in-a-row, 15x15 hex board, 2 stones/turn | 2-3 days | Learn 2-stone tactics |
| 2 | 6-in-a-row, 19x19 hex board, 2 stones/turn | 3-5 days | Target game on bounded board |
| 3 | 6-in-a-row, 31x31 hex board, 2 stones/turn | Ongoing | Near-infinite; production config |

**Transfer learning**: Initialize each phase from the previous phase's weights. The network architecture must accommodate different board sizes -- use a fully convolutional architecture (no fully-connected layers that depend on board size) so weights transfer seamlessly. This is the approach used in KataGo (Wu, 2019), which trains on multiple board sizes simultaneously.

**Alternative: multi-scale training** (KataGo-style). Instead of sequential curriculum, sample board sizes randomly during training:
- 20% of games on 15x15
- 30% on 19x19
- 30% on 25x25
- 20% on 31x31

This produces a network that generalizes across scales and avoids catastrophic forgetting of small-board tactics.

### 3.6 Hardware Requirements and Training Time Estimates

**Baseline configuration** (hobbyist/researcher):
- 1x NVIDIA RTX 4090 (24 GB VRAM, ~330 TFLOPS FP16)
- 64 GB RAM
- 12-core CPU for self-play MCTS

**Estimates for baseline config**:
- Network: 10-block, 128-filter ResNet. Forward pass: ~0.5 ms/position (batched).
- MCTS: 600 simulations/move, ~200 moves/game. With neural network batching: ~30 seconds/game.
- Self-play throughput: ~120 games/hour = ~24,000 positions/hour.
- Training throughput: ~10,000 minibatch updates/hour (batch size 256).
- **Time to basic competence** (~1000 Elo): ~2-3 days.
- **Time to strong play** (~2500 Elo): ~2-4 weeks.

**Scaled configuration** (lab):
- 4x A100 GPUs (80 GB each)
- 2 GPUs for self-play inference, 2 for training
- 256 GB RAM, 32-core CPU

**Estimates for scaled config**:
- Self-play throughput: ~2,000 games/hour.
- Training throughput: ~40,000 updates/hour.
- **Time to strong play**: ~3-5 days.

**Comparison to prior work**:
- AlphaZero (Go): 5,000 TPUs, 72 hours. We are ~1000x less compute; aiming for ~100x less performance.
- KataGo: Achieved superhuman Go in ~20 GPU-days on a V100 with algorithmic improvements. This is our realistic target efficiency.

---

## 4. Reward Shaping and Training Signal

### 4.1 Primary Outcome Signal

The standard AlphaZero approach uses **binary outcomes**: z in {-1, +1} for loss/win from the perspective of the player to move. Draws are represented as z = 0.

For InfHexTTT, we expect draws to be relatively rare in practice (even if the theoretical outcome is a draw, random or weak play will produce decisive results). As training progresses and play approaches optimality, draws may increase. The value head should output a continuous estimate v in [-1, +1], trained with MSE loss:

L_value = (z - v_θ(s))^2

**Draws**: If draws become frequent at high levels, consider a 3-valued outcome {-1, 0, +1} or even the KataGo approach of predicting a score (see Section 4.3).

### 4.2 Auxiliary Training Targets (KataGo-Style)

KataGo (Wu, 2019) demonstrated that auxiliary training targets significantly accelerate learning by providing richer gradient signal. The key auxiliary targets and their applicability to InfHexTTT:

**a) Ownership Prediction**

For each cell, predict the probability that it will be occupied by Black, White, or remain empty at the end of the game. This is extremely valuable in Go for territory estimation and translates to InfHexTTT as:

- **Influence map**: Which regions of the board are "controlled" by each player?
- **Implementation**: An auxiliary head with per-cell softmax output (3 classes: Black, White, empty).
- **Training target**: The actual final board state at game end.
- **Value for InfHexTTT**: **High**. Understanding territorial influence is crucial on an infinite board where deciding where to play (locality) is a core strategic question.

**b) Score / Advantage Estimation**

In Go, KataGo predicts the expected score (territory difference). For InfHexTTT, there is no natural "score" since the game is win/lose. However, we can define proxy scores:

- **Stone advantage proxy**: (Black stones in 5+ lines) - (White stones in 5+ lines). Not very meaningful.
- **Threat count differential**: (Black open threats of length 4+) - (White open threats of length 4+). This is more strategically meaningful.
- **Expected moves to win**: A continuous value representing how many moves the current player needs to force a win (set to infinity / max value for drawn positions).

**Recommendation**: Use **threat count differential** as a score proxy. This provides a continuous signal even in positions far from the terminal state.

**c) Threat Detection (Game-Specific)**

Unique to k-in-a-row games, explicit threat detection as an auxiliary target:

- For each cell, predict: "If Black/White places a stone here, does it create a threat of length k-1 or more?"
- This is a per-cell binary classification task.
- **Training target**: Can be computed exactly from the board state (no need for game outcome).
- **Value for InfHexTTT**: **Very high**. This directly teaches the network the tactical building blocks.

**d) Longest Line Count**

Per-player auxiliary target: predict the length of the longest contiguous line for each player.

- Simple scalar prediction, trained with MSE.
- **Value**: **Moderate**. Provides a signal about how close each player is to winning.

**Recommended auxiliary targets for InfHexTTT** (in priority order):
1. Ownership prediction (per-cell, 3-class) -- strongest signal for spatial reasoning.
2. Threat detection (per-cell, binary per player) -- teaches tactical fundamentals.
3. Longest line count (scalar per player) -- coarse progress signal.
4. Win probability decomposition: P(Black wins), P(White wins), P(draw) -- richer than single value.

**Auxiliary loss weighting**: Following KataGo, weight auxiliary losses at ~0.15-0.25 of the primary loss:

L_total = L_value + L_policy + 0.15 * L_ownership + 0.10 * L_threat + 0.05 * L_longest_line + c * L_reg

### 4.3 Shaped Rewards vs. Pure Outcome

**Pure outcome** (AlphaZero): Only the final game result provides signal. All intermediate positions are labeled with the same z.

**Shaped rewards**: Provide intermediate rewards for achieving sub-goals (e.g., forming a 4-in-a-row, blocking an opponent's threat).

**Analysis for InfHexTTT**:

| Approach | Pros | Cons |
|---|---|---|
| Pure outcome | No reward hacking; learns true value | Sparse signal; slow early learning |
| Shaped rewards | Faster early learning; richer gradients | Risk of reward hacking; may learn suboptimal strategies |
| Pure outcome + auxiliary targets | Best of both worlds; auxiliary targets provide dense signal without distorting the value function | Slightly more complex implementation |

**Recommendation**: Use **pure outcome for the primary value head** with **auxiliary targets for dense signal**. This is the KataGo approach and avoids the pitfalls of reward shaping while still providing rich training signal.

### 4.4 Temperature Scheduling for Self-Play

Temperature controls the stochasticity of move selection during self-play:

- **τ = 1**: Moves sampled proportionally to MCTS visit counts. Maximum exploration.
- **τ → 0**: Moves selected greedily (highest visit count). Maximum exploitation.
- **τ > 1**: More random than visit-count proportional. Rarely useful.

**AlphaZero schedule**: τ = 1 for the first 30 moves, then τ → 0 for the remainder.

**KataGo schedule**: More nuanced -- uses a stochastic "playout cap randomization" (see Section 5) instead of pure temperature scheduling.

**Recommended schedule for InfHexTTT**:

```
For move number m (counting half-moves, i.e., individual stone placements):
  if m <= 20:  τ = 1.0    (first ~10 full turns: full exploration)
  elif m <= 40: τ = 0.5   (next ~10 full turns: moderate exploration)
  else:         τ = 0.1   (remaining moves: near-greedy)
```

The first 20 half-moves (10 full turns) with τ = 1 ensures opening diversity. The taper to near-greedy ensures endgame positions are played accurately, providing clean training signal for the value head.

---

## 5. Handling the Exploration-Exploitation Tradeoff in Self-Play

### 5.1 The Opening Diversity Problem

A persistent challenge in self-play systems is **mode collapse in openings**: the network converges on a small set of opening sequences and never explores alternatives. This produces a strong but brittle player that can be exploited by novel openings.

This is especially acute in InfHexTTT because:
- The infinite board means the first move can be anywhere (or equivalently, we normalize so the first move is always at the origin, but the second player's response creates the opening structure).
- With 2 stones per turn, the combinatorial space of the first few moves is enormous.

### 5.2 Dirichlet Noise

Following AlphaZero, we add Dirichlet noise to the root node of MCTS:

P(a) = (1 - ε) * p_θ(a) + ε * Dir(α)

Where:
- ε = 0.25 (fraction of noise)
- α = Dirichlet concentration parameter

**Choosing α**: AlphaZero used α = 0.03 for Go (361 moves), α = 0.3 for chess (roughly: α ≈ 10/N where N is the typical number of legal moves). For InfHexTTT with decomposed half-moves on a ~750-cell board, the number of legal single-stone placements is ~700. Following the heuristic:

α ≈ 10 / N_legal ≈ 10 / 700 ≈ 0.015

However, with a good policy network, most probability mass is on ~50-100 moves. Using the effective number of plausible moves:

α ≈ 10 / 100 ≈ 0.10

**Recommendation**: Start with α = 0.08, tune based on opening diversity metrics. If openings are too narrow, increase α; if play quality degrades, decrease it.

### 5.3 Temperature Annealing

(See Section 4.4 for the schedule.) The key insight is that temperature and Dirichlet noise serve complementary roles:

- **Dirichlet noise**: Ensures the MCTS search explores unexpected moves at the root, potentially discovering new lines.
- **Temperature**: Ensures the final move selection retains some stochasticity, producing diverse game records for training.

Both are necessary. Dirichlet noise alone with greedy selection (τ = 0) can still produce repetitive play because the MCTS search will often "recover" from the noise and converge on the same best move. Temperature ensures the second- and third-best moves are sometimes selected.

### 5.4 Forced Opening Diversification

Beyond noise and temperature, we can explicitly force opening diversity:

**Method 1: Random first N moves**. For the first K half-moves, select moves uniformly at random from the top M policy network suggestions. E.g., K=4 (first 2 full turns), M=10.

- Pros: Guaranteed diversity.
- Cons: Early random moves may be very bad, producing low-quality training data for opening positions.
- Mitigation: Only use this for a fraction of self-play games (e.g., 20%).

**Method 2: Opening book seeding**. Maintain a list of known interesting openings and seed some fraction of self-play games with these. Update the list as training progresses.

- Pros: Ensures specific strategically interesting regions are explored.
- Cons: Introduces human bias; may not scale.
- Recommendation: Skip this for a pure self-play approach, but useful for debugging.

**Method 3: Symmetry breaking**. Since the game has D6 symmetry, we can normalize the first move to the origin and force the second player's first pair of stones into a specific canonical region (e.g., one of the 12 symmetry-equivalent fundamental domains). This doesn't diversify openings per se, but ensures we cover all structurally distinct responses.

### 5.5 Playout Cap Randomization (KataGo)

KataGo introduced **playout cap randomization** (Wu, 2019) as a powerful alternative to temperature scheduling:

- For each move, randomly choose between a "full" search (N simulations) and a "reduced" search (N/8 simulations).
- Full-search moves are used for training the policy head (high-quality targets).
- Both full and reduced moves are used for training the value head (since the value target is the game outcome, not the search policy).
- The reduced-search moves introduce natural stochasticity (the policy is noisier with fewer simulations), replacing the need for high temperature.

**Benefits**:
1. Training data has both high-quality policy targets (full search) and diverse positions (reduced search).
2. No need to tune temperature schedules.
3. The reduced-search moves effectively explore more of the game tree at lower computational cost.

**Recommendation for InfHexTTT**: Implement playout cap randomization with:
- Full search: 600 simulations.
- Reduced search: 75 simulations (1/8 of full).
- Probability of full search: 0.25 (i.e., 25% of moves use full search).
- Only full-search moves contribute to policy loss; all moves contribute to value loss.

This is one of KataGo's most impactful innovations and should be adopted from the start.

---

## 6. Evaluation and Progress Tracking

### 6.1 Elo Estimation from Self-Play

**Method**: Maintain a set of checkpoint networks {θ_1, θ_2, ..., θ_N}. Periodically play round-robin tournaments between recent checkpoints. Estimate Elo ratings using BayesElo (Coulom, 2008) or the simpler Elo system.

**Practical details**:
- Keep the last ~20 checkpoints for Elo estimation.
- Play 50-100 games per pair (25-50 as each color).
- Use low temperature (τ = 0.1) and no Dirichlet noise for evaluation.
- Elo ratings are relative; anchor the first checkpoint at Elo 0.

**Expected Elo curve**: Rapid initial improvement (100-200 Elo/day), slowing to ~10-50 Elo/day after 1-2 weeks, with eventual plateaus that are broken by architectural improvements or hyperparameter changes.

**Interpreting Elo**:
- 100 Elo difference: ~64% win rate for the stronger player.
- 200 Elo: ~76%.
- 400 Elo: ~91%.
- 800 Elo: ~99%.

### 6.2 Benchmark Positions and Tactical Puzzles

Create a suite of hand-crafted positions that test specific capabilities:

**Category 1: Tactical puzzles** (win-in-N-moves):
- Forced win with a double threat (1-move puzzle).
- Forced win requiring a sequence of threats (3-5 move puzzle).
- Defensive puzzles: find the only move that prevents immediate loss.

**Category 2: Strategic positions**:
- Positions where one player has a clear strategic advantage (e.g., a well-placed central formation) but no immediate tactical threats.
- Positions where the correct play is to expand influence rather than extend a line.

**Category 3: Endgame positions**:
- Nearly won positions where precise play is needed.
- Positions that are theoretically drawn but require accurate defense.

**Evaluation metric**: For each puzzle, measure:
- Whether the top-1 move matches the known solution.
- Whether the known solution is in the top-3 moves.
- The value head's assessment of the position (should match the known outcome).

**Puzzle creation**: Initially create 50-100 puzzles manually. As training progresses, automatically extract interesting positions from self-play games (positions where the MCTS value swung dramatically, indicating critical moments).

### 6.3 Fixed Opponent Benchmarks

Maintain a ladder of fixed opponents:

1. **Random player**: Uniform random move selection. Any trained network should achieve >99% win rate quickly.
2. **Greedy heuristic**: A hand-coded player that scores moves based on line lengths, threats, and blocking. Target: >95% win rate within 1-2 days.
3. **1-ply search + heuristic**: Greedy player with 1-ply lookahead. Target: >90% win rate within 3-5 days.
4. **Weak MCTS (100 simulations, small network)**: An early checkpoint or a deliberately weakened MCTS player. Target: >80% win rate within 1 week.
5. **Previous checkpoints**: The gating mechanism described in Section 3.4.

**Playing conditions**: 100 games per opponent (50 as each color), τ = 0.1, same MCTS budget as training.

### 6.4 Detecting Training Collapse and Catastrophic Forgetting

**Training collapse** manifests as:
- Elo plateau or decrease over extended periods (>2 days without improvement).
- Value head predicting near-constant values (e.g., always ~0) for diverse positions.
- Policy head becoming overly peaked (entropy collapse): the policy assigns >90% probability to a single move in most positions.
- Loss on held-out validation data diverging from training loss.

**Detection metrics** (monitor continuously):
- Policy entropy (averaged over recent games). Expected range: 1.5-4.0 nats. Below 1.0 indicates collapse.
- Value head prediction distribution. Should have meaningful spread across [-1, +1].
- Win rate against previous checkpoints. Should remain >50%.
- KL divergence between current policy and policy from N steps ago. Sudden spikes indicate instability.
- Game length distribution. If games become very short or very long suddenly, something is wrong.

**Catastrophic forgetting**: The network loses previously learned skills, often when transitioning from one training phase to another or when the self-play distribution shifts.

**Mitigation strategies**:
1. **Replay buffer diversity**: Maintain a fraction (~10%) of "museum" positions from earlier training stages.
2. **Periodic re-evaluation against old opponents**: If win rate against a previously beaten opponent drops, increase the learning rate decay or reduce the training data freshness weight.
3. **Elastic weight consolidation** (Kirkpatrick et al., 2017): Penalize changes to weights that are important for previously learned tasks. Often not necessary with a well-designed replay buffer.
4. **Checkpoint ensemble**: If a new checkpoint is worse than the old one on the gating match, keep the old one (this is the default AlphaZero behavior).

---

## 7. Practical Engineering Considerations

### 7.1 Framework Choices

**Training framework**:

| Framework | Pros | Cons | Recommendation |
|---|---|---|---|
| PyTorch | Largest ecosystem, easy debugging, dynamic graphs | Slightly slower than JAX for pure throughput | **Recommended for initial development** |
| JAX/Flax | JIT compilation, vmap for batched inference, XLA backend | Steeper learning curve, smaller community for RL | Consider for production/scaled training |
| TensorFlow | Mature, good deployment tools | Declining community momentum for research | Not recommended |

**MCTS implementation**:

| Language | Pros | Cons | Recommendation |
|---|---|---|---|
| Python | Easy prototyping, integrates with training | 10-100x slower than compiled languages | Prototype only |
| C++ | Fast, mature (used by Leela Chess Zero, KataGo) | Complex build systems, harder debugging | **Recommended for production** |
| Rust | Fast, memory-safe, modern tooling | Smaller ML ecosystem | Strong alternative to C++ |
| Cython/Numba | Middle ground: Python-like with compiled speed | Can be finicky; not as fast as C++/Rust | Viable for intermediate stage |

**Recommended stack**:
- **Phase 1 (prototyping)**: Python + PyTorch for everything. MCTS in Python with NumPy vectorization. Slow but fast to iterate.
- **Phase 2 (production)**: PyTorch for training, C++ or Rust for MCTS with Python bindings (pybind11 or PyO3). Neural network inference via TorchScript or ONNX Runtime.

### 7.2 Data Pipeline

**Self-play game serialization**:

Games should be stored in a compact binary format. Recommended schema:

```
Game {
  metadata: {
    game_id: uint64,
    checkpoint_id: uint32,
    timestamp: uint64,
    result: int8 (-1, 0, +1),
    num_moves: uint16,
    board_size: uint16 (for the cropped window)
  },
  moves: [Move] {
    position: (int16, int16)  // axial hex coordinates
    mcts_visits: [(position, visit_count)] // top-K moves
    mcts_value: float32
    mcts_policy: [(position, probability)] // top-K
  },
  // Auxiliary targets computed post-hoc:
  ownership_map: [uint8]  // per-cell, at game end
}
```

**Storage format options**:
- **Protocol Buffers**: Good compression, schema evolution, cross-language support. Used by KataGo.
- **FlatBuffers**: Zero-copy deserialization, good for MCTS data.
- **NumPy .npz**: Simple, good for prototyping.
- **SQLite**: Queryable, good for analysis.

**Recommendation**: Use Protocol Buffers for the game format, backed by an SQLite index for querying games by checkpoint, result, length, etc. For the replay buffer in memory, use a ring buffer of NumPy arrays.

**Storage estimates**:
- Per game: ~5-20 KB (compressed).
- 100,000 games: ~0.5-2 GB.
- 1,000,000 games: ~5-20 GB.
- Manageable on any modern machine.

### 7.3 Distributed Training Considerations

Even at small scale (1-4 GPUs), some distribution is beneficial:

**Single-machine, multi-GPU**:
- GPU 0: Training (data-parallel if 2+ GPUs available for training).
- GPU 1: Self-play inference (batch neural network evaluations for MCTS).
- CPU: MCTS tree search (embarrassingly parallel across games).

**Communication pattern**:
- Self-play workers send completed games to a shared replay buffer (shared memory or a simple queue).
- Training loop pulls minibatches from the replay buffer.
- After each training epoch, the updated weights are copied to the self-play inference GPU.

**Multi-machine** (for scaling):
- Use Ray (Moritz et al., 2018) or a simple TCP-based protocol.
- Self-play workers on separate machines download weights from a central weight server.
- Games are uploaded to a central replay buffer (Redis, or a simple file server).

**For InfHexTTT specifically**: A single machine with 1-2 GPUs is sufficient for meaningful training. The bottleneck is typically MCTS speed, not training throughput. Prioritize optimizing MCTS implementation over distributed training.

### 7.4 Debugging Self-Play Systems

Self-play RL systems are notoriously difficult to debug. Common failure modes and diagnosis:

**Failure mode 1: Network produces uniform policy**
- Symptom: Policy entropy stays high, Elo doesn't improve.
- Cause: Learning rate too high, or policy loss weight too low, or bugs in the feature encoding.
- Diagnosis: Visualize policy output for simple positions. A well-trained network should strongly prefer moves near existing stones.
- Fix: Reduce learning rate, increase policy loss weight, verify feature encoding.

**Failure mode 2: Value head always predicts 0**
- Symptom: Value loss plateaus at ~0.5 (MSE with binary targets).
- Cause: Value head has insufficient capacity, or training data is too diverse (equal wins and losses).
- Diagnosis: Check value predictions on positions where one player has an overwhelming advantage.
- Fix: Increase value head capacity, or check that the outcome label z is correctly computed from the correct player's perspective.

**Failure mode 3: Games are too short (random-looking play)**
- Symptom: Average game length is <20 moves, many 1-move wins.
- Cause: MCTS not running enough simulations, or neural network not loaded correctly.
- Diagnosis: Inspect actual game records; check that MCTS is using the neural network (not just random rollouts).
- Fix: Verify the inference pipeline, increase simulations.

**Failure mode 4: Games are too long (passive play)**
- Symptom: Games last hundreds of moves, neither player creates threats.
- Cause: Network learned to avoid risk, possibly due to a draw-heavy training signal.
- Diagnosis: Check if the value head predicts ~0 for most positions. Check if the policy avoids aggressive moves.
- Fix: Increase exploration (Dirichlet noise, temperature), add auxiliary threat detection targets, check if the reward signal is correct.

**Failure mode 5: Cyclical Elo (improving then degrading)**
- Symptom: Elo oscillates with a period of days.
- Cause: The network learns to exploit the previous version's weakness, but this strategy is itself exploitable. This is a form of non-transitive strategy cycling (Balduzzi et al., 2019).
- Diagnosis: Check if the network loses to checkpoints from 2-3 versions ago.
- Fix: Increase replay buffer diversity, add older checkpoints to the evaluation pool, use population-based training (Jaderberg et al., 2017).

**General debugging strategy**:
1. Start with the simplest possible variant (4-in-a-row, small board) and verify the pipeline works end-to-end.
2. Add complexity incrementally.
3. Log extensively: every game, every MCTS search, every training loss.
4. Visualize early and often: board states, policy heatmaps, value estimates.

### 7.5 Open-Source References

| Project | Language | Relevance | URL |
|---|---|---|---|
| KataGo | C++/Python | State-of-the-art Go AI; best reference for training innovations | github.com/lightvector/KataGo |
| Leela Chess Zero (lc0) | C++/Python | Large-scale distributed self-play for chess | github.com/LeelaChessZero/lc0 |
| OpenSpiel | C++/Python | Google's framework for game-playing AI research; includes MCTS and AlphaZero implementations | github.com/google-deepmind/open_spiel |
| Polygames | C++/Python | Facebook's framework; supports hexagonal games and Connect6-style games | github.com/facebookarchive/Polygames |
| minigo | Python/TF | Minimal AlphaGo Zero implementation; good for learning | github.com/tensorflow/minigo |
| muzero-general | Python | General MuZero implementation; useful reference for training loops | github.com/werner-duvaud/muzero-general |
| alpha-zero-general | Python | Simple, educational AlphaZero for arbitrary games | github.com/suragnair/alpha-zero-general |

**Most relevant for InfHexTTT**:
1. **Polygames**: Already supports hexagonal board games and Connect6-like games. The most directly applicable codebase.
2. **KataGo**: The gold standard for training innovations (auxiliary targets, playout cap randomization, multi-scale training). Worth studying in detail even though it targets Go.
3. **OpenSpiel**: Provides a clean abstraction for game environments that we can implement InfHexTTT against.

---

## Recommended Training Pipeline for Infinite Hex Tic-Tac-Toe

### Phase 1: Initial Architecture and Small-Scale Experiments (Weeks 1-3)

**Goals**: Validate the pipeline end-to-end, establish baselines, identify bugs.

**Architecture**:
- Network: ResNet with 8 residual blocks, 128 filters per layer.
- Input features: 5 planes (Black stones, White stones, player-to-move indicator, move-number-within-turn indicator [first or second stone of the 2-stone turn], all-ones plane for board boundary).
- Board representation: 19x19 hex grid (axial coordinates), zero-padded.
- Policy head: Single-stone placement (decomposed 2-stone moves). Output: 19x19 = 361 probabilities.
- Value head: Single scalar in [-1, +1].
- Auxiliary heads: Ownership (19x19x3), threat detection (19x19x2).

**Training configuration**:
- MCTS simulations per move: 200 (reduced for speed during prototyping).
- Self-play games per iteration: 500.
- Training steps per iteration: 500 (batch size 256).
- Replay buffer: 5,000 games (sliding window).
- Learning rate: 0.02, cosine annealing to 0.0002 over 100 iterations.
- Weight decay: 1e-4.
- Temperature: τ = 1.0 for first 10 half-moves, then 0.3.
- Dirichlet noise: α = 0.10, ε = 0.25.
- Optimizer: SGD with momentum 0.9 (following AlphaZero; Adam is viable but SGD is more stable for self-play).

**Milestones**:
- Week 1: Pipeline running end-to-end. Self-play generates games, training reduces loss, evaluation runs.
- Week 2: Network beats random player >99%. Network beats greedy heuristic >80%.
- Week 3: Network achieves ~500-800 Elo (relative). Visualize learned policy and value for interpretability.

**Hardware**: 1x RTX 4090 or equivalent. ~8-12 hours per training iteration (500 games + 500 training steps). Total: ~50-100 GPU-hours for Phase 1.

**Key experiments**:
1. Verify that decomposed 2-stone moves work correctly in MCTS.
2. Compare with and without auxiliary targets (ownership, threats). Expect 20-50% faster Elo improvement with auxiliary targets, consistent with KataGo findings.
3. Test curriculum: start with 4-in-a-row on 11x11, then 5-in-a-row on 15x15, then 6-in-a-row on 19x19. Compare against training directly on 6-in-a-row.
4. Measure wall-clock time per game, positions per second, and identify bottlenecks.

### Phase 2: Full-Scale Self-Play Training (Weeks 4-10)

**Goals**: Train a strong player on the full 6-in-a-row game. Achieve superhuman performance relative to any hand-coded heuristic.

**Architecture** (scaled up):
- Network: 15 residual blocks, 192 filters.
- Board representation: 31x31 hex grid (dynamically re-centered).
- Policy head: 31x31 = 961 output positions per half-move.
- All auxiliary heads active.

**Training configuration**:
- MCTS simulations per move: 600 (full search) / 75 (reduced search, playout cap randomization).
- Playout cap randomization: 25% full, 75% reduced.
- Self-play: Continuous asynchronous generation. Target: 200+ games/hour.
- Training: Continuous, 1 minibatch update per ~10 new positions added to buffer.
- Replay buffer: 25,000 games (~5 million positions), sliding window.
- Learning rate schedule: 0.01 for first 50K steps, 0.001 for next 100K, 0.0001 thereafter.
- Weight decay: 1e-4.
- Temperature: τ = 1.0 for first 20 half-moves, τ = 0.5 for half-moves 21-40, τ = 0.1 thereafter.
- Dirichlet noise: α = 0.08, ε = 0.25.
- Gating: New checkpoint replaces best if >55% win rate over 400 evaluation games.
- Checkpoint interval: Every 5,000 training steps.

**Multi-scale training** (KataGo-style):
- 30% of games on 19x19.
- 40% of games on 25x25.
- 30% of games on 31x31.
- This leverages the fully convolutional architecture and improves generalization.

**Milestones**:
- Week 4-5: Transition to async pipeline. Validate throughput targets.
- Week 6: ~1500 Elo. Defeats 1-ply search heuristic >95%.
- Week 8: ~2000 Elo. Opens with diverse, coherent strategies. Value head accurately predicts game outcomes from mid-game positions (>70% accuracy).
- Week 10: ~2500 Elo. Strong tactical play. Recognizes and creates complex threat sequences.

**Hardware**: Ideally 2x A100 (1 for training, 1 for self-play inference) + 16-32 CPU cores for MCTS. Estimated cost at cloud rates (~$3/hr per A100): ~$2,000-4,000 for Phase 2.

**With 1x RTX 4090**: Feasible but ~3x slower. Extend timeline to weeks 4-16.

### Phase 3: Refinement and Optimization (Weeks 11-16)

**Goals**: Push performance further, analyze learned strategies, prepare for deployment.

**Architecture** (final):
- Network: 20 residual blocks, 256 filters. Or transformer-based architecture (see below).
- Board: 31x31 hex grid with dynamic re-centering and learned positional encoding.

**Refinement strategies**:

1. **Network architecture search**: Test SE-ResNet (Squeeze-and-Excitation, Hu et al., 2018, used in Leela Chess Zero), bottleneck residual blocks, and small vision transformers. The hex grid's non-Euclidean geometry may benefit from attention-based architectures.

2. **MCTS improvements**:
   - **Progressive widening**: Limit the number of children expanded at each node, proportional to visit count. Critical for high-branching-factor games (Coulom, 2007).
   - **Virtual loss**: For parallel MCTS (multiple threads), add virtual losses to in-flight nodes to encourage exploration of different branches. Standard technique, used in all major implementations.
   - **First Play Urgency (FPU)**: Set the value estimate for unexplored children to a pessimistic value (e.g., parent value - 0.2). Reduces wasted simulations on unlikely moves. Used in KataGo and Leela Chess Zero.

3. **Analysis of learned strategies**:
   - Extract the network's preferred openings and compare across training stages.
   - Identify common patterns in won/lost games.
   - Create an opening book from the strongest checkpoint's preferred moves.
   - Analyze whether the network has discovered any novel tactical or strategic concepts.

4. **Robustness testing**:
   - Play against adversarial opponents designed to exploit potential weaknesses (e.g., an opponent that always plays in distant regions to test the network's locality reasoning).
   - Test on board sizes not seen during training (e.g., 37x37).
   - Measure performance degradation under reduced MCTS budget (e.g., 100 simulations instead of 600).

**Milestones**:
- Week 12: ~2800-3000 Elo. Network discovers non-obvious strategic patterns.
- Week 14: Publish analysis of learned strategies and game-theoretic insights.
- Week 16: Final model. Documentation complete.

### Specific Hyperparameter Recommendations (Summary Table)

| Hyperparameter | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Residual blocks | 8 | 15 | 20 |
| Filters per layer | 128 | 192 | 256 |
| Board size (training) | 19x19 | 19-31x31 (multi-scale) | 31x31 |
| MCTS sims (full) | 200 | 600 | 800-1200 |
| MCTS sims (reduced) | -- | 75 | 100-150 |
| Playout cap randomization | No | Yes (25% full) | Yes (25% full) |
| Batch size | 256 | 256-512 | 512-1024 |
| Learning rate (initial) | 0.02 | 0.01 | 0.005 |
| Learning rate (final) | 0.0002 | 0.0001 | 0.00005 |
| LR schedule | Cosine | Step decay | Cosine with warm restarts |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | SGD (momentum 0.9) | SGD (momentum 0.9) | SGD or LAMB |
| Replay buffer (games) | 5,000 | 25,000 | 50,000 |
| Temperature (early moves) | 1.0 (10 half-moves) | 1.0 (20 half-moves) | 1.0 (20 half-moves) |
| Temperature (late moves) | 0.3 | 0.1 | 0.1 |
| Dirichlet α | 0.10 | 0.08 | 0.06 |
| Dirichlet ε | 0.25 | 0.25 | 0.25 |
| Gating threshold | 55% | 55% | 55% |
| Gating games | 200 | 400 | 400 |
| cPUCT (MCTS exploration) | 2.5 | 2.0 | 1.5-2.0 |
| FPU reduction | 0.0 | -0.2 | -0.2 |
| Auxiliary loss weight (total) | 0.20 | 0.30 | 0.30 |

### Hardware Requirements and Timeline Estimates

| Configuration | Phase 1 | Phase 2 | Phase 3 | Total |
|---|---|---|---|---|
| 1x RTX 4090 | 3 weeks | 12 weeks | 6 weeks | ~21 weeks |
| 2x A100 | 2 weeks | 6 weeks | 4 weeks | ~12 weeks |
| 4x A100 | 1.5 weeks | 4 weeks | 3 weeks | ~8.5 weeks |
| 8x A100 (cluster) | 1 week | 2.5 weeks | 2 weeks | ~5.5 weeks |

**Cloud cost estimates** (at ~$3/hr per A100):
- 2x A100 for 12 weeks: ~$12,000.
- 4x A100 for 8.5 weeks: ~$17,000.
- 1x RTX 4090 (owned): electricity only, ~$50-100 total.

**Recommendation**: Start with a single owned GPU (RTX 4090 or equivalent) for Phase 1. Use cloud instances for Phase 2-3 if timeline is important. A single RTX 4090 can achieve meaningful results in ~5 months; this is the most cost-effective path for a research project.

### Key Risks and Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MCTS too slow due to high branching factor | High | Training bottleneck | Decompose 2-stone moves; implement progressive widening; optimize MCTS in C++/Rust |
| Opening mode collapse | Medium | Weak, brittle player | Dirichlet noise, temperature, playout cap randomization, forced opening diversity |
| Value head fails to learn (always predicts ~0) | Medium | No training progress | Auxiliary targets (ownership, threats); verify data pipeline; curriculum learning from simpler variants |
| Infinite board representation issues | Medium | Board boundary artifacts | Dynamic re-centering; large enough window (31x31); learned positional encoding |
| Game is "too drawn" at high skill levels | Low-Medium | Training signal degrades | Verify with long self-play tournaments; consider adding swap rule for balance; shaped auxiliary rewards |
| Training instability (loss spikes, Elo collapse) | Medium | Wasted compute | Frequent checkpointing; conservative learning rate; gating mechanism prevents deploying worse models |
| Non-transitive strategy cycles | Low-Medium | Elo stagnates despite learning | Population-based training; diverse evaluation opponents; larger replay buffer |
| Hardware failure / data loss | Low | Days-weeks of lost progress | Automated checkpointing to cloud storage every 2 hours; replay buffer snapshots daily |
| Hex grid convolution implementation bugs | Medium | Incorrect feature extraction | Use established hex convolution libraries (e.g., from Polygames); extensive unit testing on known positions |

### Critical Path Items

The following items are on the critical path and should be prioritized:

1. **Hex grid convolution implementation**: Standard 2D convolutions on a square grid do not correctly capture hexagonal adjacency. Options:
   - Offset coordinates with standard convolutions (simple but introduces artifacts at grid boundaries).
   - Axial coordinates with masked convolutions (cleaner).
   - Graph neural network approach (most general but slower).
   - **Recommendation**: Axial coordinate representation with standard 3x3 convolutions, where the 6 hex neighbors map to 6 of the 8 positions in a 3x3 kernel, and the remaining 2 kernel positions are masked to zero. This is efficient and correct. See Polygames (Cazenave et al., 2020) for reference implementation.

2. **Decomposed 2-stone MCTS**: The MCTS tree must correctly handle the fact that each "turn" consists of 2 sequential stone placements by the same player. The tree has alternating layers: Player-A-stone-1, Player-A-stone-2, Player-B-stone-1, Player-B-stone-2, etc. The value backup must correctly assign the outcome to the player whose turn it is (both stones of a 2-stone turn share the same "player to move"). This is a common source of bugs.

3. **Dynamic board re-centering**: Since the board is infinite, the neural network input must be a finite window. This window must be re-centered based on the current extent of play, with sufficient margin for future moves. If play extends beyond the window, the window must grow or shift. A 31x31 hex window (~750 cells) should be sufficient for nearly all games, but edge cases must be handled.

---

## References

- Allis, L. V., van der Meulen, M., & van den Herik, H. J. (1994). Go-Moku and Threat-Space Search. University of Limburg.
- Balduzzi, D., Garnelo, M., Bachrach, Y., Czarnecki, W. M., Perolat, J., Jaderberg, M., & Graepel, T. (2019). Open-ended learning in symmetric zero-sum games. *ICML 2019*.
- Beck, J. (2008). *Combinatorial Games: Tic-Tac-Toe Theory*. Cambridge University Press.
- Cazenave, T., Chen, Y.-C., Chen, G.-W., Chen, S.-Y., Chiu, X.-D., Dehos, J., ... & Wu, I.-C. (2020). Polygames: Improved zero learning. *ICGA Journal*, 42(4), 244-256.
- Coulom, R. (2007). Efficient selectivity and backup operators in Monte-Carlo tree search. *CG 2006*, LNCS 4630, 72-83.
- Coulom, R. (2008). Whole-history rating: A Bayesian rating system for players of time-varying strength. *CG 2008*, LNCS 5131, 113-124.
- Hales, A. W., & Jewett, R. I. (1963). Regularity and positional games. *Transactions of the AMS*, 106(2), 222-229.
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR 2018*.
- Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Kavukcuoglu, K. (2017). Population based training of neural networks. *arXiv preprint arXiv:1711.09846*.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Vinyals, O., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
- Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., ... & Stoica, I. (2018). Ray: A distributed framework for emerging AI applications. *OSDI 2018*.
- Nash, J. (1952). Some games and machines for playing them. *RAND Corporation Report D-1164*.
- Pascutto, G.-C., & Linscott, T. (2019). Leela Chess Zero. https://lczero.org.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.
- Wu, I.-C. (2001). *An Application-Independent Automatic Game-Playing Program*. Ph.D. thesis, National Chiao Tung University.
- Wu, I.-C., & Huang, D.-Y. (2005). A new family of k-in-a-row games. *ACG 2005*, 180-194.
- Wu, I.-C., & Huang, D.-Y. (2006). A new family of k-in-a-row games. *ICGA Journal*, 29(1), 26-34.
- Wu, D. J. (2019). Accelerating self-play learning in Go. *arXiv preprint arXiv:1902.10565*.

---

*This document is part of the Hex-TTT Neural Network research series. For network architecture details, see Document 03. For game environment implementation, see Document 02.*

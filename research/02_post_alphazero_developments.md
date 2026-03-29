# Post-AlphaZero Developments in Game-Playing AI (2019--2025)

**Research Survey for the Infinite Hexagonal Tic-Tac-Toe Project**

---

## 1. MuZero: Planning Without a Perfect Simulator

### 1.1 Motivation and Core Idea

AlphaZero (Silver et al., 2018) achieved superhuman performance in Go, chess, and shogi but required a *perfect simulator* of the environment---i.e., the exact rules of the game had to be programmed in advance so that Monte Carlo Tree Search (MCTS) could forward-simulate states during planning. MuZero (Schrittwieser et al., 2020) eliminates this requirement by *learning* an internal model of the environment and planning entirely within a learned latent space. The key insight is that a useful model need not reconstruct pixel-perfect observations; it only needs to predict the quantities relevant for planning: **rewards, values, and policies**.

### 1.2 Architecture: Three Learned Functions

MuZero factorizes the environment model into three neural networks, all trained end-to-end:

1. **Representation function** $h_\theta$: Maps a sequence of past observations $o_1, \ldots, o_t$ to an initial hidden state $s^0 = h_\theta(o_1, \ldots, o_t)$. This is the "encoder" that grounds the latent space in actual observations. In practice, the observation stack typically includes the most recent 32 frames (Atari) or the current board position plus history (board games).

2. **Dynamics function** $g_\theta$: Given a hidden state $s^k$ and an action $a^{k+1}$, predicts the next hidden state and the immediate reward: $(r^{k+1}, s^{k+1}) = g_\theta(s^k, a^{k+1})$. This is the learned "transition model." Critically, $s^{k+1}$ lives in latent space and has no obligation to correspond to a decodable observation---it only needs to support accurate downstream predictions.

3. **Prediction function** $f_\theta$: Given any hidden state $s^k$, outputs a policy $\mathbf{p}^k$ and a value estimate $v^k$: $(\mathbf{p}^k, v^k) = f_\theta(s^k)$. This is architecturally identical to AlphaZero's policy-value head.

### 1.3 Planning in Latent Space

At decision time, MuZero runs MCTS entirely within the latent space:

1. Encode the current observation: $s^0 = h_\theta(o_t)$.
2. At each tree node with state $s^k$, expand by selecting an action $a$ via PUCT (as in AlphaZero), then compute $(r^{k+1}, s^{k+1}) = g_\theta(s^k, a)$ and $(\mathbf{p}^{k+1}, v^{k+1}) = f_\theta(s^{k+1})$.
3. Back up values through the tree incorporating the predicted rewards.
4. After a fixed number of simulations (e.g., 800 for board games, 50 for Atari), select the action with the highest visit count at the root.

No environment simulator is ever called during search. The entire tree is "imagined" by the learned model.

### 1.4 Training: Unrolled MCTS

Training proceeds as follows:

- **Data collection**: The agent plays episodes using MCTS-guided action selection, storing $(o_t, a_t, r_t, \ldots)$ trajectories in a prioritized replay buffer.
- **Unrolled loss**: For each sampled position $t$ in a trajectory, MuZero unrolls the dynamics model for $K$ hypothetical steps (typically $K = 5$). At each unrolled step $k = 0, \ldots, K$, three losses are computed:
  - **Policy loss**: Cross-entropy between the predicted policy $\mathbf{p}^k$ and the MCTS visit-count distribution $\boldsymbol{\pi}_{t+k}$ from the corresponding real search.
  - **Value loss**: Mean-squared error (or cross-entropy over a discretized support) between the predicted value $v^k$ and the $n$-step bootstrapped return.
  - **Reward loss**: Cross-entropy between the predicted reward $r^k$ and the observed reward $u_{t+k}$.
- All three networks are trained jointly by backpropagating through the unrolled computation graph (similar to training an RNN through time).

### 1.5 Key Results

| Domain | MuZero | AlphaZero | Notes |
|--------|--------|-----------|-------|
| Go | 5,005 Elo | 5,018 Elo | Within statistical noise |
| Chess | 3,430 Elo | 3,430 Elo | Matched exactly |
| Shogi | 4,654 Elo | 4,413 Elo | MuZero slightly stronger |
| Atari (57 games) | 731% median HNS | N/A (not applicable) | SOTA at time of publication |

MuZero matched or exceeded AlphaZero in all three board games *without access to the game rules during search*, and simultaneously achieved state-of-the-art results across 57 Atari games, demonstrating generality across fundamentally different domains (perfect-information board games vs. visually complex single-player environments).

### 1.6 Implications

The elimination of the perfect-simulator requirement opened model-based reinforcement learning to domains where writing a simulator is expensive or impossible. However, for well-defined board games where rules are known exactly, MuZero's learned model introduces approximation error without clear benefit over AlphaZero's exact forward model. The primary advantage in such settings is architectural unification: a single algorithm works across multiple domains.

---

## 2. EfficientZero: Sample-Efficient Model-Based RL

### 2.1 The Sample Efficiency Problem

MuZero, despite its generality, requires hundreds of millions of frames of interaction to achieve strong Atari performance---roughly 200 million frames, equivalent to ~38 days of continuous play at 60 fps. EfficientZero (Ye et al., 2021) targets the low-data regime, asking: can we achieve superhuman Atari performance with only 100,000 environment interactions (~2 hours of real-time gameplay)?

### 2.2 Three Key Innovations

EfficientZero builds on MuZero and introduces three modifications:

#### 2.2.1 Self-Supervised Temporal Consistency Loss

The learned dynamics model in MuZero can drift arbitrarily in latent space because no loss constrains the hidden states $s^k$ to remain consistent with actual future observations. EfficientZero adds a **self-supervised consistency loss** inspired by BYOL (Grill et al., 2020):

$$\mathcal{L}_{\text{consist}} = \| \text{proj}(s^{k+1}) - \text{sg}[\text{proj}(h_\theta(o_{t+k+1}))] \|_2^2$$

where $\text{sg}[\cdot]$ denotes stop-gradient and $\text{proj}(\cdot)$ is a learned projection head. This forces the dynamics model's predicted next state to be consistent with the state that would be obtained by directly encoding the actual next observation. A target (momentum-averaged) encoder is used to prevent collapse, following the BYOL paradigm.

#### 2.2.2 Value Prefix Prediction

In environments with sparse or delayed rewards, predicting single-step rewards is insufficient to learn good value functions from limited data. EfficientZero replaces single-step reward prediction with **value prefix prediction**: at each unrolled step $k$, the model predicts the cumulative $n$-step reward sum $\sum_{i=0}^{n-1} \gamma^i r_{t+k+i}$. This provides a denser learning signal because it aggregates reward information over multiple timesteps, helping the model learn long-range reward structure even with few samples.

#### 2.2.3 End-to-End Learned MCTS with Reanalysis

EfficientZero employs a more aggressive form of **reanalysis** (originally introduced in MuZero Reanalyze): periodically, past trajectories in the replay buffer are re-searched with the latest model parameters, generating fresher policy and value targets. This squeezes more learning from each collected sample. Combined with prioritized replay, reanalysis ensures that the training signal from limited data remains maximally informative as the model improves.

### 2.3 Results

On the Atari 100k benchmark (100,000 environment steps, ~26 games):

- **EfficientZero** achieved **190.4% mean human-normalized score** and **116.0% median HNS**, making it the first algorithm to surpass human-level performance in the 100k regime at the median.
- By comparison, the strongest prior methods (SPR, DrQ) achieved approximately 60--70% median HNS.
- EfficientZero outperformed human players on 22 of 26 tested games with only 2 hours of gameplay data.

### 2.4 Relevance

The techniques in EfficientZero are broadly applicable to any MuZero-style system where data is expensive. The self-supervised consistency loss, in particular, is a general-purpose regularizer for learned world models that prevents latent-space drift.

---

## 3. Sampled MuZero and Stochastic MuZero

### 3.1 Sampled MuZero: Scaling to Large and Continuous Action Spaces

Standard MCTS requires enumerating all legal actions at each node, which becomes intractable when the action space is large or continuous. **Sampled MuZero** (Hubert et al., 2021) modifies the MCTS procedure so that at each node, only a *sampled subset* of actions is considered for expansion:

- At each node, $K$ actions are sampled from the current policy network $\mathbf{p}^k = f_\theta(s^k)$.
- MCTS proceeds over only these sampled actions, using a modified PUCT formula that corrects for the sampling distribution.
- The completions guarantee of the original PUCT is relaxed in favor of practical scalability.

**Key theoretical result**: Hubert et al. (2021) prove that the modified search procedure is still a **policy improvement operator**---i.e., the action distribution produced by MCTS with sampled actions is at least as good as the prior policy, even though only a subset of actions is expanded.

Sampled MuZero was applied to challenging continuous-control tasks and demonstrated competitive performance with model-free methods while retaining the planning capabilities of MuZero.

### 3.2 Stochastic MuZero: Modeling Environment Stochasticity

A separate limitation of the original MuZero is that the dynamics function is deterministic: given $(s^k, a^{k+1})$, it produces a single $(r^{k+1}, s^{k+1})$. Many real environments are stochastic (e.g., card games with random draws, dice rolls, stochastic opponent behavior). **Stochastic MuZero** (Antonoglou et al., 2022) introduces an explicit stochastic variable into the dynamics model:

$$(r^{k+1}, s^{k+1}) = g_\theta(s^k, a^{k+1}, z^{k+1})$$

where $z^{k+1}$ is a discrete stochastic variable ("chance outcome") sampled from a learned prior $c_\theta(s^k, a^{k+1})$. During search, chance nodes are inserted into the MCTS tree, analogous to how expectimax handles stochastic transitions.

**Architecture details**:
- The chance outcome $z$ is quantized into a discrete codebook using a VQ-VAE-style encoder applied to the actual next observation.
- During training, an "afterstate" representation is used: the dynamics model first computes a deterministic afterstate $\bar{s}^{k+1} = g^{\text{det}}_\theta(s^k, a^{k+1})$, then the stochastic transition applies: $s^{k+1} = g^{\text{stoch}}_\theta(\bar{s}^{k+1}, z^{k+1})$.
- The prior $c_\theta(\bar{s}^{k+1})$ is trained to predict the distribution over chance outcomes.

**Key results**: Stochastic MuZero achieved strong performance in 2048 (a stochastic single-player game) and backgammon (stochastic two-player), domains where deterministic MuZero struggles because averaging over stochastic outcomes in a deterministic latent model produces blurred, uninformative predictions.

---

## 4. Gumbel AlphaZero and Gumbel MuZero

### 4.1 Motivation

Standard MCTS with PUCT, as used in AlphaZero and MuZero, has a known theoretical weakness: the guarantee that MCTS visit counts converge to the optimal policy requires the number of simulations to approach infinity. With a fixed, small simulation budget, PUCT can exhibit suboptimal behavior---for instance, it may waste simulations on clearly inferior actions due to the exploration bonus, or fail to sufficiently refine the value estimates of the top candidates.

Danihelka et al. (2022) proposed **Gumbel AlphaZero** and **Gumbel MuZero**, which replace the PUCT-based action selection in MCTS with a procedure based on the **Gumbel-Top-k trick**, providing a *policy improvement guarantee* even with a very small, fixed number of simulations.

### 4.2 The Gumbel-Top-k Trick

The core mechanism:

1. At the root node, sample independent Gumbel(0) noise $g_a$ for each action $a$.
2. Compute $\tilde{q}_a = g_a + \log \pi_a$ for each action, where $\pi_a$ is the prior policy probability from the neural network.
3. Select the top-$k$ actions by $\tilde{q}_a$ values (where $k$ is the simulation budget divided by the maximum tree depth, or simply the number of actions to consider).
4. Use the simulation budget to evaluate these $k$ actions by expanding them in the search tree.
5. After search, select the action that maximizes $g_a + \log \pi_a + \sigma(q_a)$, where $q_a$ is the improved value estimate from search and $\sigma$ is a monotonically increasing transformation.

**The key theoretical property**: This procedure is equivalent to sampling from an improved policy $\pi'$ that is *provably better* than $\pi$ in the policy improvement sense, regardless of the simulation budget. The Gumbel noise provides the right amount of stochastic exploration, and the sequential halving of candidates ensures efficient use of the simulation budget.

### 4.3 Sequential Halving

To efficiently allocate simulations among the top-$k$ candidate actions, Gumbel MuZero employs a **sequential halving** strategy:

- Start with $k$ candidate actions.
- Allocate a fraction of the simulation budget to evaluate all $k$ candidates.
- Eliminate the bottom half based on their $g_a + \log \pi_a + \sigma(\hat{q}_a)$ scores.
- Repeat: allocate more simulations to the remaining $k/2$ candidates, eliminate the bottom half, and so on.
- The final surviving action is selected.

This yields a logarithmic dependence of tree depth on the number of actions, making it far more efficient than standard PUCT for large action spaces.

### 4.4 Results

- In 9x9 Go, Gumbel MuZero with **8 simulations** matched the performance of standard MuZero with **64 simulations**---an 8x reduction in search compute.
- In 19x19 Go, Gumbel MuZero with 16 simulations achieved an Elo comparable to standard MuZero with ~200 simulations.
- The improvement is most pronounced at low simulation budgets, exactly the regime relevant for fast play or resource-constrained settings.
- At very high simulation counts (800+), the gap narrows as standard MCTS eventually converges.

### 4.5 Significance

Gumbel MuZero decouples the *quality* of search from the *quantity* of simulations far more effectively than PUCT. This is especially valuable for games with high branching factors, where the simulation budget per action at the root is inherently limited.

---

## 5. KataGo: Engineering Superhuman Go AI

### 5.1 Overview

KataGo (Wu, 2019) is an open-source Go AI that achieved superhuman strength while being dramatically more training-efficient than AlphaZero, Leela Zero, or ELF OpenGo. Rather than a single algorithmic breakthrough, KataGo represents a systematic accumulation of engineering and architectural improvements. Wu (2019) reports that KataGo reached the playing strength of ELF OpenGo's final release (which trained on 2,000 TPU-v3s for two weeks) using only **35 V100 GPUs for ~19 days**---roughly 50x less compute.

### 5.2 Auxiliary Training Targets

One of KataGo's most impactful innovations is the use of **auxiliary prediction targets** beyond the standard policy and value heads. These include:

1. **Ownership prediction**: For each intersection on the board, predict the probability that it will be owned by Black, White, or neither at the end of the game. This is a dense per-cell prediction that provides a rich gradient signal. The ownership head encourages the network to develop a spatial understanding of territory and influence.

2. **Score prediction**: Predict the final score difference (margin of victory). While the value head only predicts win/loss (a binary outcome), the score head predicts the *magnitude*, which provides useful gradient signal in positions that are clearly won or lost (where the value head saturates).

3. **Score distribution prediction**: Predict the full distribution over possible final score differences, discretized into bins. This captures uncertainty in the game outcome and is especially useful for komi (handicap) sensitivity.

4. **Future position prediction**: Predict aspects of board states several moves into the future, encouraging the network to internalize forward-looking features.

### 5.3 Architecture Innovations

KataGo uses a **modified ResNet** backbone with several refinements:

- **Global pooling layers**: In addition to the standard convolutional residual blocks, KataGo includes a branch that applies global average pooling and global max pooling, feeds the result through a small fully connected layer, and broadcasts the output back to every spatial position via a bias. This allows the network to incorporate global board features (e.g., total stone count, liberties distribution) without relying solely on the receptive field of the convolutional stack.

- **Nested bottleneck residual blocks**: The residual blocks use a bottleneck structure (1x1 conv -> 3x3 conv -> 1x1 conv) that reduces the computational cost per block while maintaining representational capacity.

- **Input feature engineering**: KataGo uses a rich set of hand-crafted input features including ladder detection, liberty counts, and pass-alive territory, which accelerate early training by providing the network with features that would otherwise take many training steps to learn.

### 5.4 Training Efficiency Improvements

Key factors contributing to KataGo's training efficiency:

- **Playout cap randomization**: During self-play, the number of MCTS simulations per move is randomized (from a low minimum to the full budget), creating a curriculum that starts with fast, noisy games and progressively refines. Moves played with few simulations still contribute useful policy targets because the network has to learn to play well even with limited search.

- **Dynamic variance-scaled cPUCT**: The exploration constant in PUCT is dynamically scaled based on the variance of child values, allowing the search to explore more in uncertain positions and exploit more in clear positions.

- **Aggressive reuse of self-play data**: KataGo reuses each self-play game for multiple training updates (high replay ratio), combined with careful management of the replay buffer to avoid stale data.

- **Forced playouts and policy target pruning**: Modifications to the MCTS search to reduce the noise in policy targets and improve the signal-to-noise ratio of training data.

### 5.5 Ongoing Development and Impact

As of 2025, KataGo continues to be actively developed and is the strongest freely available Go engine. It has been instrumental in:
- Professional Go analysis and commentary.
- Research into adversarial vulnerabilities of Go AI (Wang et al., 2023; Wu et al., 2023 showed adversarial policies that exploit KataGo, leading to defensive training improvements).
- Serving as a baseline for new algorithmic research in game-playing AI.

---

## 6. Other Notable Developments

### 6.1 Polygames (Facebook AI Research)

**Polygames** (Cazenave et al., 2020) is a general game-playing framework from Facebook AI Research that combines AlphaZero-style self-play with support for a diverse set of board games via the Ludii general game system. Key contributions:

- Demonstrated that a single AlphaZero-style training pipeline could achieve strong play across dozens of different board games (Hex, Havannah, Breakthrough, Connect6, etc.) with minimal game-specific tuning.
- Introduced architectural search over network configurations (depth, width, pooling types) tailored to different board geometries, including **hexagonal convolutional networks** for hex-grid games.
- Showed that games with hexagonal boards benefit from hex-aware convolution kernels that respect the 6-neighbor topology rather than the standard 8-neighbor square grid.

### 6.2 OpenSpiel (Lanctot et al., 2019)

**OpenSpiel** (Lanctot et al., 2019) is a comprehensive open-source framework for research in games, developed at DeepMind. It is not itself an AI system but provides:

- Implementations of >70 games (perfect and imperfect information, simultaneous and sequential, one-player through many-player).
- Reference implementations of >30 algorithms including MCTS, CFR, AlphaZero, deep CFR, policy gradient methods, and evolutionary strategies.
- Standardized interfaces for benchmarking.
- Written in C++ with Python bindings, supporting both research prototyping and efficient large-scale training.

OpenSpiel has become the de facto standard for benchmarking new game-playing algorithms in the research community.

### 6.3 Leela Chess Zero (Lc0)

**Leela Chess Zero** (Lc0) is a community-driven, open-source chess engine that replicates the AlphaZero approach using distributed volunteer computing. Key milestones:

- Reached top-3 in the TCEC (Top Chess Engine Championship) Superfinal by 2020, competing with Stockfish.
- Trained on billions of self-play games contributed by thousands of volunteer GPUs worldwide.
- Demonstrated that the AlphaZero paradigm could be successfully replicated outside of Google's infrastructure, though requiring significantly more wall-clock time due to lower compute throughput.
- Explored architectural variations including SE-ResNets (squeeze-and-excitation blocks), different head structures, and transformer-based architectures.
- As of 2024--2025, Lc0 with MCTS remains competitive with Stockfish+NNUE in long-time-control games, though Stockfish dominates rapid play due to its faster per-move computation.

### 6.4 Student of Games (Schmid et al., 2023)

**Student of Games** (SoG) (Schmid et al., 2023) is a significant step toward a *general* game-playing algorithm that works in both perfect and imperfect information settings:

- Combines **growing-tree counterfactual regret minimization (GT-CFR)** for imperfect-information game solving with **self-play and learned value/policy networks** from the AlphaZero tradition.
- Uses a sound search algorithm that handles information asymmetry (private cards in poker, hidden units in Stratego) by maintaining belief distributions over hidden state.
- Achieved strong performance in both poker (imperfect information) and Go/chess (perfect information) within a single framework---the first algorithm to do so competently.
- Trained a learned model that generalizes across games with different information structures.

**Limitations**: SoG did not match the absolute playing strength of specialized systems (KataGo in Go, Pluribus in poker) but demonstrated that a unified framework is viable.

### 6.5 Additional Developments

- **ReAnalyze / MuZero Reanalyze** (Schrittwieser et al., 2021): Improved the original MuZero by periodically re-running MCTS on stored trajectories with the latest network, generating fresher training targets. This is a form of off-policy correction that significantly improves sample efficiency.

- **SpeedyZero** (Mei et al., 2023): Focused on wall-clock training speed, achieving EfficientZero-level Atari performance in ~30 minutes using a distributed architecture with 300 CPU actors and 4 GPUs. Key innovations include parallel MCTS, priority-refresh replay, and pipelined network updates.

- **AlphaZero-style approaches for other domains**: The self-play paradigm has been extended to mathematical theorem proving (AlphaProof, DeepMind 2024), chip design (Mirhoseini et al., 2021), and code generation, demonstrating the generality of the MCTS + neural network architecture beyond traditional games.

- **Adversarial robustness of game-playing AI**: Work by Wang et al. (2023) showed that even superhuman Go AIs (KataGo) can be exploited by adversarial policies that target blind spots in the network. This spurred research into adversarial training for game AI and highlighted that superhuman average performance does not imply robustness.

---

## 7. Comparative Summary

| System | Year | Rules Required | Stochastic Env | Action Space | Key Innovation | Sample Efficiency |
|--------|------|---------------|----------------|--------------|----------------|-------------------|
| AlphaZero | 2018 | Yes (simulator) | No | Discrete, small | Self-play + MCTS | Moderate (~40h TPU) |
| MuZero | 2020 | No (learned model) | Partial | Discrete, small | Learned dynamics | Low (200M frames Atari) |
| EfficientZero | 2021 | No (learned model) | Partial | Discrete, small | Consistency loss, value prefix | **High** (100k frames) |
| Sampled MuZero | 2021 | No | Partial | **Continuous/large** | Sampled MCTS | Moderate |
| Stochastic MuZero | 2022 | No | **Yes** | Discrete | Chance nodes in latent space | Moderate |
| Gumbel MuZero | 2022 | No (or Yes) | Either | Either | Gumbel-Top-k, sequential halving | Moderate (but fewer sims) |
| KataGo | 2019-- | Yes | No | Discrete | Auxiliary targets, engineering | **High** (50x less compute than ELF) |
| Student of Games | 2023 | Yes | Yes | Discrete | GT-CFR + self-play | Moderate |

---

## 8. Implications for Infinite Hexagonal Tic-Tac-Toe

The following analysis considers which post-AlphaZero innovations are most relevant for training a neural-network-based AI for infinite hexagonal tic-tac-toe---a fully observable, deterministic, two-player, zero-sum game played on an unbounded hexagonal grid where the effective action space grows with each move.

### 8.1 MuZero's Learned Dynamics vs. Known-Rules Search

Infinite hexagonal tic-tac-toe is a **fully observable, deterministic game with simple, known rules**: players alternate placing stones on hexagonal cells, and the win condition is a line of a specified length. The game state is fully determined by the history of moves, and the transition function is trivial (place a stone, flip the turn). In this setting, **AlphaZero-style planning with an exact forward model is strictly preferable to MuZero's learned dynamics** for several reasons:

- The dynamics function in MuZero introduces approximation error that accumulates over multi-step lookahead. For a deterministic game with a trivial transition function, this error is pure overhead.
- The representation and dynamics networks in MuZero consume parameters that could otherwise be allocated to a stronger value/policy network.
- Training the dynamics model requires additional loss terms and careful balancing of the multi-task objective.

**Recommendation**: Use AlphaZero-style search with the known game rules. MuZero's learned model is unnecessary overhead for this domain.

### 8.2 KataGo's Auxiliary Targets and Hex Tic-Tac-Toe Analogues

KataGo's auxiliary training targets are highly relevant and have direct analogues for hexagonal tic-tac-toe:

1. **Ownership prediction**: Predict, for each hexagonal cell, the probability that it will be "controlled" by each player (occupied and contributing to threatening lines) at the end of the game. On a hex grid, this amounts to predicting the spatial influence map---which regions of the board each player will dominate. This is a dense supervisory signal that encourages the network to learn territory and influence concepts.

2. **Score/margin prediction**: While tic-tac-toe variants are typically win/loss/draw, a continuous proxy could be defined: predict the *move advantage* (how many moves ahead of or behind the winning player is when the game ends) or predict the number of threats each player has at terminal states. This prevents the value head from saturating in clearly won/lost positions and provides gradient signal throughout training.

3. **Threat-line prediction**: An auxiliary head could predict the number and locations of open-ended threat lines (sequences of $k-1$ stones in a row with open endpoints) for each player. This is game-specific but directly analogous to KataGo's territory-awareness features and would encourage the network to learn tactical patterns.

4. **Future position prediction**: Predict the board state $n$ moves ahead. For a hexagonal grid, this could be approximated as predicting which cells will be occupied within the next 4--6 moves, encouraging forward-looking representations.

**Recommendation**: Implement at least ownership and threat-line auxiliary heads. These provide dense spatial gradients that are especially valuable for a game played on a potentially large, sparse board.

### 8.3 Gumbel MuZero for Efficient Search

Infinite hexagonal tic-tac-toe presents a **high branching factor problem**: the number of legal moves grows linearly with the board area, and on an unbounded grid, the effective action space could be very large (all empty cells within some radius of existing stones). This makes Gumbel MuZero's innovations particularly relevant:

- **Gumbel-Top-k action selection** can efficiently focus search on the most promising actions without wasting simulations on clearly inferior moves. This is critical when the branching factor is 50--200+ (typical for mid-game positions on a large hex grid).
- **Sequential halving** allocates the simulation budget optimally: rather than spreading 800 simulations across 150 legal moves (~5 each), it progressively eliminates unpromising actions and concentrates simulations on the top candidates.
- The **policy improvement guarantee** with a small, fixed budget means the system can achieve strong play even with tight computational constraints---important for a game where the branching factor makes deep search expensive.

**Quantitative estimate**: For a position with 100 legal moves and a budget of 200 simulations, standard PUCT would average 2 simulations per move at the root. Gumbel MuZero with sequential halving would select ~20 initial candidates, eliminate to ~10, then ~5, allocating ~20--40 simulations each to the final candidates. This yields far more reliable value estimates for the top actions.

**Recommendation**: Implement Gumbel MuZero-style search as the primary search algorithm. The branching factor of hex tic-tac-toe makes this the single most impactful algorithmic choice from the post-AlphaZero literature.

### 8.4 EfficientZero's Sample Efficiency Techniques

While EfficientZero was designed for the Atari 100k regime (learning from limited environment interaction), its techniques have modified applicability for self-play board game training:

- **Self-supervised consistency loss**: In a known-rules setting, this is less necessary because we can verify the model's predictions against the exact simulator. However, if auxiliary prediction heads are used (Section 8.2), a consistency loss between predicted and actual future features could regularize the learned representations.

- **Value prefix prediction**: This has a direct analogue in board games. Instead of predicting only the final game outcome (win/loss), predict the "value trajectory"---the cumulative value signal over the next $n$ moves. This is related to temporal-difference learning with varying horizons and could accelerate value function learning by providing intermediate supervision.

- **Aggressive reanalysis**: This is highly applicable. Re-searching stored games with the latest network to generate fresh policy targets is essentially free (no environment interaction needed, just inference) and can significantly improve training efficiency. KataGo also employs a form of this.

**Recommendation**: Implement reanalysis (re-search past games with updated networks). Consider value prefix prediction as a training target if early experiments show slow value convergence.

### 8.5 Synthesis: Recommended Architecture

Based on the above analysis, the recommended architecture for infinite hexagonal tic-tac-toe combines:

1. **AlphaZero-style exact forward model** (not MuZero's learned model) for the game dynamics.
2. **Gumbel MuZero search** with sequential halving for efficient action selection under high branching factor.
3. **KataGo-style auxiliary targets** (ownership, threat-lines, score margin) for dense training signal on the hex grid.
4. **Hexagonal convolutions** (following Polygames) that respect the 6-neighbor topology of the hex grid.
5. **Aggressive reanalysis** of past self-play games with updated network parameters.
6. **Playout cap randomization** (from KataGo) for training efficiency.

This combination takes the most relevant innovation from each post-AlphaZero system while avoiding unnecessary complexity (learned dynamics models, stochastic chance nodes) that is not required for a deterministic, fully observable, known-rules game.

---

## References

- Antonoglou, I., Schrittwieser, J., Ozair, S., Hubert, T., & Silver, D. (2022). Planning in stochastic environments with a learned model. *ICLR 2022*.
- Cazenave, T., Chen, Y.-C., Chen, G.-W., Chen, S.-Y., Chiu, X.-D., Dehos, J., ... & Ye, L. (2020). Polygames: Improved zero learning. *ICGA Journal*, 42(4), 244--256.
- Danihelka, I., Guez, A., Schrittwieser, J., & Silver, D. (2022). Policy improvement by planning with Gumbel. *ICLR 2022*.
- Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS 2020*.
- Hubert, T., Schrittwieser, J., Antonoglou, I., Barekatain, M., Schmitt, S., & Silver, D. (2021). Learning and planning in complex action spaces. *ICML 2021*.
- Lanctot, M., Lockhart, E., Lespiau, J.-B., Zambaldi, V., Upadhyay, S., Perolat, J., ... & Tuyls, K. (2019). OpenSpiel: A framework for reinforcement learning in games. *arXiv preprint arXiv:1908.09453*.
- Mei, J., Zhang, H., & Pan, W. (2023). SpeedyZero: Massively parallel reinforcement learning via model-based planning. *arXiv preprint*.
- Mirhoseini, A., Goldie, A., Yazgan, M., Jiang, J., Songhori, E., Wang, S., ... & Dean, J. (2021). A graph placement methodology for fast chip design. *Nature*, 594(7862), 207--212.
- Schmid, M., Moravcik, M., Burch, N., Kadlec, R., Davidson, J., Waugh, K., ... & Bowling, M. (2023). Student of Games: A unified learning algorithm for both perfect and imperfect information games. *Science Advances*, 9(46).
- Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., ... & Silver, D. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model. *Nature*, 588(7839), 604--609.
- Schrittwieser, J., Hubert, T., Mandhane, A., Barekatain, M., Antonoglou, I., & Silver, D. (2021). Online and offline reinforcement learning by planning with a learned model. *NeurIPS 2021*.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140--1144.
- Wang, T., Gleave, A., Tseng, T., Pelrine, K., Steinhardt, J., & Russell, S. (2023). Adversarial policies beat superhuman Go AIs. *ICML 2023*.
- Wu, D. J. (2019). Accelerating self-play learning in Go. *arXiv preprint arXiv:1902.10565*.
- Wu, D. J., et al. (2023). Adversarial training and defensive techniques in KataGo. *KataGo GitHub repository and technical documentation*.
- Ye, W., Liu, S., Kurutach, T., Abbeel, P., & Gao, Y. (2021). Mastering Atari games with limited data. *NeurIPS 2021*.

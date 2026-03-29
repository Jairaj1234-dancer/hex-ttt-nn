# AlphaGo, AlphaGo Zero, and AlphaZero: Training Architectures and Neural Network Design

**A Graduate-Level Research Survey**

---

## 1. Introduction

The AlphaGo family of programs---AlphaGo (Silver et al., 2016), AlphaGo Zero (Silver et al., 2017a), and AlphaZero (Silver et al., 2018)---represents a landmark progression in combining deep neural networks with Monte Carlo Tree Search (MCTS) for mastering perfect-information games. This document provides a detailed technical account of the architectures, training pipelines, loss functions, and design rationale behind each system, drawing from the original publications and their supplementary materials.

**Primary References:**

- **(Silver et al., 2016)**: Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484--489.
- **(Silver et al., 2017a)**: Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354--359.
- **(Silver et al., 2017b)**: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. *arXiv:1712.01815*. (Preprint of the AlphaZero work.)
- **(Silver et al., 2018)**: A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140--1144.

---

## 2. AlphaGo (2016)

### 2.1 Overview

AlphaGo was the first program to defeat a professional Go player (Fan Hui, 2-dan) 5--0 in October 2015, and subsequently defeated Lee Sedol (9-dan) 4--1 in March 2016 (Silver et al., 2016). Its architecture combined four distinct components: a supervised learning (SL) policy network, a reinforcement learning (RL) policy network, a value network, and a fast rollout policy, all integrated through MCTS.

### 2.2 Supervised Learning Policy Network

The SL policy network $p_\sigma$ was trained on a dataset of approximately 30 million board positions from 160,000 games played on the KGS Go Server. The network architecture was a 13-layer convolutional neural network (CNN) with the following structure (Silver et al., 2016, Extended Data Table 2):

- **Input representation**: 19 x 19 x 48 feature planes encoding stone positions, liberties, capture information, legality, turn history, and other handcrafted features.
- **Hidden layers**: 12 convolutional layers, each with 192 filters of kernel size 5 x 5 (first layer) or 3 x 3 (subsequent layers), followed by ReLU activations.
- **Output**: A softmax over all 19 x 19 = 361 intersections, representing a probability distribution over legal moves.

The network was trained to predict expert moves via stochastic gradient descent (SGD) on cross-entropy loss. It achieved 57.0% accuracy on a held-out test set, compared to 44.4% for previous state-of-the-art systems (Silver et al., 2016). A larger architecture with additional layers and 384 filters did not substantially improve accuracy, suggesting the 13-layer design was near the representational saturation point for the given input features.

### 2.3 Reinforcement Learning Policy Network

The RL policy network $p_\rho$ was initialized from the weights of the SL policy network and then improved through self-play reinforcement learning (Silver et al., 2016). The procedure was:

1. The current RL policy network plays games against a randomly selected earlier version of itself (to stabilize training and prevent overfitting to the current policy).
2. Games are played to completion. The outcome $z \in \{-1, +1\}$ serves as the reward signal.
3. Weights are updated via REINFORCE (Williams, 1992) policy gradient: $\Delta\rho \propto \frac{\partial \log p_\rho(a|s)}{\partial \rho} z_t$.

After training, the RL policy network won more than 80% of games against the SL policy network (Silver et al., 2016). Crucially, the RL policy network also won 85% of games against the strongest open-source Go program, Pachi, which performed 100,000 MCTS simulations per move.

### 2.4 Value Network

The value network $v_\theta(s)$ was a regression network trained to predict the expected game outcome from position $s$ under self-play by the RL policy network (Silver et al., 2016). Its architecture was similar to the policy network but with one critical difference:

- **Output**: A single scalar (via a tanh activation), predicting the expected outcome $v_\theta(s) \approx \mathbb{E}[z | s, p_\rho]$.
- **Training data**: 30 million positions, each sampled from a *distinct* self-play game to minimize correlations between training examples. This was essential; training on multiple positions from the same game led to severe overfitting.
- **Loss**: Mean squared error (MSE) between predicted value and actual game outcome.

The value network achieved an MSE of 0.226 on held-out data and approached the accuracy of full Monte Carlo rollouts with the RL policy network, but was orders of magnitude faster to evaluate (Silver et al., 2016, Extended Data Figure 4).

### 2.5 Fast Rollout Policy

In addition to the deep neural networks, AlphaGo used a fast rollout policy $p_\pi$ --- a linear softmax over a small set of pattern-based features. This policy was significantly weaker but extremely fast: it could evaluate a position in approximately 2 microseconds, compared to 3 milliseconds for the SL policy network on a GPU (a factor of ~1500x faster) (Silver et al., 2016). The rollout policy achieved 24.2% prediction accuracy on expert moves.

### 2.6 MCTS Integration

AlphaGo's MCTS operated as follows (Silver et al., 2016, Methods):

**Selection:** At each node $s$, the action $a$ maximizing

$$a^* = \arg\max_a \left[ Q(s,a) + u(s,a) \right]$$

was selected, where $Q(s,a)$ is the mean action-value from previous simulations and $u(s,a) \propto \frac{p_\sigma(a|s)}{1 + N(s,a)}$ is an exploration bonus proportional to the prior probability from the SL policy network, decaying with visit count $N(s,a)$.

**Expansion:** When a leaf node $s_L$ was reached, it was expanded and the SL policy network's output $p_\sigma(\cdot|s_L)$ was stored as prior probabilities for the new edges.

**Evaluation:** The leaf node was evaluated by a *mixture* of two signals:
$$V(s_L) = (1 - \lambda) \, v_\theta(s_L) + \lambda \, z_L$$
where $v_\theta(s_L)$ is the value network's estimate and $z_L$ is the outcome of a random rollout from $s_L$ using the fast rollout policy $p_\pi$. The mixing parameter $\lambda$ was set to 0.5 based on empirical tuning (Silver et al., 2016, Extended Data Table 4).

**Backup:** The value $V(s_L)$ was backed up through all edges traversed during the simulation, updating $Q(s,a)$ and $N(s,a)$.

AlphaGo used 1,920 CPUs and 280 GPUs in its distributed version for the match against Lee Sedol, running approximately 100,000 MCTS simulations per move (Silver et al., 2016).

### 2.7 Key Results

| Match | Result | Significance |
|-------|--------|-------------|
| AlphaGo vs. Fan Hui (Oct 2015) | 5--0 | First defeat of a professional Go player by a program |
| AlphaGo vs. Lee Sedol (Mar 2016) | 4--1 | Defeat of a world-class 9-dan professional |

An ablation study demonstrated that removing either the value network or the rollout evaluations significantly weakened play, confirming the importance of the dual-evaluation approach (Silver et al., 2016, Extended Data Figure 4). The single-machine version of AlphaGo (48 CPUs, 8 GPUs) achieved an estimated Elo rating of 3,140, while the distributed version reached 3,739 (Silver et al., 2016).

---

## 3. AlphaGo Zero (2017)

### 3.1 The Tabula Rasa Breakthrough

AlphaGo Zero eliminated all reliance on human expert data (Silver et al., 2017a). It learned entirely from self-play, starting from random play with no domain knowledge beyond the rules of Go. Despite this, it surpassed the strength of the version that defeated Lee Sedol within 36 hours of training and exceeded the strength of AlphaGo Master (which had defeated the world number one, Ke Jie, 3--0) within 72 hours.

Three critical simplifications over AlphaGo:

1. **No human data** --- the network is trained purely from self-play reinforcement learning.
2. **No handcrafted features** --- the input is raw board state (stone positions + simple derived features), rather than the 48 engineered feature planes of AlphaGo.
3. **No rollouts** --- the value head of the neural network entirely replaces Monte Carlo rollouts, eliminating the fast rollout policy.

### 3.2 Input Representation

The input to AlphaGo Zero was a 19 x 19 x 17 tensor (Silver et al., 2017a, Methods):

- **8 binary planes** for current player's stones over the last 8 time steps (current position + 7 history steps).
- **8 binary planes** for opponent's stones over the last 8 time steps.
- **1 binary plane** indicating the current player's color (all ones for black, all zeros for white).

This totals 17 feature planes per position. The use of history planes allows the network to detect ko situations and other temporal patterns without explicit feature engineering.

### 3.3 Unified Dual-Headed ResNet Architecture

AlphaGo Zero replaced AlphaGo's separate policy and value networks with a single neural network $f_\theta$ with two output heads (Silver et al., 2017a):

$$(\mathbf{p}, v) = f_\theta(s)$$

where $\mathbf{p}$ is a probability vector over moves and $v \in [-1, 1]$ is a scalar value estimate.

**Architecture details** (Silver et al., 2017a, Methods):

**Residual Tower (body):**
- 1 convolutional block: 256 filters, 3 x 3 kernel, stride 1, batch normalization, ReLU.
- Followed by either 19 or 39 residual blocks (the paper reports results for both; the final strongest version used 40 residual blocks, i.e., 1 initial convolutional block + 39 residual blocks, though the paper also references a 20-block version used for ablation comparisons).
- Each residual block: two convolutional layers (256 filters, 3 x 3 kernel, stride 1, batch normalization, ReLU after each), with a skip connection from input to output.

**Policy Head:**
- 1 convolutional layer: 2 filters, 1 x 1 kernel, batch normalization, ReLU.
- 1 fully connected layer: output size 19 x 19 + 1 = 362 (361 board positions + 1 pass move).
- Softmax activation to produce the probability vector $\mathbf{p}$.

**Value Head:**
- 1 convolutional layer: 1 filter, 1 x 1 kernel, batch normalization, ReLU.
- 1 fully connected layer: 256 hidden units, ReLU.
- 1 fully connected layer: output size 1, tanh activation to produce $v \in [-1, 1]$.

The total number of parameters for the 40-block version was approximately 80 million (Silver et al., 2017a, Supplementary Information). The 20-block version had roughly 40 million parameters.

### 3.4 Self-Play Reinforcement Learning Loop

The training pipeline consisted of three interleaved processes (Silver et al., 2017a):

**Process 1 --- Self-Play Data Generation:**
- The latest neural network $f_\theta$ guides MCTS to generate complete self-play games.
- For each position $s_t$ in the game, MCTS runs 1,600 simulations and produces a policy vector $\boldsymbol{\pi}_t$ (the normalized visit counts of the root's children) and a value estimate.
- Each game is played to termination, yielding outcome $z \in \{-1, +1\}$.
- The resulting training examples are $(s_t, \boldsymbol{\pi}_t, z_t)$, where $z_t$ is the game outcome from the perspective of the current player at time $t$.
- 25,000 games of self-play were generated per training iteration. Each self-play game used approximately 1,600 MCTS simulations per move.

**Process 2 --- Neural Network Training:**
- The network parameters $\theta$ are updated by sampling uniformly from the most recent 500,000 games of self-play data (a replay buffer).
- Mini-batch SGD with momentum is applied to minimize the combined loss function (see Section 6).
- Training used a mini-batch size of 2,048, learning rate of 0.01 (annealed during training), momentum of 0.9, and L2 regularization coefficient $c = 10^{-4}$ (Silver et al., 2017a, Methods).

**Process 3 --- Evaluation and Model Replacement:**
- After every 1,000 training steps, the updated network is evaluated against the current best network by playing 400 games.
- If the new network wins 55% or more of the games, it replaces the current best network and becomes the one used for self-play data generation.
- This checkpoint mechanism ensures that training data quality is monotonically non-decreasing.

The 20-block network was trained for 3 days (approximately 4.9 million games of self-play) and the 40-block network for approximately 40 days (approximately 29 million games of self-play), using 4 TPUs for training and 64 GPU workers for self-play (Silver et al., 2017a).

### 3.5 MCTS in AlphaGo Zero

MCTS in AlphaGo Zero was simplified relative to AlphaGo (Silver et al., 2017a):

**Selection** used PUCT (Polynomial Upper Confidence Trees):

$$a_t = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$

where $P(s, a)$ is the prior probability from the policy head, $N(s, a)$ is the visit count, and $c_{\text{puct}}$ is an exploration constant. In AlphaGo Zero, $c_{\text{puct}}$ was set to a value that controlled the degree of exploration (approximately 1.0--2.5 depending on the configuration).

**Expansion and Evaluation:** When a leaf node $s_L$ is reached, the neural network is evaluated *once*: $(P(s_L, \cdot), V(s_L)) = f_\theta(s_L)$. The value $V(s_L)$ is used directly --- no rollouts. The prior probabilities $P(s_L, \cdot)$ are stored on the edges.

**Backup:** The value $V(s_L)$ is propagated back along the traversed path, updating $Q(s,a)$ as the mean of all values backed up through edge $(s, a)$.

### 3.6 Exploration Mechanisms

**Dirichlet Noise at the Root:**
To ensure exploration in the self-play games, Dirichlet noise was added to the prior probabilities at the root node (Silver et al., 2017a):

$$P(s_{\text{root}}, a) = (1 - \varepsilon) \cdot p_a + \varepsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

where $\varepsilon = 0.25$ and $\alpha = 0.03$ (for the 19 x 19 Go board; chosen to be roughly $10/n$ where $n$ is the approximate number of legal moves). This encourages the MCTS to explore moves that the policy network assigns low probability, which is critical for discovering novel strategies.

**Temperature-Based Move Selection:**
During the opening phase of each self-play game (first 30 moves), moves were selected proportionally to the visit counts with temperature $\tau = 1$:

$$\pi(a | s) \propto N(s, a)^{1/\tau}$$

For the remainder of the game, the temperature was set to an infinitesimal value $\tau \to 0$, making the policy greedy (selecting the most-visited move). This ensured diversity in the opening while maintaining strong play in the middle and endgame (Silver et al., 2017a).

### 3.7 Key Results

| Model | Elo Rating | Training Time |
|-------|-----------|---------------|
| AlphaGo Zero (20 blocks, 3 days) | 4,670 | 72 hours |
| AlphaGo Zero (40 blocks, 40 days) | 5,185 | ~40 days |
| AlphaGo Lee (vs. Lee Sedol version) | 3,739 | Months |
| AlphaGo Master (vs. Ke Jie version) | 4,858 | Months |

AlphaGo Zero (40-block) defeated AlphaGo Zero (20-block) in 90% of games, and defeated AlphaGo Master in 89% of games (Silver et al., 2017a). A key finding was that a single dual-headed network outperformed separate policy and value networks (an ablation showed that the dual-headed architecture achieved stronger play than two independent networks, likely due to shared low-level features improving both heads).

---

## 4. AlphaZero (2018)

### 4.1 Generalization Across Games

AlphaZero generalized the AlphaGo Zero approach to three games---chess, shogi, and Go---using a single unified algorithm with minimal game-specific adaptation (Silver et al., 2018). The only game-specific elements were the input/output representation (board encoding and action space) and the game rules for move generation and termination.

### 4.2 Architectural Differences from AlphaGo Zero

AlphaZero made several simplifications relative to AlphaGo Zero (Silver et al., 2018, Supplementary Materials):

1. **No checkpoint evaluation**: Instead of evaluating new networks against the current best and only replacing on a win threshold, AlphaZero continuously used the latest network parameters for self-play. This simplified the pipeline and eliminated the evaluation bottleneck.

2. **Draws**: The value target was extended to $z \in \{-1, 0, +1\}$ to handle draws in chess and shogi (Go has no draws under Chinese rules as used in AlphaGo Zero).

3. **No data augmentation**: AlphaGo Zero exploited the 8-fold symmetry of the Go board (rotations and reflections) to augment training data. AlphaZero did not use symmetry augmentation, since chess and shogi lack this symmetry (castling rights, pawn direction, etc.).

4. **Hyperparameter tuning**: The exploration noise parameter $\alpha$ was adjusted per game: $\alpha = 0.3$ for chess, $\alpha = 0.15$ for shogi, and $\alpha = 0.03$ for Go (reflecting the different branching factors: ~30 for chess, ~80 for shogi, ~250 for Go).

5. **Same architecture for all games**: A single ResNet architecture (either 20 blocks with 256 filters, or the deeper version) was applied identically, demonstrating the generality of the approach.

### 4.3 Network Architecture (AlphaZero)

The architecture was essentially the same as AlphaGo Zero's dual-headed ResNet (Silver et al., 2018):

- **Body**: 19 residual blocks (20 blocks total including initial convolutional block), each with 256 filters.
- **Policy head**: Game-specific output size. For chess: 4,672 possible moves (encoded as 8 x 8 x 73 planes representing source square and move type). For shogi: 11,259 possible moves. For Go: 362 (19 x 19 + pass).
- **Value head**: Identical structure to AlphaGo Zero (1 filter convolution, FC-256-ReLU, FC-1-tanh).

**Input representations** (Silver et al., 2018, Methods):
- **Chess**: 8 x 8 x 119 (piece positions for last 8 moves for each of 6 piece types x 2 colors = 96 planes, plus 7 constant planes for castling rights, move counters, etc., additional meta-planes for total of 119).
- **Shogi**: 9 x 9 x 362 planes.
- **Go**: 19 x 19 x 17 (same as AlphaGo Zero).

### 4.4 Training Details

Training was performed on 5,000 first-generation TPUs for self-play generation and 64 second-generation TPUs for training the network (Silver et al., 2018). Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| MCTS simulations per move | 800 |
| Mini-batch size | 4,096 |
| Learning rate | 0.2, annealed to 0.02, 0.002, 0.0002 |
| Momentum | 0.9 |
| L2 regularization | $c = 10^{-4}$ |
| Replay buffer | Most recent 1,000,000 games |
| Training steps | 700,000 (chess), 700,000 (shogi), 700,000 (Go) |
| Training time | ~9 hours (chess), ~12 hours (shogi), ~13 days (Go) |
| Self-play games generated | 44M (chess), 44M (shogi), 21M (Go) |

(Silver et al., 2018, Table S3, Supplementary Materials)

### 4.5 Key Results

| Game | Opponent | AlphaZero Elo | Result |
|------|----------|--------------|--------|
| Chess | Stockfish (v8, 3,400+ Elo) | ~3,600+ | +155 -6 =839 (1000 games) |
| Shogi | Elmo (World Computer Shogi Champion 2017) | --- | +98.2% win rate in 1000 games (+91 -8 =1) |
| Go | AlphaGo Zero (3-day, 20-block) | --- | +61 -39 =0 (100 games) |

In chess, AlphaZero achieved superhuman performance after approximately 4 hours of training (~300,000 training steps) (Silver et al., 2018). The chess results were particularly striking because AlphaZero searched only ~80,000 positions per second compared to Stockfish's ~70 million, yet achieved substantially stronger play by evaluating far fewer but more promising positions.

---

## 5. The MCTS + Neural Network Synergy

### 5.1 How the Policy Head Guides Search

The policy head output $P(s, a)$ serves as the prior probability in the PUCT selection formula. This means the neural network immediately biases the search toward moves it considers promising, dramatically pruning the effective branching factor. Without learned priors, MCTS would need to explore an impractically large tree (Silver et al., 2017a).

Formally, the prior probability affects the exploration bonus term in PUCT:

$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

Moves with higher prior probability receive proportionally larger bonuses, causing them to be explored earlier and more frequently. As $N(s, a)$ grows, the bonus diminishes, allowing the search to eventually explore lower-prior moves --- but the prior ensures the most promising moves are explored first.

### 5.2 How the Value Head Replaces Rollouts

In AlphaGo, leaf evaluation required rolling out the game to completion using a fast rollout policy, which was both noisy and computationally expensive. AlphaGo Zero showed that a trained value head provides a more accurate and computationally cheaper evaluation (Silver et al., 2017a, Extended Data Figure 3). The value network effectively "compresses" the information from millions of game completions into a single forward pass, providing a smooth, differentiable estimate of position value.

The ablation study in (Silver et al., 2017a) demonstrated that using only the value head (without rollouts) outperformed using only rollouts (without the value head), and also outperformed the mixed evaluation used in AlphaGo. This is a remarkable result: a learned evaluation function surpassed a brute-force Monte Carlo estimate.

### 5.3 Policy Improvement Through Search

MCTS acts as a powerful *policy improvement operator*. Given a neural network policy $p_\theta$ and value function $v_\theta$, MCTS produces an improved policy $\boldsymbol{\pi}$ (the normalized root visit counts) that is strictly better than the raw network policy (Silver et al., 2017a). This is analogous to the policy improvement theorem in dynamic programming: look-ahead search with value estimates yields better action selection than the value estimates alone.

The self-play training loop then closes the circle:

1. **MCTS improves the policy**: $\boldsymbol{\pi}$ is a better policy than $p_\theta$ because it incorporates look-ahead.
2. **Training improves the network**: The network is trained to match $\boldsymbol{\pi}$, i.e., $p_\theta \to \boldsymbol{\pi}$.
3. **The improved network enables better MCTS**: A stronger prior and value estimate enables more effective search.

This creates a virtuous cycle of mutual improvement. Silver et al. (2017a) describe this as "policy iteration" where MCTS plays the role of the policy improvement step and neural network training plays the role of policy evaluation.

### 5.4 Self-Play Data Generation Process

The detailed self-play data generation pipeline for AlphaGo Zero / AlphaZero:

1. **Initialize** with the current best network $f_\theta$.
2. **For each game**:
   a. Start from the initial board position.
   b. At each position $s_t$:
      - Run 1,600 MCTS simulations (AlphaGo Zero) or 800 simulations (AlphaZero) from $s_t$.
      - At the root, add Dirichlet noise to the prior: $P'(s_t, a) = (1-\varepsilon) P(s_t, a) + \varepsilon \eta_a$.
      - After simulations, compute the search policy $\boldsymbol{\pi}_t$ as the normalized visit counts of root children.
      - Select a move: $a_t \sim \boldsymbol{\pi}_t^{1/\tau}$ where $\tau = 1$ for opening moves, $\tau \to 0$ later.
      - Record the training example $(s_t, \boldsymbol{\pi}_t)$.
   c. Play until game termination; record outcome $z$.
   d. Assign $z_t = z$ if the player at time $t$ is the winner, $z_t = -z$ otherwise.
3. **Store** all $(s_t, \boldsymbol{\pi}_t, z_t)$ in the replay buffer.

---

## 6. Loss Functions

### 6.1 Combined Loss

The neural network $f_\theta$ is trained end-to-end by minimizing the following loss (Silver et al., 2017a):

$$\ell = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c \|\theta\|^2$$

where:

- $z$ is the actual game outcome ($+1$ or $-1$, or $0$ for draws in AlphaZero).
- $v$ is the value head output.
- $\boldsymbol{\pi}$ is the MCTS search policy (normalized visit counts).
- $\mathbf{p}$ is the policy head output.
- $c = 10^{-4}$ is the L2 regularization coefficient.
- $\theta$ is the full set of network parameters.

### 6.2 Value Loss: MSE with Game Outcome

The value loss $(z - v)^2$ is a standard mean squared error between the network's prediction of the game outcome and the actual result. Using the actual game outcome as the target (rather than, say, a bootstrapped estimate from the value network) provides an unbiased, low-variance training signal. The game outcome is the ground truth for the value function of the optimal policy, making this a form of Monte Carlo policy evaluation.

The choice of MSE (rather than, e.g., cross-entropy on a discretized outcome) reflects the fact that $v$ is a continuous prediction on $[-1, 1]$ and the target $z$ is at the boundaries. Under this formulation, $v$ can be interpreted as the expected outcome under optimal play from both sides (Silver et al., 2017a).

### 6.3 Policy Loss: Cross-Entropy with MCTS Visit Counts

The policy loss $-\boldsymbol{\pi}^\top \log \mathbf{p}$ is the cross-entropy between the MCTS-improved policy and the raw network policy. This is a critical design choice: the target is *not* the actual move played, but the full *distribution* over moves produced by MCTS. This is information-theoretically richer than a one-hot target --- the network learns not just which move is best, but the relative merits of all moves as estimated by the search.

This choice means the network is trained to match the search policy, which embodies the policy improvement step. The MCTS visit distribution is a soft target that encodes the relative strength of different moves, providing a richer gradient signal than a hard expert label.

### 6.4 L2 Regularization

L2 regularization $c \|\theta\|^2$ with $c = 10^{-4}$ prevents overfitting to the self-play data. This is standard practice for deep networks and is particularly important here because the training data is generated by the network itself, creating a risk of "echo chamber" effects where the network overfits to its own biases. The relatively small coefficient reflects the large volume of training data (millions of self-play positions).

### 6.5 Why This Combination Works

The elegance of this loss function lies in how it unifies the three objectives:

- The **value loss** teaches the network to evaluate positions accurately, enabling strong leaf evaluation in MCTS.
- The **policy loss** teaches the network to replicate the search's improved policy, enabling effective tree pruning via priors.
- The **L2 term** prevents overfitting and encourages generalization.

The shared trunk architecture means that features useful for evaluation (value head) are also available for move selection (policy head), and vice versa. This creates a synergistic training dynamic: better value estimates lead to better MCTS evaluations, which produce better policy targets, which train better shared features, which improve value estimates further.

---

## 7. Key Architectural Decisions and Their Rationale

### 7.1 Why Residual Networks?

The shift from plain CNNs (AlphaGo) to ResNets (AlphaGo Zero) was motivated by the well-established benefits of residual connections (He et al., 2016):

1. **Depth without degradation**: Residual connections allow gradient flow through skip connections, enabling training of much deeper networks (40 blocks = 80 convolutional layers + additional head layers) without the vanishing gradient problem.
2. **Empirical superiority**: Silver et al. (2017a) directly compared a 12-layer plain CNN (similar to AlphaGo) against a 20-block ResNet and found the ResNet achieved significantly stronger play (Extended Data Figure 2). The ResNet's Elo advantage over the plain CNN grew over training, indicating that depth with residual connections enables learning of increasingly sophisticated features.
3. **Feature reuse**: Residual connections enable the network to learn modifications to features across layers rather than learning entire new representations, which is more parameter-efficient.

### 7.2 Why a Dual-Headed Architecture?

The dual-headed (two-head) design where a shared body produces both policy and value outputs was a key innovation of AlphaGo Zero:

1. **Shared representation**: Lower-level features (e.g., local patterns, tactical motifs) are useful for both evaluating a position and selecting moves. A shared trunk avoids redundant computation and enables cross-pollination between the two objectives.
2. **Regularization effect**: Multi-task learning acts as a form of implicit regularization. The value head's objective prevents the policy head from overfitting to superficial move patterns, and vice versa.
3. **Computational efficiency**: A single forward pass produces both the prior probabilities (for MCTS selection) and the leaf value (for MCTS evaluation). This halves the neural network computation per MCTS simulation compared to using two separate networks.
4. **Empirical validation**: Silver et al. (2017a) showed that the dual-headed architecture outperformed separate networks in both Elo and training speed (Extended Data Figure 2).

### 7.3 Batch Normalization

Every convolutional layer in the residual tower is followed by batch normalization (Ioffe & Szegedy, 2015) before the ReLU activation. In the AlphaGo Zero / AlphaZero context:

- It stabilizes training by normalizing activations, which is crucial when training from self-play data whose distribution shifts as the network improves.
- It allows higher learning rates, accelerating training.
- It provides a mild regularization effect through the stochasticity of mini-batch statistics.

The ordering is: Convolution -> Batch Normalization -> ReLU, following the original ResNet pre-activation scheme (He et al., 2016).

### 7.4 ReLU Activations

ReLU (Rectified Linear Unit) activations $f(x) = \max(0, x)$ were used throughout. The choice is standard for deep CNNs and provides:

- Sparse activation patterns, which are computationally efficient.
- Non-saturating gradients (for positive inputs), avoiding the vanishing gradient problem.
- Fast computation compared to sigmoid or tanh.

The only exception is the final activation in the value head, which uses tanh to bound the output to $[-1, 1]$.

### 7.5 Global Structure of the Value Head

The value head uses a 1 x 1 convolution (reducing from 256 feature maps to 1), followed by a fully connected layer with 256 hidden units and ReLU, then a final FC layer with tanh output. The design rationale:

1. **1 x 1 convolution**: Reduces spatial feature maps to a single channel, providing a compressed spatial summary of the position.
2. **Fully connected layer**: Aggregates information across all spatial locations. This is essential because positional value depends on the *global* configuration of stones, not just local patterns. The FC layer implicitly performs a form of global reasoning (note: this is equivalent to flattening followed by a dense layer, not global average pooling; the global average pooling interpretation sometimes cited is slightly inaccurate for the original AlphaGo Zero architecture as described in the paper).
3. **Tanh output**: Bounds the value to $[-1, 1]$, matching the range of game outcomes.

### 7.6 The PUCT Selection Formula

The specific variant of UCT used in AlphaGo Zero and AlphaZero is based on Rosin (2011)'s Predictor UCT:

$$a_t = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]$$

This differs from standard UCT (Kocsis & Szepesvari, 2006) by replacing the $\sqrt{\log N_{\text{parent}}}$ term with $\sqrt{N_{\text{parent}}}$ and weighting the exploration term by the prior $P(s,a)$. The effect is:

- The exploration bonus decays as $\sim 1/\sqrt{N(s,a)}$ rather than $\sim \sqrt{\log(N_{\text{parent}})/N(s,a)}$, providing slightly stronger exploration.
- The prior $P(s,a)$ directly biases exploration toward promising moves, making the search policy-guided from the first simulation.

### 7.7 Virtual Loss for Parallel MCTS

To enable parallel evaluation of the neural network on GPUs/TPUs, AlphaGo and its successors use **virtual loss** (Silver et al., 2016; Silver et al., 2017a). When a simulation thread selects an edge $(s, a)$, a "virtual loss" is temporarily added: $N(s, a)$ is incremented and $W(s, a)$ (the total value) is decremented by a virtual loss amount before the simulation completes. This has the effect of:

1. **Discouraging other threads from selecting the same path**, promoting diversity in parallel simulations.
2. **Maintaining correct statistics asymptotically**: Once the simulation completes, the virtual loss is removed and replaced by the actual neural network evaluation.

In AlphaGo Zero, 8 simulations were run in parallel per search thread, with virtual losses of $n_{\text{vl}} = 3$ (Silver et al., 2017a, Supplementary Information). This enabled efficient batching of neural network evaluations on the GPU/TPU.

---

## 8. Summary of Architectural Evolution

| Feature | AlphaGo (2016) | AlphaGo Zero (2017) | AlphaZero (2018) |
|---------|---------------|---------------------|------------------|
| Human data | 160K games (SL) | None | None |
| Input features | 48 handcrafted planes | 17 raw planes | Game-specific raw planes |
| Network type | Separate 13-layer CNNs | Dual-headed 40-block ResNet | Dual-headed 20-block ResNet |
| Rollouts | Yes (fast policy) | No | No |
| Leaf evaluation | $0.5 v_\theta + 0.5 z_{\text{rollout}}$ | $v_\theta$ only | $v_\theta$ only |
| Data augmentation | N/A | 8-fold symmetry | None |
| Checkpoint eval | N/A | Yes (55% threshold) | No (continuous) |
| Games covered | Go | Go | Chess, Shogi, Go |
| MCTS simulations/move | ~100,000 | 1,600 | 800 |
| Elo (Go) | 3,739 (distributed) | 5,185 (40-block) | Comparable to AGZ-20b |

---

## 9. Implications for Infinite Hexagonal Tic-Tac-Toe

The AlphaZero framework provides a strong starting point for training an agent to play infinite hexagonal tic-tac-toe (hex-grid, unbounded board, 6-in-a-row win condition, two-piece-per-turn after the first move). However, several fundamental differences between Go/chess and this game require significant architectural and algorithmic adaptations.

### 9.1 Hexagonal Grid Geometry

**What transfers directly:**
- The convolutional neural network paradigm remains appropriate. Hexagonal grids have local spatial structure amenable to convolution.
- The dual-headed ResNet architecture (shared body, policy head, value head) transfers directly.

**What needs rethinking:**
- **Hexagonal convolutions**: Standard square convolutions on a rectangular pixel grid do not align with hexagonal adjacency. Each hex cell has 6 neighbors (not 4 or 8). Two main approaches exist:
  - *Offset coordinates*: Map the hex grid to a rectangular array with offset rows/columns and use standard convolutions, accepting that the kernel's spatial structure does not perfectly match hex adjacency. This is the simplest approach and may be "good enough" given sufficient training data.
  - *Axial/cube coordinates with hex-aware kernels*: Use axial coordinates $(q, r)$ and define custom convolution kernels that respect the 6-neighbor topology. This is more principled and preserves the rotational symmetries of the hexagonal grid. Libraries for hexagonal convolutions exist (e.g., HexagDLy).
- **Symmetry group**: The hexagonal grid has 6-fold rotational symmetry and 6 reflection axes (the dihedral group $D_6$, with 12 elements), compared to the square grid's $D_4$ (8 elements). Data augmentation should exploit this 12-fold symmetry to multiply training data.

### 9.2 Infinite/Unbounded Board

This is the most significant departure from standard AlphaZero. Go has a fixed 19 x 19 board; chess has 8 x 8. An unbounded board presents fundamental challenges:

**What needs rethinking:**
- **Fixed-size input**: Neural networks require fixed-size input tensors. The solution is to define a *viewport* or *bounding box* around the current stones, with sufficient margin for expansion. A reasonable approach:
  - Compute the axis-aligned bounding box of all placed stones.
  - Extend by a margin of $M$ cells in each direction (e.g., $M = 3$ to $6$, given the 6-in-a-row win condition).
  - If the viewport is smaller than some minimum size, pad to a minimum (e.g., 11 x 11 or 15 x 15).
  - If it exceeds a maximum size, this represents a potential issue, but in practice the active playing area should be bounded for reasonable game lengths.
- **Variable action space**: The policy head cannot output a fixed-size softmax over all possible moves since the board is infinite. Options:
  - Output a spatial policy map over the viewport, with the same spatial dimensions as the input. Each cell in the viewport is a candidate move. This is the most natural AlphaZero-like approach.
  - Restrict legal moves to cells within or adjacent to the viewport (a heuristic, but reasonable: placing a stone far from all existing stones is almost certainly suboptimal for 6-in-a-row).
- **Translational invariance**: On an infinite board, the absolute position of stones does not matter; only their relative configuration does. This suggests that the input should be *centered* (e.g., on the centroid of existing stones or the last move) rather than using absolute coordinates. Convolutional networks inherently capture translational equivariance, which aligns well with this property.
- **No edge effects**: Unlike Go, where edges and corners are strategically significant, the infinite board has no edges. This simplifies the value function in some respects (no edge-related patterns to learn) but removes a source of strategic asymmetry.

### 9.3 6-in-a-Row Win Condition

**What transfers directly:**
- The value head's structure (predicting game outcome) is unchanged. The win condition is a binary outcome (win/loss, or draw if one is possible).
- MCTS with PUCT selection works identically regardless of the win condition.

**What needs rethinking:**
- **Longer tactical horizons**: 6-in-a-row requires building longer chains than 5-in-a-row (Gomoku) or capturing groups (Go). The network must detect threats and partial chains of length 1 through 6 in all 3 hexagonal axis directions. This places a premium on the network's ability to detect oriented linear patterns, which standard convolutions handle well but hex-aware convolutions would handle better.
- **Threat detection at distance**: Forced sequences (threats, double threats) may extend over many moves. The receptive field of the network must be large enough to capture these. With 3 x 3 kernels and $K$ residual blocks, the receptive field has diameter $\sim 4K + 1$. For a 20-block network, this is ~81 cells, which should be sufficient for local tactical calculations on a hex grid.
- **Defensive complexity**: With 6 cells needed, the game likely has a different balance of offense vs. defense compared to 5-in-a-row. The network and MCTS will need to discover this balance through self-play.

### 9.4 Two-Piece-Per-Turn Mechanics

This is the most novel game-mechanical feature and requires the most significant adaptation:

**What needs rethinking:**
- **Action space**: Each turn (except the first) consists of placing *two* stones. This could be modeled in two ways:
  - *Composite action*: A single turn is an ordered pair of placements $(a_1, a_2)$. The action space is $O(n^2)$ where $n$ is the number of legal placement cells. This is likely too large for a flat policy head.
  - *Sequential sub-actions* (recommended): Model each turn as two sequential decisions. After placing the first stone, the board state updates (the first stone is now placed), and the network is queried again for the second placement. The MCTS tree then has turns that consist of two plies for one player. This is the more natural approach and has precedent in multi-action game modeling.
- **MCTS structure**: The tree must accommodate the two-ply-per-turn structure. Each "player turn" node has children representing the first placement, and each first-placement node has children representing the second placement. The value backup should occur per *turn* (both placements), not per *ply*.
- **First-move exception**: The first player places only one stone on the first turn. This is a minor special case that the network can learn from the input representation (e.g., an input feature plane indicating the turn number or the total stone count).
- **Input representation**: The network should encode not only the current board state but also whether the current player has already placed their first stone this turn (i.e., a binary flag plane indicating "first placement" vs. "second placement" within a turn).

### 9.5 Summary of Adaptations

| AlphaZero Component | Transfers Directly? | Required Adaptation |
|---------------------|---------------------|---------------------|
| Dual-headed ResNet | Yes (with modifications) | Hex convolutions or offset-grid convolutions |
| MCTS + PUCT | Mostly yes | Two-ply turn structure |
| Self-play training loop | Yes | No change |
| Loss function | Yes | No change |
| Dirichlet noise | Yes | Adjust $\alpha$ for action space size |
| Temperature move selection | Yes | Apply per-turn (to each placement separately) |
| Input: fixed-size tensor | No | Dynamic viewport with centering |
| Output: fixed-size policy | No | Spatial policy over viewport |
| Symmetry augmentation | Yes, but different | $D_6$ (12-fold) instead of $D_4$ (8-fold) |
| Value prediction | Yes | No change |

The core insight of AlphaZero---that a neural network providing prior probabilities and value estimates, trained through self-play against MCTS-improved targets, produces superhuman play---is fully applicable to infinite hexagonal tic-tac-toe. The main engineering challenges are handling the unbounded board through a dynamic viewport mechanism and modeling the two-piece-per-turn mechanic through sequential sub-actions within MCTS.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*.
- Kocsis, L., & Szepesvari, C. (2006). Bandit Based Monte-Carlo Planning. *ECML*.
- Rosin, C. D. (2011). Multi-armed bandits with episode context. *Annals of Mathematics and Artificial Intelligence*, 61(3), 203--230.
- Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484--489.
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017a). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354--359.
- Silver, D., Hubert, T., Schrittwieser, J., et al. (2017b). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. *arXiv:1712.01815*.
- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140--1144.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229--256.

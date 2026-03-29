# Monte Carlo Tree Search Adaptations for Non-Standard Game Mechanics

**Research Document 04 -- MCTS for Multi-Move Turns, Large Action Spaces, and Hexagonal Connection Games**

*Date: 2026-03-29*

---

## Table of Contents

1. [MCTS Fundamentals Review](#1-mcts-fundamentals-review)
2. [MCTS for Multi-Move Turns](#2-mcts-for-multi-move-turns)
3. [MCTS for Large/Infinite Action Spaces](#3-mcts-for-largeinfinite-action-spaces)
4. [Transposition Tables and DAGs in MCTS](#4-transposition-tables-and-dags-in-mcts)
5. [Progressive Strategies and Early Termination](#5-progressive-strategies-and-early-termination)
6. [Scaling MCTS: Simulation Budget and Parallelism](#6-scaling-mcts-simulation-budget-and-parallelism)
7. [MCTS Design for Infinite Hex Tic-Tac-Toe](#mcts-design-for-infinite-hex-tic-tac-toe)

---

## 1. MCTS Fundamentals Review

### 1.1 The UCB1 Formula and Its Descendants

Monte Carlo Tree Search (Coulom, 2006; Kocsis & Szepesvari, 2006) builds an asymmetric search tree guided by random simulations. The foundational selection policy derives from the Upper Confidence Bound for Bandits (UCB1) (Auer et al., 2002):

$$
\text{UCB1}(s, a) = \bar{Q}(s, a) + C \sqrt{\frac{\ln N(s)}{N(s, a)}}
$$

where $\bar{Q}(s, a)$ is the mean value of action $a$ from state $s$, $N(s)$ is the visit count of state $s$, $N(s, a)$ is the visit count of action $a$ from $s$, and $C$ is the exploration constant (theoretically $\sqrt{2}$ for regret-optimal bounds, but tuned in practice).

**UCT (Upper Confidence bounds applied to Trees)** (Kocsis & Szepesvari, 2006) applies UCB1 recursively at each internal node of the search tree. UCT treats each node's children as a multi-armed bandit problem, achieving logarithmic regret per node while the overall tree converges to the minimax value.

**PUCT (Predictor + UCT)**, as used in AlphaGo Zero and AlphaZero (Silver et al., 2017; 2018), incorporates a learned prior policy $\pi_\theta(a|s)$:

$$
\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot \pi_\theta(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
$$

Key differences from UCB1: (i) the exploration term scales linearly with the prior $\pi_\theta(a|s)$, meaning the neural network's policy output directly controls which branches are explored first; (ii) the $\sqrt{\ln N}$ is replaced with $\sqrt{N}$, yielding slightly more aggressive exploration; (iii) as $N(s,a) \to \infty$, the term decays to zero and selection becomes purely exploitation-driven. The constant $c_{\text{puct}}$ is often set dynamically -- AlphaZero uses $c_{\text{puct}}(s) = \log\left(\frac{1 + N(s) + c_{\text{base}}}{c_{\text{base}}}\right) + c_{\text{init}}$ with $c_{\text{base}} = 19652$ and $c_{\text{init}} = 1.25$ (Silver et al., 2018).

### 1.2 The Four Phases

Each MCTS iteration executes four phases:

**Phase 1: Selection.** Starting from the root, traverse the tree by selecting the child that maximizes the selection policy (UCT, PUCT, etc.) at each internal node. This continues until a leaf node is reached -- either an unexpanded node or a terminal state.

**Phase 2: Expansion.** At the leaf, one or more children are added to the tree. In vanilla MCTS, a single child corresponding to an untried action is added. In AlphaZero-style MCTS, all legal actions are added at once (with prior probabilities from the policy network), but only the selected child receives a simulation/evaluation.

**Phase 3: Simulation / Evaluation.** In classical MCTS, a random rollout (playout) is run from the expanded node to a terminal state, yielding a win/loss/draw signal. In AlphaZero-style MCTS, the rollout is replaced entirely by a value network $v_\theta(s)$ that estimates the expected outcome from state $s$. This substitution is critical for efficiency -- rollouts in complex games are uninformative, while a trained value network provides a much stronger signal (Silver et al., 2017).

**Phase 4: Backpropagation.** The evaluation result $v$ is propagated back along the path from the leaf to the root. Each node along the path updates its visit count $N(s,a) \leftarrow N(s,a) + 1$ and its cumulative value $W(s,a) \leftarrow W(s,a) + v$, where $v$ is negated at each level for two-player zero-sum games ($Q$ is always from the perspective of the player to move).

```
function MCTS_SEARCH(root_state, num_simulations):
    root = Node(state=root_state)

    for i = 1 to num_simulations:
        node = root
        state = root_state.copy()

        # SELECTION
        while node.is_fully_expanded() and not state.is_terminal():
            action = argmax_a PUCT(node.state, a)
            node = node.children[action]
            state.apply(action)

        # EXPANSION
        if not state.is_terminal():
            prior_probs = policy_network(state)
            node.expand(state.legal_actions(), prior_probs)

        # EVALUATION
        value = value_network(state)  # or rollout(state)

        # BACKPROPAGATION
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # flip for opponent
            node = node.parent

    return root.best_child(temperature)
```

### 1.3 Virtual Loss for Parallelization

When running MCTS with multiple threads (or batching evaluations for GPU), the naive approach of locking the tree is prohibitively slow. Virtual loss (Chaslot et al., 2008b) solves this by temporarily penalizing nodes that are currently being evaluated by other threads:

When thread $t$ selects node $(s, a)$ for traversal, it immediately applies:
- $N(s, a) \leftarrow N(s, a) + n_{\text{vl}}$
- $W(s, a) \leftarrow W(s, a) - n_{\text{vl}}$

This makes the node appear worse (lower $Q$ and reduced exploration bonus), deterring other threads from selecting the same path. When the evaluation completes, the virtual loss is reversed and the actual value is added:
- $N(s, a) \leftarrow N(s, a) - n_{\text{vl}} + 1$
- $W(s, a) \leftarrow W(s, a) + n_{\text{vl}} + v$

AlphaZero uses $n_{\text{vl}} = 1$ (Silver et al., 2018). With virtual loss, multiple threads can traverse the tree concurrently with minimal lock contention (only atomic increments needed). The trade-off: virtual loss introduces a pessimistic bias that may reduce search quality when the number of concurrent threads is very large relative to the tree size, since many paths are simultaneously penalized (Segal, 2010).

### 1.4 Dirichlet Noise at Root

To ensure exploration during self-play training, AlphaZero adds Dirichlet noise to the root prior:

$$
\pi'(a|s_{\text{root}}) = (1 - \epsilon) \cdot \pi_\theta(a|s_{\text{root}}) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)
$$

where $\epsilon = 0.25$ and $\alpha$ is inversely scaled with the number of legal moves. For Go ($\sim 361$ moves), $\alpha = 0.03$; for chess ($\sim 30$ moves), $\alpha = 0.3$; the heuristic is $\alpha \approx 10/n$ where $n$ is the typical number of legal moves (Silver et al., 2018). The Dirichlet distribution concentrates mass on a few actions when $\alpha$ is small, producing "spiky" noise that encourages deep exploration of a few alternatives rather than shallow sampling of many.

Noise is applied **only at the root** and **only during training** (self-play). During competitive play, no noise is added.

---

## 2. MCTS for Multi-Move Turns

### 2.1 The Core Challenge

Standard MCTS assumes a strict alternation of players: each tree level corresponds to one player making one move. Games with multi-move turns -- where a player places $k > 1$ pieces before the turn passes -- break this assumption. In a game where each player places 2 pieces per turn, a single "turn" encompasses 2 sequential decisions by the same player before the opponent acts.

This creates several structural problems:

1. **Tree depth vs. game depth.** A game lasting $T$ turns has $2T$ individual moves (for $k=2$), doubling the tree depth if each sub-move gets its own level.
2. **Branching factor interaction.** If there are $n$ available cells, the first sub-move has $n$ options and the second has $n-1$ options, yielding $n(n-1)$ possible turn actions -- a quadratic blowup.
3. **Evaluation granularity.** The position after only the first sub-move of a turn is an "intermediate" state that never occurs in actual play (the opponent never faces it). Should the value network evaluate these states? Are they meaningful?
4. **Transpositions.** Placing piece at cell $x$ then cell $y$ yields the same board as placing at $y$ then $x$. Without deduplication, the tree wastes half its budget on redundant states.

### 2.2 Approach 1: Interleaved Sub-Move Levels

The most straightforward approach treats each sub-move as a separate tree level, with the turn structure encoded in the node metadata:

```
Level 0: Root (Player A to make sub-move 1)
Level 1: Player A sub-move 1 chosen  (Player A to make sub-move 2)
Level 2: Player A sub-move 2 chosen  (Player B to make sub-move 1)
Level 3: Player B sub-move 1 chosen  (Player B to make sub-move 2)
Level 4: Player B sub-move 2 chosen  (Player A to make sub-move 1)
...
```

**Implementation.** The tree node tracks `(board_state, current_player, sub_move_index)`. During selection and backpropagation, the value is only negated at player transitions (between levels 2 and 3, 4 and 5, etc.), not between sub-moves of the same player. The PUCT formula uses the same player's perspective for consecutive sub-move levels.

```
function BACKPROPAGATE(node, value):
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        if node.parent is not None:
            if node.parent.current_player != node.current_player:
                value = -value  # only negate at player transitions
        node = node.parent
```

**Advantages.** (i) Standard MCTS machinery applies with minor modifications. (ii) The policy network can output a distribution over single cells, which is simpler to train. (iii) The tree explores the first sub-move's consequences before committing to a second sub-move, enabling adaptive second-move selection.

**Disadvantages.** (i) The tree is twice as deep for the same number of game turns, requiring more simulations to reach the same effective search depth. (ii) Intermediate states (after only the first sub-move) are evaluated by the value network, but these states may be hard to evaluate since they represent "incomplete turns." (iii) Without transposition handling, the tree explores $(x, y)$ and $(y, x)$ as separate branches.

This approach is widely used in Connect6 programs (Wu et al., 2010; Huang et al., 2013). NCTU6, the 2006--2013 Connect6 champion program, used this interleaved approach with domain-specific enhancements (Huang et al., 2013).

### 2.3 Approach 2: Composite Move Actions

Treat the entire turn as a single atomic action. For a 2-move turn on a board with $n$ available cells, each node has up to $\binom{n}{2}$ children (unordered pairs, since order doesn't matter on the board).

```
Level 0: Root (Player A to place 2 stones)
Level 1: Player A has placed (x1, y1) and (x2, y2)  (Player B to place 2 stones)
Level 2: Player B has placed ...
...
```

**Branching factor analysis.** For a board with $n$ empty cells:
- Composite branching factor: $\binom{n}{2} = \frac{n(n-1)}{2}$
- At $n = 50$: $\binom{50}{2} = 1225$ actions per node
- At $n = 100$: $\binom{100}{2} = 4950$ actions per node
- At $n = 200$: $\binom{200}{2} = 19900$ actions per node

Compare with Go's branching factor of $\sim 250$ and chess's $\sim 35$. The composite approach is feasible for small boards but becomes intractable for larger ones.

**Advantages.** (i) Each tree level corresponds to a complete turn -- the value network always evaluates "real" positions. (ii) No transposition problem within a turn (pairs are unordered by construction). (iii) Half the tree depth compared to Approach 1.

**Disadvantages.** (i) The policy network must output a distribution over pairs, which is a much larger output space -- $O(n^2)$ vs. $O(n)$. This can be factored as $\pi(a_1, a_2 | s) = \pi(a_1 | s) \cdot \pi(a_2 | s, a_1)$ using an autoregressive decomposition, but this requires two forward passes or architectural changes. (ii) The massive branching factor means most children receive very few visits, degrading search quality unless paired with aggressive pruning or progressive widening.

### 2.4 Approach 3: Sequential Policy with Shared Search

A hybrid approach: use the interleaved tree structure (Approach 1) but employ a **dual-head policy network** that predicts both sub-moves, enabling aggressive pruning at both levels.

The policy network architecture:
- Head 1: $\pi_1(a_1 | s)$ -- distribution over first sub-move candidates
- Head 2: $\pi_2(a_2 | s, a_1)$ -- distribution over second sub-move, conditioned on the first
- Value head: $v(s)$ -- position evaluation

During MCTS:
1. At a first-sub-move node, use $\pi_1$ as the prior. Only expand the top-$k_1$ candidates (where $k_1$ is chosen to cover, say, 95% of the prior mass).
2. At a second-sub-move node (child of a first-sub-move node), use $\pi_2$ conditioned on the parent's action as the prior. Only expand the top-$k_2$ candidates.
3. The effective branching factor becomes $k_1 \cdot k_2$ instead of $n(n-1)$.

**Pruning budget.** If the policy network is well-trained and we take $k_1 = 10, k_2 = 10$, the effective branching is $100$ per turn -- comparable to Go. Even $k_1 = 20, k_2 = 20$ yields $400$, still manageable.

```
function EXPAND_WITH_PRUNING(node, state):
    if state.sub_move_index == 0:
        priors = policy_head_1(state)
        top_k = select_top_k(priors, k1, min_prob=0.01)
    else:
        priors = policy_head_2(state, parent_action=node.parent.action)
        top_k = select_top_k(priors, k2, min_prob=0.01)

    for action, prior in top_k:
        child_state = state.apply(action)
        node.add_child(action, prior, child_state)
```

**Advantages.** (i) The interleaved tree enables adaptive search -- resources are concentrated on promising first-sub-moves. (ii) Policy-guided pruning keeps the effective branching manageable. (iii) The policy network still outputs per-cell distributions, which are tractable to train.

**Disadvantages.** (i) Requires a well-trained policy network; early in training, pruning may discard the best moves. (ii) Two-headed policy is more complex to train than a single head. (iii) Still has the intermediate-state evaluation problem.

This approach is closest to what modern AlphaZero-style systems would use for multi-move games, and is the approach we recommend (Section 7).

### 2.5 Approach 4: Nested MCTS

Run a separate, inner MCTS for the second sub-move within each first-sub-move expansion. This treats the two sub-moves as a hierarchical decision:

```
function OUTER_MCTS_ITERATION(root):
    # Select first sub-move via MCTS
    node = select_leaf(root)
    first_move = expand_one_child(node)

    # Run inner MCTS to find best second sub-move
    inner_state = node.state.apply(first_move)
    second_move = INNER_MCTS(inner_state, budget=inner_sims)

    # Evaluate the composite state
    composite_state = inner_state.apply(second_move)
    value = evaluate(composite_state)

    backpropagate(node, value)
```

**Analysis.** This approach was explored by Baier & Winands (2012) in the context of multi-action games. The inner MCTS budget must be chosen carefully -- too small and the second-move selection is noisy; too large and the total computation blows up multiplicatively ($N_{\text{outer}} \times N_{\text{inner}}$). In practice, this approach is inferior to Approaches 1 and 3 because the inner search doesn't benefit from the tree built by the outer search across iterations (Baier & Winands, 2012). Each inner MCTS starts from scratch, wasting computation.

The approach can be improved by caching inner search trees across outer iterations, effectively converging to Approach 1 with additional bookkeeping overhead.

### 2.6 Comparative Analysis

| Criterion | Interleaved (1) | Composite (2) | Sequential+Pruning (3) | Nested (4) |
|---|---|---|---|---|
| Branching factor per turn | $n + (n-1) = 2n-1$ (sum) | $\binom{n}{2}$ | $k_1 + k_2$ (sum) | $n$ (outer only) |
| Effective branching (multiplicative) | $n(n-1)$ | $n(n-1)/2$ | $k_1 \cdot k_2$ | $n \cdot n_{\text{inner}}$ |
| Tree depth per turn | 2 levels | 1 level | 2 levels | 1 level (outer) |
| Transposition handling needed | Yes | No (pairs unordered) | Yes | Partial |
| Policy network output | $O(n)$ per cell | $O(n^2)$ over pairs | $O(n)$ per head | $O(n)$ per cell |
| Value network evaluates intermediate states | Yes | No | Yes | No |
| Adaptivity (search refocuses on good first moves) | High | None | High | Medium |
| Implementation complexity | Low | Low | Medium | High |

For games with moderate $n$ (say $n < 100$ available cells in the search region) and a strong policy network, **Approach 3 (sequential with pruning) is recommended.** It combines the adaptivity benefits of the interleaved structure with the manageable branching of policy-guided pruning.

### 2.7 Connect6-Specific MCTS Literature

Connect6 (Wu & Huang, 2005) is the canonical multi-move connection game: players alternate placing 2 stones per turn (except the first player's first turn, which is 1 stone) on a 19x19 Go board, aiming to connect 6 in a row.

**NCTU6** (Huang et al., 2013) used an interleaved MCTS with several Connect6-specific enhancements:
- *Dependency-based search*: a threat-space search (analogous to proof-number search) that identifies forced wins through threat sequences. This is integrated into MCTS as a solver module.
- *Relevance zone*: moves are only considered within a zone around existing stones, drastically reducing the effective branching factor.
- *Pattern-based simulation policy*: rollouts prioritize moves that extend or block threats, rather than being purely random.

**Wu et al. (2010)** studied the impact of knowledge-based simulation policies in Connect6 MCTS. They found that incorporating local pattern knowledge into the simulation policy (Phase 3) improved playing strength by approximately 400 Elo over uniform random simulations.

**Huang & Muller (2013)** demonstrated that combining MCTS with threat-based pruning in Connect6 enables the search to prove wins in mid-game positions that pure MCTS would require orders of magnitude more simulations to find.

---

## 3. MCTS for Large/Infinite Action Spaces

### 3.1 The Problem

Standard MCTS expands all legal actions at a node (or adds one untried action per iteration). When the action space is very large or infinite, this is impossible -- either the branching factor overwhelms the search budget, or there is no finite set of "legal actions" to enumerate.

Examples:
- Go on 19x19: $\sim 361$ actions (large but finite, manageable)
- Connect6 composite moves on 19x19: $\sim 64{,}000$ pair-actions (very large)
- Infinite-board connection games: unbounded action space
- Continuous-action games (robotics, continuous control): infinite action space

### 3.2 Progressive Widening

Progressive widening (Coulom, 2007; Chaslot et al., 2008a) limits the number of children expanded at each node as a function of the node's visit count:

$$
|C(s)| = \lfloor k \cdot N(s)^\alpha \rfloor
$$

where $|C(s)|$ is the maximum number of children, $N(s)$ is the visit count, and $k, \alpha$ are hyperparameters. Typically $\alpha \in [0.25, 0.5]$, meaning the number of children grows sub-linearly with visits.

**Mechanism.** When a node is visited and $|C(s)| < k \cdot N(s)^\alpha$, a new child is added (sampled from a prior distribution or chosen by some heuristic). Otherwise, selection proceeds among existing children.

```
function SELECT_OR_EXPAND(node):
    max_children = floor(k * node.visit_count^alpha)

    if len(node.children) < max_children:
        # Expand: add a new child
        action = sample_new_action(node)  # from policy prior or uniform
        child = node.add_child(action)
        return child
    else:
        # Select: choose among existing children via PUCT
        return argmax_child(node, PUCT)
```

**For infinite boards:** progressive widening is essential. The prior distribution from which new actions are sampled can be the policy network's output (a spatial distribution over a cropped region of the board), or a heuristic distribution that concentrates mass near existing stones.

Couetoux et al. (2011) proved that MCTS with progressive widening converges to the optimal policy in continuous action spaces, provided the sampling distribution has support over the optimal action and $\alpha < 1$.

### 3.3 Policy-Guided Expansion

In AlphaZero-style MCTS, the policy network $\pi_\theta$ provides priors over all legal actions. For large action spaces, **only actions above a probability threshold are expanded**:

$$
\mathcal{A}_{\text{expand}}(s) = \{a : \pi_\theta(a|s) > \tau\} \cup \text{top-}k(\pi_\theta(\cdot|s))
$$

The threshold $\tau$ and minimum $k$ ensure that (i) only plausible moves are considered, and (ii) at least $k$ moves are always available for exploration. KataGo (Wu, 2019) uses a variant of this: legal actions are sorted by policy prior, and only the top ones are expanded initially, with more added as the node accumulates visits (combining policy guidance with progressive widening).

The critical insight: **with a strong policy network, the effective branching factor is determined by the policy's entropy, not by the number of legal moves.** If the policy concentrates 95% of its mass on 15 moves, the MCTS effectively has a branching factor of ~15 regardless of whether there are 100 or 10,000 legal moves.

### 3.4 Sampled MCTS (Sampled MuZero)

Hubert et al. (2021) introduced Sampled MuZero, which extends MuZero to large and continuous action spaces by sampling a subset of actions at each node:

1. At each node, sample $K$ actions from a proposal distribution $\pi_0$ (e.g., the policy network or a mixture of the policy and uniform noise).
2. Run MCTS only over these $K$ sampled actions.
3. The improved policy target for training is computed as the MCTS visit distribution over the $K$ sampled actions, re-weighted by importance sampling to correct for the sampling bias.

The policy improvement step becomes:
$$
\pi_{\text{target}}(a|s) \propto \frac{N(s, a)}{\pi_0(a|s)} \quad \text{for } a \in \text{sampled set}
$$

This ensures that the policy improvement is unbiased even though only a subset of actions was searched. Sampled MuZero achieved superhuman performance in several continuous-action domains where standard MCTS is inapplicable.

For infinite-board games, a natural proposal distribution is the policy network restricted to a spatial window around the "active" region, with a small uniform component covering the rest to allow long-range exploration.

### 3.5 Dynamic Board Cropping

For infinite or very large boards, a practical approach is to restrict the MCTS to a dynamically defined region:

**Zone of Interest (ZoI):** Define a rectangular (or hexagonal) bounding box around all placed stones, expanded by a margin $m$:

$$
\text{ZoI} = \text{BoundingBox}(\text{all stones}) + m
$$

Only cells within the ZoI are considered as legal moves in the MCTS. The margin $m$ controls the trade-off between search focus and the possibility of distant strategic plays.

```
function COMPUTE_ZONE_OF_INTEREST(board, margin):
    if board.is_empty():
        return centered_region(radius=margin)

    min_q, max_q = extremes of q-coordinates of all stones
    min_r, max_r = extremes of r-coordinates of all stones

    return HexRegion(
        q_range = [min_q - margin, max_q + margin],
        r_range = [min_r - margin, max_r + margin]
    )
```

**Adaptive margin.** The margin can be adapted based on game phase:
- Early game (few stones): larger margin ($m = 3$--$5$) to allow creative opening play
- Mid/late game (many stones, threats active): smaller margin ($m = 1$--$2$) since play is concentrated

**Multi-resolution search.** An advanced variant maintains two zones:
1. A "tactical zone" (small margin) with full MCTS search
2. A "strategic zone" (large margin) with progressive widening or limited-budget search

This mirrors how human players think -- detailed calculation in the immediate area, broader strategic assessment further out.

The ZoI approach was used in MoHex (Arneson et al., 2010; Huang et al., 2013) for standard-board Hex, and extended to larger boards by Pawlewicz & Hayward (2015).

### 3.6 Combining Approaches

In practice, large-action-space MCTS uses multiple techniques simultaneously:

1. **Zone of Interest** defines the candidate set ($n_{\text{ZoI}}$ cells)
2. **Policy network** assigns priors over the ZoI cells
3. **Policy-guided expansion** initially expands only high-prior moves ($k_{\text{init}}$ children)
4. **Progressive widening** adds more children as visits accumulate

This pipeline reduces the effective action space from infinity $\to$ $n_{\text{ZoI}}$ $\to$ $k_{\text{init}}$ $\to$ grows slowly with visits.

---

## 4. Transposition Tables and DAGs in MCTS

### 4.1 Why Transpositions Matter for Multi-Move Games

In a game where Player A places two stones per turn, placing at $(x, y)$ in that order produces the same board state as placing at $(y, x)$. With $n$ candidates for the first sub-move and $n-1$ for the second, there are $n(n-1)$ ordered sequences but only $\binom{n}{2} = n(n-1)/2$ unique board states. **Exactly half of all paths in the interleaved tree are transpositions.**

Without transposition detection, the MCTS explores each pair twice, effectively halving its simulation budget. For a search budget of $N$ simulations, only $N/2$ reach unique states. This is a severe penalty, and transposition handling is not optional -- it is essential.

### 4.2 Converting the MCTS Tree to a DAG

A transposition table (TT) maps board states (via hash) to MCTS nodes. When expanding a child, the TT is consulted; if the resulting state already exists, the child pointer is redirected to the existing node instead of creating a new one. The tree becomes a directed acyclic graph (DAG).

```
function EXPAND_WITH_TT(node, action, transposition_table):
    child_state = node.state.apply(action)
    state_hash = zobrist_hash(child_state)

    if state_hash in transposition_table:
        existing_node = transposition_table[state_hash]
        node.children[action] = existing_node
        existing_node.parents.append(node)  # DAG: multiple parents
        return existing_node
    else:
        new_node = Node(state=child_state)
        transposition_table[state_hash] = new_node
        node.children[action] = new_node
        new_node.parents.append(node)
        return new_node
```

**Zobrist hashing** (Zobrist, 1970) is the standard method for board state hashing. Assign a random 64-bit integer $Z[c][p]$ to each cell $c$ and player $p$. The hash of a position is $H = \bigoplus_{(c,p) \in \text{stones}} Z[c][p]$, computed incrementally as stones are added ($H' = H \oplus Z[c_{\text{new}}][p_{\text{new}}]$). For infinite boards, $Z$ values can be computed lazily via a hash function seeded on coordinates: $Z[c][p] = \text{hash}(c.q, c.r, p)$.

### 4.3 Challenges: Backpropagation in DAGs

In a tree, each node has exactly one parent, so backpropagation follows a unique path to the root. In a DAG, a node may have multiple parents. This raises the question: when backpropagating a value from a leaf, which parents should receive the update?

**Approach A: All-parents backpropagation.** Propagate the value to all parents of the current node, then to all grandparents, etc. This ensures all paths through the node benefit from the simulation. However, it can lead to overcounting -- a single simulation updates many nodes, inflating visit counts and potentially skewing $Q$ values. Kishimoto & Muller (2005) analyze this overcounting problem.

**Approach B: Single-path backpropagation.** Only propagate along the path that was traversed during selection (the path from root to leaf that was actually followed). Parents that weren't on the selection path don't receive the update. This preserves the tree-MCTS semantics exactly but reduces the benefit of transposition detection (some shared nodes don't benefit from cross-path information).

**Approach C: Weighted backpropagation.** When propagating to multiple parents, weight the update by the fraction of traffic each parent contributes. If parent $p_i$ has sent $n_i$ selections through this node, weight the update to $p_i$ by $n_i / \sum_j n_j$. This is a principled compromise but adds implementation complexity.

**Practical recommendation.** For the multi-move case where the primary transpositions are sub-move reorderings, **single-path backpropagation (Approach B)** is simplest and sufficient. The transposition table still helps by avoiding redundant node creation and neural network evaluations. Over many simulations, both orderings of a pair will be visited roughly equally, and their shared node accumulates statistics from both paths.

### 4.4 Virtual Loss in DAGs

Virtual loss in DAGs requires care. If a node has multiple parents and thread $T_1$ is traversing through parent $P_1$ while thread $T_2$ is traversing through parent $P_2$, both will apply virtual loss to the shared child. This is actually correct behavior -- it discourages a third thread from also visiting this node, regardless of the path taken. The key invariant: virtual loss is applied/reversed per traversal, not per parent. Use atomic operations on the node's visit count and value sum.

### 4.5 Canonical Ordering for Move Pairs

For multi-move transpositions specifically, an alternative to full DAG conversion is **canonical ordering**: always store move pairs in a canonical order (e.g., the sub-move with the smaller coordinate index comes first). During expansion, if a first-sub-move $a_1$ is selected and then a second-sub-move $a_2 < a_1$ is considered, the expansion lookup checks the TT for the canonical pair $(a_2, a_1)$.

This can be implemented without converting the tree to a DAG by using the TT only to share neural network evaluations (avoiding redundant forward passes) while keeping the tree structure intact.

```
function CANONICAL_PAIR(move_a, move_b):
    if move_a.index <= move_b.index:
        return (move_a, move_b)
    else:
        return (move_b, move_a)

function GET_OR_COMPUTE_EVALUATION(state, move1, move2, eval_cache):
    canonical = CANONICAL_PAIR(move1, move2)
    cache_key = hash(state.before_turn, canonical)

    if cache_key in eval_cache:
        return eval_cache[cache_key]

    result_state = state.apply(move1).apply(move2)
    value = value_network(result_state)
    eval_cache[cache_key] = value
    return value
```

---

## 5. Progressive Strategies and Early Termination

### 5.1 Solver Integration in MCTS

Standard MCTS converges to the optimal policy asymptotically but never "proves" a result. **Solver-enhanced MCTS** (Winands et al., 2008) integrates exact game-theoretic solving into the MCTS framework:

When a terminal node is reached (win/loss/draw), the result is propagated upward with special semantics:
- If all children of a MAX node are proven losses, the MAX node is a proven loss.
- If any child of a MAX node is a proven win, the MAX node is a proven win.
- If all children of a MIN node are proven wins, the MIN node is a proven win.
- If any child of a MIN node is a proven loss, the MIN node is a proven loss.

Proven nodes are excluded from further search (or given infinite/zero value to redirect search elsewhere). This is particularly effective in the endgame, where MCTS can exactly solve subtrees.

```
function BACKPROPAGATE_WITH_SOLVER(node, value, is_proven):
    while node is not None:
        node.visit_count += 1
        node.total_value += value

        if is_proven:
            node.proven_value = value  # +1 (win) or -1 (loss)
            # Check if parent is now proven
            if node.parent is not None:
                check_parent_proven(node.parent)

        value = -value
        node = node.parent

function CHECK_PARENT_PROVEN(parent):
    if parent.is_max_node:
        if any(child.proven_value == +1 for child in parent.children):
            parent.proven_value = +1  # proven win
        elif all(child.proven_value == -1 for child in parent.children):
            parent.proven_value = -1  # proven loss
    else:  # min node
        if any(child.proven_value == -1 for child in parent.children):
            parent.proven_value = -1  # proven loss (for max player)
        elif all(child.proven_value == +1 for child in parent.children):
            parent.proven_value = +1
```

### 5.2 Threat-Space Search for Connection Games

Connection games (Hex, Havannah, Y, Connect6) have a special structure: the goal is to form a connected path between designated regions. This enables **threat-space search** (Allis et al., 1994; Hayward et al., 2003), which identifies forcing sequences where one player makes threats that the opponent must answer.

A **threat** is a move (or set of moves) that, if unanswered, immediately wins. In Hex, a "virtual connection" (VC) between two cells means the current player can connect them regardless of the opponent's responses (Anshelevich, 2002). **H-search** (Anshelevich, 2002) and its successors compute virtual connections:

- **Full VC**: player can connect cells $a$ and $b$ even if the opponent moves first.
- **Semi-VC**: player can connect cells $a$ and $b$ if the player moves first.

These can be composed: if semi-VCs share a "carrier" (the set of cells needed), they can be combined into full VCs. The resulting connection analysis can prove wins or losses without exhaustive search.

**Integration with MCTS:** Threat-space search can be invoked:
1. **At leaf evaluation**: instead of (or in addition to) calling the value network, run a quick threat-space search. If a forced win/loss is found, return the proven result.
2. **As a simulation policy**: during rollouts, prioritize moves identified as threats or threat-responses.
3. **As a pruning mechanism**: if a branch has a proven threat-based refutation, prune it from the search.

MoHex (Arneson et al., 2010; Huang et al., 2013) and subsequent Hex programs demonstrate that combining MCTS with virtual connection computation dramatically improves playing strength, especially in the endgame where forcing sequences are common.

For multi-move connection games, threats are even more powerful: placing two stones per turn enables "double threats" where both sub-moves create independent threats that the opponent cannot both answer. Identifying such patterns can guide the policy network and simulation policy.

### 5.3 Pattern-Based Pruning for k-in-a-Row Games

In games where the objective involves forming a line of $k$ consecutive pieces, local patterns provide strong heuristics:

- **Threat patterns**: open-four (four in a row with both ends open), half-open four, open three, etc. Each has a known urgency level.
- **Defensive patterns**: opponent's threats that must be blocked immediately.
- **Dead patterns**: cells that cannot contribute to any winning line (e.g., surrounded by opponent stones).

These patterns can be used to:
1. **Prune the action space**: remove "dead" cells from consideration.
2. **Order moves**: prioritize threat moves and defensive moves in the selection policy.
3. **Detect forced wins/losses**: if a player has an unstoppable double-threat, the position is won.

For hexagonal grids, patterns adapt to 6 directions (3 axes) instead of 4 (2 axes for orthogonal + 2 diagonals). The basic pattern vocabulary is similar but the combinatorics differ.

**Relevance Zone (RZ) pruning** (Hayward & van Rijswijck, 2006) restricts the search to cells that are "relevant" -- connected to existing stones or within the influence region of active threats. This can reduce the effective action space by 80--90% in mid-game positions.

---

## 6. Scaling MCTS: Simulation Budget and Parallelism

### 6.1 How Many Simulations Are Needed?

The required simulation budget depends on:

1. **Effective branching factor $b$.** MCTS visits scale roughly as $b^d$ for depth $d$, so reducing $b$ (via policy pruning) has an exponential effect on required simulations.

2. **Game complexity.** More complex games (larger state spaces, deeper game trees) require more simulations for the same quality of play. Empirically:
   - Tic-tac-toe: $\sim 10^2$ simulations suffice (trivial game)
   - Connect4: $\sim 10^3$--$10^4$ simulations for strong play
   - Chess (AlphaZero): 800 simulations per move during self-play, 10$^5$+ during evaluation
   - Go (AlphaZero): 800 simulations per move during self-play, 1600 during evaluation
   - Go (AlphaGo Master): $\sim 10^5$ simulations per move (but with a weaker network)

3. **Neural network quality.** A stronger policy network reduces the effective branching factor (concentrating prior mass on fewer moves), and a stronger value network provides more accurate leaf evaluations, both reducing the required simulation count. Silver et al. (2017) showed that AlphaGo Zero with 1600 simulations surpassed AlphaGo Lee with $10^5$ simulations, entirely due to the improved neural network.

4. **Position sharpness.** Quiet positions (many moves of similar value) require fewer simulations than sharp positions (one correct move, all others losing). MCTS naturally allocates more time to sharp positions due to the value spread.

**Rule of thumb for a new game:** Start with 400--800 simulations per move during self-play training. Once the network is strong, even 100--200 simulations may suffice for most positions. During evaluation or competitive play, use the maximum budget affordable within time constraints.

### 6.2 Parallelization Strategies

Three main approaches exist for parallel MCTS (Chaslot et al., 2008b; Segal, 2010):

**Root Parallelization.** Run $P$ independent MCTS trees from the same root, each with $N/P$ simulations. Merge the results by summing visit counts across trees. Simple to implement (no synchronization needed), but suboptimal -- the trees don't share information, so effective simulation count is less than $N$.

```
function ROOT_PARALLEL_MCTS(state, N, P):
    trees = []
    parallel for i = 1 to P:
        tree = MCTS_SEARCH(state, N / P)
        trees.append(tree)

    # Merge visit counts at root children
    merged = {}
    for tree in trees:
        for action, child in tree.root.children:
            merged[action].visits += child.visits
            merged[action].total_value += child.total_value

    return argmax(merged, key=lambda a: merged[a].visits)
```

**Leaf Parallelization.** From a single tree, select $P$ leaves simultaneously (using virtual loss to diversify paths), evaluate them in a batch on the GPU, then backpropagate all results. This is the preferred approach for GPU-based systems because:
- Neural network evaluation is the bottleneck (GPU inference)
- GPUs are most efficient with large batches
- Virtual loss ensures the $P$ leaves are diverse

AlphaZero uses this approach: 8 threads select leaves in parallel, evaluations are batched on TPUs, and results are backpropagated asynchronously (Silver et al., 2018).

```
function BATCH_MCTS(root, num_simulations, batch_size):
    for batch_start = 0 to num_simulations step batch_size:
        leaves = []
        paths = []

        # Select batch_size leaves with virtual loss
        for i = 1 to batch_size:
            path, leaf = SELECT_WITH_VIRTUAL_LOSS(root)
            leaves.append(leaf)
            paths.append(path)

        # Batch evaluate on GPU
        values, policies = neural_network.batch_evaluate(
            [leaf.state for leaf in leaves]
        )

        # Expand and backpropagate
        for i = 1 to batch_size:
            EXPAND(leaves[i], policies[i])
            REMOVE_VIRTUAL_LOSS(paths[i])
            BACKPROPAGATE(paths[i], values[i])
```

**Tree Parallelization.** Multiple threads share a single tree with fine-grained locking (or lock-free operations). Each thread independently runs select-expand-evaluate-backpropagate, using virtual loss for diversity. This is the most general approach and achieves the best scaling, but requires careful concurrent data structure design.

In practice, leaf parallelization is most common because it naturally matches GPU batch requirements.

### 6.3 Batch Neural Network Evaluation for GPU Efficiency

Neural network inference on GPUs is highly parallel -- evaluating 1 state takes almost the same time as evaluating 64 or 128 states due to GPU saturation. This makes batching essential:

- **Minimum batch size**: 8--32 for GPU utilization
- **Optimal batch size**: 64--256 for modern GPUs (RTX 3090, A100)
- **Maximum useful batch size**: beyond 256--512, latency increases without throughput gains

The simulation-to-evaluation ratio determines the effective batch size. If MCTS runs 800 simulations with a batch size of 32, that's 25 batch evaluations per move. At 5ms per evaluation, that's 125ms per move -- fast enough for real-time play.

**Asynchronous evaluation pipeline:**
```
# Producer threads (MCTS)
for each thread:
    loop:
        leaf = select_leaf_with_virtual_loss(root)
        evaluation_queue.push(leaf)
        result = leaf.result_future.wait()
        expand_and_backpropagate(leaf, result)

# Consumer thread (GPU)
loop:
    batch = evaluation_queue.pop_batch(max_size=BATCH_SIZE, timeout=1ms)
    results = neural_network.evaluate_batch(batch)
    for leaf, result in zip(batch, results):
        leaf.result_future.set(result)
```

### 6.4 Simulation Budget Allocation

Not all moves deserve the same search effort. **Pondering** (thinking on the opponent's time) and **dynamic allocation** improve overall playing strength:

- **Time management**: allocate more simulations to critical moves (when the top two actions have similar visit counts, indicating uncertainty). AlphaZero uses a simple heuristic: if the most-visited action after $N_{\text{min}}$ simulations has more than 50% of total visits, stop early (Silver et al., 2018).
- **Pondering**: during the opponent's turn, continue MCTS from the expected opponent response. If the opponent plays the expected move, the search tree is reused. Otherwise, the tree is discarded (or the relevant subtree is extracted).
- **Move-dependent allocation**: opening moves (high branching, low urgency) may get fewer simulations than mid-game moves (complex tactics, high urgency).

---

## MCTS Design for Infinite Hex Tic-Tac-Toe

This section provides concrete architectural recommendations for applying MCTS to Infinite Hex Tic-Tac-Toe -- a two-player game on an infinite hexagonal grid where each player places 2 stones per turn (except the first player's first turn: 1 stone), and the objective is to connect 4 in a row along any of the three hexagonal axes.

### 7.1 Recommended Tree Structure

**Use Approach 3: Interleaved sub-move levels with policy-guided pruning.**

Rationale:
- The interleaved structure allows the search to be adaptive: after exploring several first-sub-moves and their consequences, the search naturally concentrates on the most promising first moves.
- Policy-guided pruning keeps the effective branching factor manageable.
- The policy network outputs a per-cell distribution (not per-pair), which is feasible to train from the start.

**Tree structure:**
```
Root (Player A, sub-move 1)
├── Cell (3,1) → Node (Player A, sub-move 2)
│   ├── Cell (4,2) → Node (Player B, sub-move 1)  [Board: A at (3,1),(4,2)]
│   ├── Cell (2,0) → Node (Player B, sub-move 1)  [Board: A at (3,1),(2,0)]
│   └── ...
├── Cell (4,2) → Node (Player A, sub-move 2)
│   ├── Cell (3,1) → Node (Player B, sub-move 1)  [TRANSPOSITION: same as above]
│   └── ...
└── ...
```

**Backpropagation rule:** Negate the value only at player transitions (sub-move 2 $\to$ opponent's sub-move 1). Within a player's two sub-moves, the value sign is preserved.

**Branching factor analysis for Infinite Hex TTT:**

Let $n$ be the number of cells in the Zone of Interest. With policy-guided pruning:

| Phase | $n$ (ZoI cells) | Raw branching per turn | With pruning ($k_1 = k_2 = 15$) | Effective per-level |
|---|---|---|---|---|
| Early game (turn 2--5) | 20--40 | 380--1560 pairs | 225 | $\sim 15$ per sub-move |
| Mid game (turn 5--15) | 40--80 | 1560--6320 pairs | 225 | $\sim 15$ per sub-move |
| Late game (turn 15+) | 60--120 | 3540--14280 pairs | 225 | $\sim 15$ per sub-move |

With $k_1 = k_2 = 15$, the effective branching factor per turn is $15 \times 15 = 225$. Since transposition detection halves the unique states, the effective unique branching is $\sim 113$ per turn. This is comparable to Go ($\sim 250$ moves per turn) and well within the capability of neural-guided MCTS with 400--800 simulations.

As the policy network improves through training, $k_1$ and $k_2$ can be reduced to $\sim 8$--$10$, bringing the effective branching to $\sim 40$--$50$ unique pairs per turn.

### 7.2 Handling the Infinite Board

**Zone of Interest (ZoI) with adaptive margin:**

```python
def compute_zone_of_interest(board_state, margin=3):
    """
    Returns the set of hex cells eligible for MCTS expansion.
    Uses axial coordinates (q, r) for the hex grid.
    """
    if board_state.is_empty():
        # First move: center cell and immediate neighbors
        return hex_neighbors(origin=(0, 0), radius=margin)

    stones = board_state.all_stone_positions()
    min_q = min(s.q for s in stones) - margin
    max_q = max(s.q for s in stones) + margin
    min_r = min(s.r for s in stones) - margin
    max_r = max(s.r for s in stones) + margin

    # All hex cells within the bounding box
    candidates = set()
    for q in range(min_q, max_q + 1):
        for r in range(min_r, max_r + 1):
            if (q, r) not in board_state.occupied:
                candidates.add((q, r))

    return candidates
```

**Recommended margin values:**
- $m = 3$ for general play (captures moves within 3 hexes of any stone)
- $m = 2$ for faster search during self-play training
- $m = 4$ for analysis/evaluation when time is not constrained

**Why margin = 3:** In a connect-4 game on a hex grid, a winning line spans 4 cells. A stone 3 cells away from any existing stone can contribute to a winning line involving that stone. Margin 3 is therefore the minimum to guarantee no winning-relevant move is outside the ZoI (assuming at least one stone of the line is already placed). Margin 2 is acceptable during training since the policy network learns to occasionally place distant stones, and the ZoI will expand to include those positions on subsequent turns.

**Neural network input:** The policy and value networks take a fixed-size grid centered on the ZoI centroid. The input size should be at least $(2m + d_{\text{max}}) \times (2m + d_{\text{max}})$ where $d_{\text{max}}$ is the maximum span of stones in any axis. In practice, a $15 \times 15$ or $19 \times 19$ hex grid input (re-centered each turn) is sufficient for mid-game positions.

**Progressive widening as backup:** For positions where the ZoI is very large (e.g., stones are spread across a wide area), apply progressive widening within the ZoI:

$$
k_{\text{expand}}(N) = \min\left(|\text{ZoI}|, \; \lfloor 8 \cdot N^{0.4} \rfloor\right)
$$

This ensures that even with 200 ZoI cells, the search starts with $\sim 8$ children per node and grows to $\sim 50$ after 1000 visits.

### 7.3 Transposition Table Design for Move-Pair Symmetry

**Zobrist hashing for infinite hex grid:**

```python
import hashlib
import struct

class InfiniteZobristHash:
    """
    Lazy Zobrist hash for infinite hex grid.
    Z[q][r][player] is derived deterministically from coordinates,
    so no pre-allocated table is needed.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self._cache = {}

    def _get_z(self, q, r, player):
        key = (q, r, player)
        if key not in self._cache:
            # Deterministic 64-bit hash from coordinates
            data = struct.pack('iii', q, r, player)
            h = hashlib.blake2b(data, digest_size=8,
                                key=struct.pack('i', self.seed)).digest()
            self._cache[key] = struct.unpack('Q', h)[0]
        return self._cache[key]

    def compute(self, board_state):
        h = 0
        for (q, r), player in board_state.stones.items():
            h ^= self._get_z(q, r, player)
        return h

    def update(self, current_hash, q, r, player):
        """Incremental update: XOR in/out a stone."""
        return current_hash ^ self._get_z(q, r, player)
```

**Transposition table for sub-move pairs:**

For the interleaved tree, transpositions occur when the same two stones are placed in different orders. The TT should detect this:

```python
class TranspositionTable:
    def __init__(self):
        self.table = {}  # hash -> MCTSNode

    def lookup_or_create(self, board_state, zobrist_hash):
        """
        Returns existing node if this board state was reached before
        (possibly via a different move order), else creates new node.
        """
        if zobrist_hash in self.table:
            existing = self.table[zobrist_hash]
            # Verify (hash collision check)
            if existing.board_state == board_state:
                return existing, True  # transposition found

        new_node = MCTSNode(board_state, zobrist_hash)
        self.table[zobrist_hash] = new_node
        return new_node, False
```

**Critical detail:** The Zobrist hash of the board state after both sub-moves is the same regardless of move order (XOR is commutative). The TT lookup at the end of a two-sub-move sequence will always find the transposition. At the intermediate level (after only the first sub-move), there is no transposition -- the hash is different for different first moves -- which is correct, since the intermediate states have different search properties (different second-move candidates).

**Memory budget for TT:** Each TT entry stores a hash (8 bytes), visit count (4 bytes), total value (4 bytes), prior (4 bytes), and children pointers ($\sim 64$ bytes for 8 children). Approximately 100 bytes per entry. With 10M entries, the TT uses $\sim 1$ GB. This is sufficient for games of moderate length -- a 30-turn game with 800 simulations per move and $\sim 100$ unique expansions per simulation generates $\sim 2.4$M unique states over the entire game.

**TT clearing policy:** Clear between games. Within a game, reuse across moves: after the opponent moves, extract the subtree rooted at the opponent's move and discard the rest.

### 7.4 Recommended Simulation Budget and Parallelization

**Simulation budget:**

| Phase | Budget per move | Rationale |
|---|---|---|
| Self-play training (early) | 200--400 | Network is weak; more sims won't help much. Speed is critical for generating training data. |
| Self-play training (mature) | 400--800 | Stronger network benefits from more search. AlphaZero uses 800 for Go. |
| Evaluation / tournament | 800--1600 | Maximum quality for assessment. |
| Analysis mode | 1600--6400 | For humans studying positions. |

**Parallelization strategy: Batched leaf parallelization.**

```
Configuration:
- Search threads: 4 (for CPU-side tree traversal)
- GPU batch size: 32-64 (for neural network evaluation)
- Virtual loss: n_vl = 3 (slightly higher than AlphaZero's 1, to
  encourage more diverse leaf selection with 4 threads)
```

**Pipeline:**
1. 4 CPU threads concurrently select leaves (with virtual loss), each selecting 8--16 leaves before yielding.
2. Leaves are queued for GPU evaluation.
3. When the queue reaches the batch size (or a timeout of 1ms elapses), the batch is evaluated on the GPU.
4. Results are returned to the CPU threads, which expand nodes and backpropagate.

**Expected throughput** (on a single RTX 3080-class GPU with a moderately sized network):
- Neural network forward pass: $\sim 2$ms for a batch of 64
- CPU tree operations: $\sim 0.1$ms per simulation
- Effective throughput: $\sim 30{,}000$ simulations/second
- 800 simulations per move: $\sim 25$ms per move
- Self-play game ($\sim 30$ turns = 60 moves): $\sim 1.5$s per game

This means $\sim 2400$ self-play games per hour on a single GPU, or $\sim 57{,}000$ games per day. With 4 GPUs, this reaches $\sim 200{,}000$ games/day, sufficient for a strong training run over 1--2 weeks.

### 7.5 Integration with the Neural Network

**Network architecture requirements for MCTS integration:**

The neural network must provide two outputs for MCTS:
1. **Policy head**: $\pi_\theta(a | s)$ -- probability distribution over all cells in the input grid. This is used as the prior in PUCT. Cells outside the ZoI are masked to zero.
2. **Value head**: $v_\theta(s) \in [-1, +1]$ -- expected game outcome from the perspective of the player to move ($+1$ = win, $-1$ = loss, $0$ = draw).

**Policy as MCTS prior:**

```python
def get_mcts_priors(state, network, zoi_cells):
    """
    Get policy priors for MCTS expansion.
    Returns dict: cell -> prior probability (normalized over ZoI).
    """
    raw_logits = network.policy_forward(state)  # shape: (H, W)

    # Mask: only ZoI cells that are unoccupied
    mask = torch.zeros_like(raw_logits)
    for (q, r) in zoi_cells:
        grid_q, grid_r = world_to_grid(q, r, state.center)
        if 0 <= grid_q < H and 0 <= grid_r < W:
            mask[grid_q, grid_r] = 1.0

    masked_logits = raw_logits + (1 - mask) * (-1e9)
    priors = F.softmax(masked_logits.flatten(), dim=0)

    result = {}
    for (q, r) in zoi_cells:
        grid_q, grid_r = world_to_grid(q, r, state.center)
        result[(q, r)] = priors[grid_q * W + grid_r].item()

    return result
```

**Two-headed policy for sub-moves:**

For better MCTS performance with multi-move turns, the network can have two policy heads:

- **Head 1** ($\pi_1$): Distribution over first sub-move candidates given the current board.
- **Head 2** ($\pi_2$): Distribution over second sub-move candidates, conditioned on the first sub-move (which is encoded as an additional channel in the input).

During MCTS:
- At a first-sub-move node, use $\pi_1$ as the prior.
- At a second-sub-move node, re-run the network with the first sub-move applied (or use a cached representation with an additional input channel marking the first sub-move) to get $\pi_2$.

**Alternatively**, a single policy head can serve both sub-moves: the first sub-move uses $\pi$ on the current board; the second sub-move uses $\pi$ on the board with the first sub-move applied. This is simpler and the approach we recommend starting with. The network naturally learns that intermediate states (one stone placed this turn) are different from turn-boundary states because the stone-count parity differs.

**Value for evaluation:**

The value head is called at every leaf node, including intermediate states (after the first sub-move). This is correct: the value represents "expected outcome for the player to move, assuming optimal play from here." Even though intermediate states never occur in actual play (the current player will immediately make a second move), the value network should still output a meaningful evaluation -- the expected outcome if the current player makes an optimal second move followed by optimal play by both sides.

**Training target for the value head:** The actual game outcome from the position, as in standard AlphaZero. No special handling is needed for intermediate vs. turn-boundary states.

**Training target for the policy head:** The MCTS visit distribution at the node. For first-sub-move nodes, the target is the visit distribution over first sub-moves. For second-sub-move nodes, the target is the visit distribution over second sub-moves given the first. Both are standard MCTS policy targets.

### 7.6 Special Considerations for the Asymmetric First Move

The first player's first turn consists of only 1 stone (not 2). This is a standard balancing mechanism (also used in Connect6). MCTS must handle this:

**Tree structure for the first turn:**
```
Root (Player 1, sub-move 1, FIRST TURN)
├── Cell (0,0) → Node (Player 2, sub-move 1)  [Turn passes after 1 stone]
├── Cell (1,0) → Node (Player 2, sub-move 1)
└── ...
```

**Implementation:** The node metadata includes a `moves_remaining` counter. For the first turn, `moves_remaining = 1`. For all subsequent turns, `moves_remaining = 2`. When `moves_remaining` reaches 0, the turn passes to the opponent.

```python
class MCTSNode:
    def __init__(self, board, current_player, moves_remaining):
        self.board = board
        self.current_player = current_player
        self.moves_remaining = moves_remaining  # 1 or 2

    def apply_move(self, cell):
        new_board = self.board.place(cell, self.current_player)
        if self.moves_remaining == 1:
            # Turn ends, switch player
            return MCTSNode(new_board,
                          opponent(self.current_player),
                          moves_remaining=2)
        else:
            # Same player, one move left
            return MCTSNode(new_board,
                          self.current_player,
                          moves_remaining=1)
```

**Network input encoding for the first turn:** Include a binary feature plane indicating whether this is the first turn (or equivalently, whether the current player has 1 or 2 moves remaining). This helps the network distinguish first-turn positions from mid-turn positions.

**Dirichlet noise for the first move:** Since the first move has branching factor $n$ (not $n(n-1)/2$) and is strategically important (center cells have a strong advantage in hex games), use:
- $\alpha = 0.3$ for the first move (moderate noise, $\sim 10$--$20$ candidate cells)
- $\alpha = 10/n_{\text{ZoI}}$ for subsequent moves

**Opening symmetry:** On an infinite hex grid, the first move is always effectively the same due to translation invariance. The center of the coordinate system can be set to the first player's stone, making the first move trivially at $(0, 0)$. If the game implementation uses a canonical first move, the MCTS can skip searching the first move entirely.

For the second player's first turn (2 stones), the search is critical -- the response to the opening stone determines the game's trajectory. Allocate full simulation budget here.

---

## References

- Allis, L. V., van der Meulen, M., & van den Herik, H. J. (1994). Proof-number search. *Artificial Intelligence*, 66(1), 91--124.
- Anshelevich, V. V. (2002). A hierarchical approach to computer Hex. *Artificial Intelligence*, 134(1--2), 101--120.
- Arneson, B., Hayward, R. B., & Henderson, P. (2010). MoHex wins Hex tournament. *ICGA Journal*, 33(3), 180--183.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2--3), 235--256.
- Baier, H., & Winands, M. H. M. (2012). Nested Monte Carlo Tree Search for online planning in large MDPs. In *Proceedings of the 20th European Conference on Artificial Intelligence (ECAI)*.
- Chaslot, G., Winands, M. H. M., van den Herik, H. J., Uiterwijk, J., & Bouzy, B. (2008a). Progressive strategies for Monte-Carlo Tree Search. *New Mathematics and Natural Computation*, 4(3), 343--357.
- Chaslot, G., Winands, M. H. M., & van den Herik, H. J. (2008b). Parallel Monte-Carlo Tree Search. In *Proceedings of the 6th International Conference on Computers and Games (CG)*, pp. 60--71.
- Couetoux, A., Hoock, J.-B., Sokolovska, N., Teytaud, O., & Bonnard, N. (2011). Continuous upper confidence trees. In *International Conference on Learning and Intelligent Optimization (LION)*, pp. 433--445.
- Coulom, R. (2006). Efficient selectivity and backup operators in Monte-Carlo Tree Search. In *Proceedings of the 5th International Conference on Computers and Games (CG)*, pp. 72--83.
- Coulom, R. (2007). Computing Elo ratings of move patterns in the game of Go. *ICGA Journal*, 30(4), 198--208.
- Hayward, R. B., Bjornsson, Y., Johanson, M., Kan, M., Po, N., & van Rijswijck, J. (2003). Solving 7x7 Hex with domination, fill-in, and virtual connections. *Theoretical Computer Science*, 349(2), 123--139.
- Hayward, R. B., & van Rijswijck, J. (2006). Hex and combinatorics. *Discrete Mathematics*, 306(19--20), 2515--2528.
- Huang, S.-C., Arneson, B., Hayward, R. B., Muller, M., & Pawlewicz, J. (2013). MoHex 2.0: A pattern-based MCTS Hex player. In *Proceedings of the 8th International Conference on Computers and Games (CG)*.
- Huang, S.-C., & Muller, M. (2013). Investigating the limits of Monte-Carlo Tree Search methods in computer Hex. In *Proceedings of the 8th International Conference on Computers and Games (CG)*.
- Hubert, T., Schrittwieser, J., Antonoglou, I., Barekatain, M., Schmitt, S., & Silver, D. (2021). Learning and planning in complex action spaces. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*.
- Kishimoto, A., & Muller, M. (2005). A general solution to the graph history interaction problem. In *Proceedings of the 19th National Conference on Artificial Intelligence (AAAI)*, pp. 644--649.
- Kocsis, L., & Szepesvari, C. (2006). Bandit based Monte-Carlo planning. In *Proceedings of the 17th European Conference on Machine Learning (ECML)*, pp. 282--293.
- Pawlewicz, J., & Hayward, R. B. (2015). Feature strength and parallelization of sibling conspiracy number search. In *Advances in Computer Games (ACG)*.
- Segal, R. B. (2010). On the scalability of parallel UCT. In *Proceedings of the 7th International Conference on Computers and Games (CG)*, pp. 36--47.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354--359.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140--1144.
- Winands, M. H. M., Bjornsson, Y., & Saito, J.-T. (2008). Monte-Carlo Tree Search solver. In *Proceedings of the 6th International Conference on Computers and Games (CG)*, pp. 25--36.
- Wu, I.-C., & Huang, D.-Y. (2005). A new family of k-in-a-row games. In *Advances in Computer Games (ACG)*, pp. 180--194.
- Wu, I.-C., Huang, S.-J., & Chang, H.-C. (2010). NCTU6: The Connect6 program that won the Man-Machine Connect6 contest. *ICGA Journal*, 33(4), 230--233.
- Wu, D. J. (2019). Accelerating self-play learning in Go. *arXiv preprint arXiv:1902.10565*.
- Zobrist, A. L. (1970). A new hashing method with application for game playing. *Technical Report 88*, Computer Science Department, University of Wisconsin-Madison.

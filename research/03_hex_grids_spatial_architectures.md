# Neural Network Architectures for Non-Standard Board Geometries

**Focus: Hexagonal Grids, Infinite/Expanding Boards, and Connection Games**

---

## Table of Contents

1. [Hexagonal Grid Representations for Neural Networks](#1-hexagonal-grid-representations-for-neural-networks)
2. [Handling Infinite/Unbounded Boards](#2-handling-infiniteunbounded-boards)
3. [Neural Networks for Connection Games](#3-neural-networks-for-connection-games)
4. [Multi-Move Turn Architectures](#4-multi-move-turn-architectures)
5. [Positional Encoding for Board Games](#5-positional-encoding-for-board-games)
6. [Input Feature Planes for Board Games](#6-input-feature-planes-for-board-games)
7. [Recommended Architecture for Infinite Hex Tic-Tac-Toe](#recommended-architecture-for-infinite-hex-tic-tac-toe)

---

## 1. Hexagonal Grid Representations for Neural Networks

### 1.1 Coordinate Systems for Hexagonal Grids

Hexagonal grids admit several coordinate systems, each with distinct computational properties. The choice of coordinate system directly affects how convolution kernels are defined, how symmetries are exploited, and how the grid maps onto tensor memory layouts.

**Cube Coordinates.** Each hex cell is addressed by a triple `(x, y, z)` subject to the constraint `x + y + z = 0` (Patel, 2015). This embeds the 2D hexagonal lattice into 3D space, making distance calculations trivial: `d(a, b) = max(|a.x - b.x|, |a.y - b.y|, |a.z - b.z|)`. The six cardinal hex directions correspond to unit steps along pairs of axes: `(+1, -1, 0), (+1, 0, -1), (0, +1, -1)` and their negatives. Cube coordinates make rotational symmetry explicit -- a 60-degree rotation is a cyclic permutation of `(x, y, z)` with sign changes -- but the redundant third coordinate wastes memory when stored naively.

**Axial Coordinates.** By dropping the redundant coordinate `z = -x - y`, we obtain the two-component system `(q, r)` where `q = x` and `r = z` (or equivalently `r = y`, depending on convention). This is storage-efficient and preserves clean neighbor computation: the six neighbors of `(q, r)` are `{(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)}`. Axial coordinates are the standard choice in game engines (Red Blob Games, 2013) and are natural for neural network implementations because they map cleanly to 2D array indices.

**Offset Coordinates.** The hex grid is stored in a rectangular array where every other row (or column) is shifted by half a cell. This is the simplest scheme for rendering and is commonly used in game programming. However, offset coordinates break the uniformity of neighbor relationships: odd and even rows have different neighbor offset tables. This irregularity propagates into convolution definitions, making offset coordinates the least desirable for neural network architectures.

```
Axial layout (flat-top hexagons):        Offset layout (odd-r shift):
  (0,0) (1,0) (2,0) (3,0)                [0,0] [1,0] [2,0] [3,0]
    (0,1) (1,1) (2,1) (3,1)                 [0,1] [1,1] [2,1] [3,1]
  (0,2) (1,2) (2,2) (3,2)                [0,2] [1,2] [2,2] [3,2]

In axial, neighbor offsets are uniform.
In offset, neighbor offsets depend on row parity.
```

### 1.2 HexConv and Hexagonal Convolution Approaches

Standard 2D convolutions assume a square lattice where each pixel has 4 edge-adjacent and 4 corner-adjacent neighbors. On a hexagonal lattice, each cell has exactly 6 equidistant neighbors, and the notion of "kernel shape" changes fundamentally.

**HexConv (Hoogeboom et al., 2018).** Hoogeboom, Peters, Cohen, and Welling introduced hexagonal convolutions as part of their work on exploiting group equivariance. Their key insight is that a hexagonal convolution kernel of "radius" `k` covers a hexagonal region of `3k^2 + 3k + 1` cells (compared to `(2k+1)^2` for a square kernel). For `k=1`, this is 7 cells (center + 6 neighbors) versus 9 for a 3x3 square kernel. They implement hex convolutions by embedding the hexagonal grid in a square grid using the axial coordinate mapping and applying a masked convolution: a standard `(2k+1) x (2k+1)` square convolution with a hex-shaped binary mask that zeros out the corners. This approach is computationally efficient because it reuses existing optimized convolution routines (cuDNN) while faithfully capturing hexagonal topology.

**Hexagonal Convolutions for Imaging Cherenkov Telescopes (Steppa & Holch, 2019).** Steppa and Holch address hexagonal convolutions motivated by the hexagonal pixel grids in Imaging Atmospheric Cherenkov Telescopes (IACTs). They compare three strategies:

1. **Rebinning to square grid** -- interpolating hex data onto a square lattice, then using standard convolutions. Simple but introduces interpolation artifacts and destroys exact hex geometry.
2. **Axial addressing with masked kernels** -- equivalent to the Hoogeboom approach. Hex data is stored in a skewed rectangular array and convolutions use hex-shaped masks.
3. **Indexing-based hex convolution** -- explicitly constructing the neighbor index map for each cell and performing the convolution via gather operations. More flexible but slower due to irregular memory access.

They find that the axial-masked approach provides the best balance of accuracy and computational efficiency, consistent with the HexConv findings.

### 1.3 The "Brick Wall" Layout Trick

A widely used practical technique for adapting square convolutions to hex grids is the **brick wall** (or "staggered row") embedding. The hex grid is stored in a standard 2D array where odd rows are conceptually shifted by half a cell width:

```
Row 0:  [ A ] [ B ] [ C ] [ D ]
Row 1:     [ E ] [ F ] [ G ] [ H ]
Row 2:  [ I ] [ J ] [ K ] [ L ]
Row 3:     [ M ] [ N ] [ O ] [ P ]
```

In tensor storage, this is just a `H x W` array. The key insight is that a **3x3 square convolution on this layout** captures exactly the 6 hex neighbors plus the center, provided we correctly interpret which cells in the kernel are "real" neighbors and which are artifacts of the rectangular embedding. Specifically, for even rows, the relevant kernel positions (in row-major order of a 3x3 kernel) are `{(0,0), (0,1), (1,-1), (1,1), (2,0), (2,1)}` plus center; for odd rows, the offsets shift. A single learned 3x3 kernel therefore implicitly learns hex-topology filters, with two of the nine weights acting on "diagonal" positions that are not true hex neighbors.

In practice, many implementations simply use unmasked 3x3 convolutions on the brick-wall layout and let the network learn to effectively zero out the non-neighbor weights. Empirically this works well (Young et al., 2016), though it is less principled than explicit masking. The approach is especially attractive because it requires zero custom CUDA kernels and runs at full speed on standard deep learning frameworks.

### 1.4 Group Equivariant Convolutions on Hexagonal Lattices

**Group Equivariant CNNs (Cohen & Welling, 2016).** Cohen and Welling formalized the notion of equivariant neural networks with respect to discrete symmetry groups. A function `f` is equivariant to a group `G` if `f(g . x) = g . f(x)` for all `g` in `G`. Standard CNNs are equivariant to translations; G-CNNs extend this to include rotations and reflections. The square lattice admits `p4m` symmetry (4 rotations x 2 reflections = 8 elements). The hexagonal lattice admits `p6m` symmetry (6 rotations x 2 reflections = 12 elements).

**P6 and P6M Equivariant Hex Convolutions (Hoogeboom et al., 2018).** The hexagonal lattice's 6-fold rotational symmetry (`C6`) is richer than the square lattice's 4-fold symmetry (`C4`). Hoogeboom et al. construct convolution layers that are equivariant to the `p6` group (translations + 6 rotations) and the `p6m` group (translations + 6 rotations + reflections). Implementation proceeds in two steps:

1. **Kernel rotation:** A single "canonical" hex kernel is rotated into 6 (or 12) orientations by permuting the kernel weights according to the group action on hexagonal neighbor indices. For a radius-1 hex kernel (7 weights), a 60-degree rotation is a cyclic permutation of the 6 neighbor weights.
2. **Feature map stacking:** The output is a stack of feature maps, one per group element. Each layer operates on this stack, applying the rotated kernels across group elements. The group structure is preserved through the network.

The result is dramatic parameter efficiency: a `p6m`-equivariant network with 12 symmetries needs roughly `1/12` the parameters of a non-equivariant network to achieve the same representational capacity for symmetric tasks. For board games on hex grids, where the value of a position is invariant to rotation and reflection of the board, this is a natural fit.

**Practical considerations.** Group equivariant layers increase the number of output channels by a factor of `|G|` (6 or 12), which increases memory and computation. However, the reduced parameter count typically allows using fewer base channels while maintaining or improving accuracy. For a hex tic-tac-toe engine, `p6` equivariance (6 rotations, no reflection) is appropriate if the game rules are rotation-symmetric; `p6m` (with reflection) applies if the board has full dihedral symmetry.

### 1.5 Comparison: Hex-Native vs. Square Embedding vs. GNNs

| Approach | Fidelity to Hex Geometry | Computational Efficiency | Implementation Complexity | Symmetry Exploitation |
|---|---|---|---|---|
| **Hex-native convolution** (masked axial) | Exact | High (uses standard conv + mask) | Low-Medium | Natural `p6m` equivariance |
| **Brick-wall square embedding** | Approximate (2 spurious neighbors) | Highest (standard conv, no mask) | Lowest | Requires manual augmentation or custom equivariance |
| **Graph Neural Network** | Exact | Lower (sparse ops, no cuDNN) | Medium-High | Permutation equivariance; rotation equivariance requires careful construction |
| **Rebinning to square** | Lossy (interpolation artifacts) | High | Low | Only `p4m` symmetry native |

For fixed-size hex boards (as in the game of Hex), hex-native convolutions via axial masking are the clear winner: they are faithful to the geometry, computationally efficient, and naturally support `p6m` equivariance. For variable-size or infinite boards, GNNs become more attractive (see Section 2). The brick-wall trick is a pragmatic middle ground that trades geometric fidelity for simplicity.

---

## 2. Handling Infinite/Unbounded Boards

### 2.1 The Core Challenge

Standard convolutional neural networks require a fixed input tensor shape `(C, H, W)` determined at architecture definition time. An infinite or expanding board violates this assumption: the spatial extent of the game state grows as play progresses, and there is no natural bounding box determined a priori. This section surveys four approaches to this problem.

### 2.2 Approach 1: Dynamic Windowing

**Concept.** At each evaluation, compute the bounding box of all occupied cells, expand it by a fixed margin `m` (to allow the network to reason about moves near the frontier), and crop the board state to this window. The result is a fixed-protocol variable-size tensor, or alternatively a fixed-size tensor centered on the centroid of play with sufficient padding.

**Fixed-size variant.** Choose a maximum window size `W_max x W_max` (e.g., 19x19 or 25x25). Center the window on the centroid of all placed stones. If the occupied region exceeds the window, this approach fails gracefully by clipping -- but in practice, for games with local interaction (like hex tic-tac-toe where winning requires 4-in-a-row), play rarely spreads beyond a moderate radius.

**Advantages:**
- Reuses standard CNN architectures without modification.
- Computationally efficient with cuDNN-optimized convolutions.
- Translation invariance of convolutions is natural: the network learns local patterns that are position-independent.

**Disadvantages:**
- Hard boundary at the window edge: the network cannot reason about stones outside the window.
- Centroid computation and re-centering introduces a form of non-stationarity: the same board state may be presented with different alignments across evaluations if the centroid shifts.
- Requires choosing `W_max`, which is a hyperparameter that trades expressiveness for efficiency.

**Mitigation: Relative centering with border features.** Include a feature plane indicating distance to the window boundary, so the network can learn boundary-aware behavior. Alternatively, use a "focus" heuristic (e.g., center on the last move rather than the centroid) to stabilize the window.

### 2.3 Approach 2: Graph Neural Networks (GNNs)

**Concept.** Represent the board as a graph `G = (V, E)` where vertices are hex cells and edges connect hex-adjacent cells. Only occupied cells and their neighbors (up to some radius) are included. The graph grows dynamically as play progresses.

**Architecture.** A message-passing GNN (Gilmer et al., 2017) iteratively updates node representations:

```
h_v^{(t+1)} = UPDATE(h_v^{(t)}, AGGREGATE({h_u^{(t)} : u in N(v)}))
```

where `N(v)` is the set of neighbors of `v`, and `UPDATE` and `AGGREGATE` are learned functions (typically MLPs and sum/mean pooling). After `T` message-passing rounds, node representations are used for move prediction (per-node classification) and a global pooling produces a scalar value estimate.

**Advantages:**
- Naturally handles variable-size boards with no fixed tensor shape.
- Exact representation of hex topology: the adjacency structure is explicitly encoded.
- Can represent arbitrary board geometries (hex, square, irregular) with the same architecture.

**Disadvantages:**
- Sparse message-passing operations are less amenable to GPU optimization than dense convolutions. Libraries like PyTorch Geometric (Fey & Lenssen, 2019) and DGL (Wang et al., 2019) mitigate this but cannot match cuDNN throughput.
- Receptive field grows linearly with depth: `T` layers give a receptive field of radius `T`. For connection games requiring global reasoning across a large board, this may necessitate deep networks or explicit global aggregation mechanisms.
- No built-in translational equivariance (each node is treated independently given its features and local structure), though this can be an advantage for non-uniform boards.

**Hybrid approach: GNN on the active region, CNN on dense patches.** One can use a GNN for global structure and a CNN for dense local feature extraction, combining them via attention or concatenation. This is unexplored in the game AI literature but has analogs in point cloud processing (Qi et al., 2017).

### 2.4 Approach 3: Attention-Based Architectures (Transformers)

**Concept.** Represent the board state as a sequence of tokens, one per occupied cell (or one per cell in the active region). Each token encodes the cell's coordinate, occupancy, and any auxiliary features. A Transformer (Vaswani et al., 2017) processes this sequence with self-attention, enabling every cell to attend to every other cell regardless of spatial distance.

**Formulation.** Let `S = {(q_i, r_i, f_i)}_{i=1}^{N}` be the set of active cells with axial coordinates `(q_i, r_i)` and feature vectors `f_i`. Each cell is embedded as:

```
x_i = MLP(f_i) + PE(q_i, r_i)
```

where `PE` is a positional encoding (see Section 5). The sequence `{x_i}` is processed by a stack of Transformer encoder layers.

**For policy output,** each token produces a logit for "play here"; additional tokens for empty cells in the frontier are appended. For value output, a `[CLS]`-like global token is pooled.

**Advantages:**
- `O(1)` depth for global reasoning: every cell attends to every other cell in a single layer, unlike GNNs which require depth proportional to graph diameter.
- Naturally handles variable-length input.
- Positional encoding can be designed to exploit hex geometry (see Section 5).

**Disadvantages:**
- `O(N^2)` attention cost where `N` is the number of active cells. For small-to-medium boards (<200 cells), this is manageable; for very large active regions, linear attention variants (Katharopoulos et al., 2020) or sparse attention (Child et al., 2019) are needed.
- Transformers lack the inductive bias of spatial locality that convolutions provide. They must learn from data that nearby cells are more relevant than distant ones, whereas CNNs encode this structurally.
- Training data efficiency is typically worse than CNNs for spatially structured problems (Dosovitskiy et al., 2021), though this gap narrows with sufficient data.

### 2.5 Approach 4: Neural Cellular Automata-Inspired Approaches

**Concept.** Neural Cellular Automata (NCA; Mordvintsev et al., 2020) apply a fixed local update rule (parameterized by a neural network) iteratively to a grid of cell states. Each cell updates based only on its immediate neighbors, and the same rule is applied everywhere (translation equivariance). The system is run for `T` steps to propagate information across the grid.

**Adaptation for board games.** The board state is initialized on a hex grid, and a learned local update rule is applied repeatedly. After `T` iterations, each cell's hidden state encodes information from radius-`T` neighborhood. A readout head extracts policy and value.

**Advantages:**
- Inherently translation-equivariant and operates on arbitrary grid sizes.
- Very parameter-efficient: a single small update network (operating on 7 hex neighbors) is applied everywhere.
- Conceptually elegant for problems where global structure emerges from local interactions.

**Disadvantages:**
- Information propagation is slow: `T` steps for radius-`T` reasoning, identical to GNNs.
- Training is challenging: backpropagating through many iterative steps leads to vanishing/exploding gradients without careful normalization.
- Less explored in the game AI context; the approach is more common in generative modeling and morphogenesis simulation.

### 2.6 Tradeoff Summary

| Approach | Comp. Cost | Spatial Reasoning | Translation Invariance | Variable Board Size | Implementation Maturity |
|---|---|---|---|---|---|
| Dynamic Windowing + CNN | Low | Excellent (local) | Yes (within window) | Via re-centering | High |
| GNN | Medium | Good (local per layer) | By construction | Native | High (PyG, DGL) |
| Transformer | High (`O(N^2)`) | Excellent (global) | Via positional encoding | Native | High |
| Neural Cellular Automata | Low per step, high total | Good (local per step) | Yes | Native | Low (experimental) |

**For infinite hex tic-tac-toe specifically:** The game has strongly local dynamics (4-in-a-row on a hex grid means the "threat radius" is at most 3 cells from any placed stone). This makes dynamic windowing with CNNs highly attractive. The active region grows slowly relative to the number of moves, and a 19x19 or 25x25 window almost certainly suffices for the first 100+ moves. The CNN's superior computational efficiency enables faster MCTS rollouts, which is the dominant factor in playing strength for AlphaZero-style systems (Silver et al., 2018).

---

## 3. Neural Networks for Connection Games

### 3.1 The Game of Hex and AI Approaches

**Hex** (invented independently by Piet Hein in 1942 and John Nash in 1948) is played on an `n x n` rhombic board of hexagonal cells. Two players alternately place stones; the first to connect their two opposing board edges wins. Hex has no draws and is PSPACE-complete for general `n` (Reisch, 1981).

**MoHex and MCTS approaches (Arneson et al., 2010; Huang et al., 2013).** MoHex, developed at the University of Alberta, combined Monte Carlo Tree Search with pattern-based knowledge. It used Benzene, a Hex-specific solver with virtual connections (guaranteed connections that cannot be interrupted) and inferior cell analysis (cells that can be safely filled without affecting the outcome). MoHex-CNN (Gao, Muller, & Hayward, 2017) replaced the rollout policy with a CNN, training on expert games. The CNN used a standard ResNet on the brick-wall embedding of the hex board.

**NeuroHex (Young et al., 2016).** Young, Vasan, and Hayward trained deep CNNs to predict expert Hex moves, achieving superhuman prediction accuracy. Key architectural findings:

- A 12-layer CNN with 3x3 filters on the brick-wall layout significantly outperformed shallower architectures.
- Input features included: current player's stones, opponent's stones, empty cells, and distance-to-edge features for both players.
- The network implicitly learned virtual connection patterns, a critical strategic concept in Hex.
- Residual connections (He et al., 2016) were essential for training deep networks effectively.

**AlphaHex.** Following AlphaZero (Silver et al., 2018), several implementations applied the AlphaZero algorithm to Hex with ResNet architectures. The key adaptation is the asymmetric board: in Hex, the two players have different edge-connection objectives, breaking the rotational symmetry that AlphaZero exploits in Go. Data augmentation is limited to a single reflection (swapping players and transposing the board).

### 3.2 Connect6

**Connect6 (Wu & Huang, 2006)** is played on a Go board (19x19 intersection grid). Black places one stone first; thereafter, players alternate placing two stones per turn. The first to form a contiguous line of 6 (horizontal, vertical, or diagonal) wins. Connect6 is notable for:

- **Multi-move turns:** Except for Black's first move, every turn involves placing two stones, creating a large branching factor (~`(361 choose 2)` for the opening). This makes pure tree search intractable.
- **Threat-space search:** Wu (2006) developed threat-space search algorithms specific to connection-line games, identifying forced sequences of threats that lead to victory regardless of opponent response.
- **Relevance zone search (RZS):** Wu and Lin (2010) advanced RZS as a way to prune the search space by identifying which cells are "relevant" to a given threat sequence.

Neural network approaches to Connect6 are less developed than for Hex, but the multi-move mechanic and large board make it a natural testbed for the techniques discussed in Section 4.

### 3.3 Havannah

**Havannah** (Freeling, 1981) is played on a hexagonal board with base size `b` (total cells `3b^2 - 3b + 1`). Players alternate placing stones; a player wins by completing any of three structures: a **ring** (cycle enclosing at least one cell), a **bridge** (connection between any two corners), or a **fork** (connection between any three edges). The multiple win conditions, each with different geometric character, pose unique challenges for neural network architecture.

**Architectural implications.** A network for Havannah must simultaneously detect:
1. **Linear connectivity** (for bridges and forks) -- similar to Hex.
2. **Cyclic structure** (for rings) -- fundamentally different; requires detecting topological properties.
3. **Corner and edge awareness** -- the network must know which cells are corners and edges of the board.

Lorentz (2011) investigated MCTS approaches to Havannah. Neural network approaches typically use ResNets on the hex grid with dedicated input feature planes for corner cells, edge cells, and distance-to-corner/edge. The ring detection problem motivates deeper networks (larger receptive field to detect large rings) or global attention mechanisms.

### 3.4 How Win-Condition Geometry Affects Architecture

The win condition's geometric structure should inform the network architecture:

| Win Condition | Geometric Property | Architectural Implication |
|---|---|---|
| **k-in-a-row** (Connect6, Hex TT) | Linear, local | Small receptive field suffices; CNNs natural |
| **Edge connection** (Hex) | Path, global | Large receptive field needed; deep ResNets or attention |
| **Ring** (Havannah) | Topological (cyclic) | Requires non-local reasoning; attention or deep GNNs |
| **Fork/Bridge** (Havannah) | Path to specific boundaries | Boundary-aware features essential |

For **hex tic-tac-toe** (4-in-a-row on a hex grid), the win condition is linear and local: a winning line spans at most 4 cells, giving a maximum "threat radius" of 3. This means a CNN with receptive field radius >= 4 (achievable with 2 layers of 3x3 hex convolutions) can detect any single threat in its entirety. Deeper networks are still needed for strategic reasoning (blocking double threats, creating forcing sequences), but the fundamental detection is local.

---

## 4. Multi-Move Turn Architectures

### 4.1 The Problem

In games where a player makes `k > 1` moves per turn (Connect6: `k=2`; hex tic-tac-toe after the first turn: `k=2`), the action space explodes combinatorially. For a board with `N` empty cells, a single-move turn has `N` possible actions, but a two-move turn has `N * (N-1) / 2` (unordered pairs) or `N * (N-1)` (ordered pairs). For `N = 300`, this is ~45,000 unordered pairs -- too many for a flat softmax policy head but not intractable.

### 4.2 Joint Action Space

**Concept.** Treat the pair of moves `(a, b)` as a single compound action. The policy head outputs a probability distribution over all `N*(N-1)/2` unordered pairs (or `N*(N-1)` ordered pairs).

**Implementation.** One approach: the policy head outputs two independent per-cell logit maps `L_1, L_2` of shape `(H, W)`, and the joint probability of playing `(a, b)` is proportional to `exp(L_1[a] + L_2[b])`. This factored form avoids materializing the full `N^2` tensor while still capturing some interaction between the two moves (through the shared backbone features).

**Advantages:** Clean probabilistic semantics; compatible with MCTS.

**Disadvantages:** Even the factored form struggles to express strong correlations between the two moves (e.g., "if I play here, I must also play there"). The fully joint distribution is intractable for large boards.

### 4.3 Sequential Move Prediction (Autoregressive Factorization)

**Concept.** Factor the joint policy as `P(a, b) = P(a) * P(b | a)`. The network first predicts the first move `a`, then conditions on `a` to predict the second move `b`.

**Implementation options:**

1. **Two-pass approach:** Run the network to get `P(a)`. Sample or enumerate top-k candidates for `a`. For each candidate, update the board state (place stone at `a`), re-run the network to get `P(b | a)`.
   - Pro: Exact conditioning. Re-uses the same network architecture.
   - Con: Multiple forward passes per turn (expensive for MCTS).

2. **Single-pass with autoregressive head:** The network backbone produces shared features. A first policy head reads off `P(a)`. A second policy head, conditioned on `a` (e.g., by adding a one-hot indicator of `a` to the feature map), reads off `P(b | a)`.
   - Pro: Single (or nearly single) forward pass.
   - Con: The conditioning mechanism must be carefully designed.

3. **Transformer-based sequential decoding:** Treat the two moves as a sequence of two tokens. The backbone encodes the board; a decoder autoregressively generates the move sequence.
   - Pro: Natural for variable-length move sequences; extends to `k > 2`.
   - Con: Increased architectural complexity.

**For MCTS integration:** The sequential factorization is most natural. In AlphaZero-style MCTS, each node corresponds to a game state. A two-move turn simply introduces an intermediate state: after the first move, the same player gets a second move before the opponent responds. The tree naturally decomposes the two-move turn into two single-move decisions, each guided by the network's policy. This is the approach used in most AlphaZero adaptations for multi-move games (Czech et al., 2021).

### 4.4 The Branching Factor Explosion and Mitigations

For a two-move turn on a board with 300 empty cells, the naive branching factor is ~45,000. In MCTS, this is mitigated by:

1. **Policy network pruning:** Only expand moves with policy probability above a threshold, or the top-k moves. A well-trained policy network concentrates probability mass on a handful of strong moves.
2. **Progressive widening (Coulom, 2007):** Start with a narrow set of candidate moves and gradually widen as the node is visited more often.
3. **Decomposition into sequential moves:** As discussed above, decomposing the turn into two sequential moves reduces the branching factor from `O(N^2)` to `O(N) + O(N)` at the cost of doubling tree depth.
4. **Symmetry reduction:** If the two moves are unordered (placing two stones of the same color), `(a, b)` and `(b, a)` are equivalent. Canonicalizing the order halves the branching factor.

---

## 5. Positional Encoding for Board Games

### 5.1 The Role of Positional Encoding

In CNN-based architectures, spatial position is implicitly encoded through the receptive field structure: a convolutional filter does not know its absolute position, but the network as a whole can infer position from the input's spatial structure (e.g., border effects, distance-to-edge features). In attention-based and GNN architectures, positional information must be explicitly provided.

### 5.2 Absolute vs. Relative Positional Encodings

**Absolute positional encoding** assigns a unique vector to each board position. For a fixed-size board, this is straightforward: learn an embedding for each `(q, r)` coordinate pair. For variable-size boards, sinusoidal encodings (Vaswani et al., 2017) generalize:

```
PE(q, r, 2i)   = sin(q / 10000^{2i/d}) + sin(r / 10000^{2i/d})
PE(q, r, 2i+1) = cos(q / 10000^{2i/d}) + cos(r / 10000^{2i/d})
```

However, naively summing the `q` and `r` components loses information (different `(q,r)` pairs can have the same sum). A better approach uses separate frequency bands for `q` and `r`, concatenated:

```
PE(q, r) = [sin(q/f_1), cos(q/f_1), ..., sin(q/f_{d/4}), cos(q/f_{d/4}),
            sin(r/f_1), cos(r/f_1), ..., sin(r/f_{d/4}), cos(r/f_{d/4})]
```

**Relative positional encoding** (Shaw et al., 2018; Dai et al., 2019) encodes the displacement between pairs of cells rather than absolute positions. For hex grids, the displacement in axial coordinates is `(dq, dr) = (q_j - q_i, r_j - r_i)`. This is natural for attention mechanisms where the attention weight between cells `i` and `j` should depend on their relative position.

**Advantages of relative encoding for infinite boards:** On an infinite board with no fixed origin, absolute positions are arbitrary. Relative encodings are translation-invariant by construction, which is both semantically correct (the value of a configuration should not depend on where it sits in the infinite plane) and practically useful (the network generalizes to unseen absolute positions).

### 5.3 Positional Encoding on Hex Grids

The hexagonal grid's geometry introduces specific considerations:

**Hex-aware distance encoding.** The hex grid distance `d_hex(a, b)` is not the Euclidean distance between axial coordinates. In cube coordinates, it is the Chebyshev distance: `d = max(|dx|, |dy|, |dz|)`. Encoding this distance (or a binned version) as a positional feature allows the network to reason about hex-topological proximity.

**Directional encoding.** The six hex directions can be encoded explicitly. For each pair of cells, the direction from `i` to `j` can be discretized into one of 6 sectors (or a finer angular decomposition). This is useful for connection games where alignment along hex axes matters for winning conditions.

**Rotation-equivariant positional encoding.** If the network should be `p6`-equivariant, the positional encoding must transform consistently under rotation. The displacement vector `(dq, dr)` in axial coordinates does not transform trivially under hex rotations (a 60-degree rotation is a linear transformation in axial space, not a simple permutation). One approach: encode the displacement in cube coordinates `(dx, dy, dz)` (where rotation is a cyclic permutation) and use a rotation-equivariant encoding function.

### 5.4 Encoding Relative Position on an Infinite Board

For an infinite board, one effective strategy is to compute positions relative to a reference point:

1. **Centroid-relative:** Compute the centroid of all placed stones and express all positions relative to it. The centroid shifts slowly as play progresses, providing a stable reference.
2. **Last-move-relative:** Express positions relative to the most recent move. This provides a strong "attention anchor" but may obscure global structure.
3. **Coordinate features as input planes:** Simply include the `q` and `r` axial coordinates (relative to centroid or window origin) as two additional input feature planes. This is the approach used in many Go engines and is simple and effective (Silver et al., 2017).

---

## 6. Input Feature Planes for Board Games

### 6.1 AlphaZero's Feature Planes

AlphaZero (Silver et al., 2018) uses a carefully designed set of input feature planes for Go:

- **Stone presence (current position):** Binary planes indicating locations of Black stones and White stones. In AlphaZero's formulation, 8 history steps are encoded (16 planes for current and 7 previous positions per color).
- **Color to play:** A uniform binary plane (all 1s if Black to play, all 0s if White).
- **Move count / liberties (for Go):** Go-specific features like liberty counts per group.

The total is 17 binary feature planes (in the 2-player, 8-history version). The key design principle is that features should be **symmetric with respect to player swap** after appropriate transformation: swapping the "my stones" and "opponent stones" planes and flipping the color-to-play plane should yield the opponent's perspective.

### 6.2 Additional Useful Features for Connection Games

Beyond the basic AlphaZero feature set, connection games benefit from domain-specific features:

**Distance-to-edge / distance-to-corner (for Hex and Havannah).** Binary or graded planes encoding each cell's distance to the board boundary. Critical for edge-connection games.

**Connection potential.** For each cell, encode whether placing a stone there would connect two existing friendly groups, or would extend a group toward an edge. This can be computed via union-find on the current stone groups.

**Threat detection.** Encode cells that are part of an active threat (e.g., a line of 3 with an open end in a 4-in-a-row game). This requires a lightweight pattern scan but provides high-value information.

**Liberties / group size (for Go-like games).** The number of empty cells adjacent to each connected group. Not directly applicable to tic-tac-toe variants but important for Go.

**Line pattern features.** For k-in-a-row games on hex grids, encode for each cell and each of the 3 hex axes: the number of consecutive friendly stones in each direction, the number of consecutive enemy stones, and whether the line is open or blocked. This gives 6 feature planes per axis, 18 total. These features are dense but extremely informative for tactical play.

### 6.3 Feature Planes for Hexagonal Tic-Tac-Toe

For a hex tic-tac-toe variant (4-in-a-row on a hex grid, 2 moves per turn after the opening):

**Core planes (6 planes):**
1. `MY_STONES` -- binary, 1 where current player has a stone
2. `OPP_STONES` -- binary, 1 where opponent has a stone
3. `EMPTY` -- binary, 1 where cell is unoccupied (redundant with 1+2 but aids learning)
4. `MOVES_THIS_TURN` -- binary, 1 at the cell where the current player has already placed their first stone this turn (relevant during the second move of a two-move turn; all zeros during the first move)
5. `COLOR_TO_PLAY` -- uniform plane, 1.0 if the current player is Player 1, 0.0 if Player 2
6. `IS_FIRST_MOVE_OF_TURN` -- uniform plane, 1.0 if this is the first move of the turn, 0.0 if second

**Spatial context planes (4 planes):**
7. `DISTANCE_TO_CENTROID` -- float, normalized distance from each cell to the centroid of all placed stones
8. `Q_COORDINATE` -- float, axial q-coordinate relative to window center (normalized)
9. `R_COORDINATE` -- float, axial r-coordinate relative to window center (normalized)
10. `MOVE_RECENCY` -- float, for each occupied cell, a decaying value indicating how recently the stone was placed (e.g., most recent = 1.0, decaying by 0.9 per move). Encodes move history without requiring multiple history planes.

**Tactical planes (6 planes, one per hex axis pair):**
For each of the 3 hex axes:
11-12. `MY_LINE_LENGTH_AXIS_k` -- for each cell, the length of the longest contiguous friendly line through that cell along axis `k`
13-16. (Similarly for the other two axes)

Alternatively, a simpler encoding:
11. `MY_OPEN_THREES` -- binary, 1 at cells that are part of an open-ended line of 3 friendly stones (immediate win threat)
12. `OPP_OPEN_THREES` -- binary, 1 at cells that are part of an open-ended line of 3 opponent stones (must block)

**Total: 10-16 feature planes.** This is leaner than AlphaZero's 17 planes but encodes game-specific tactical information. The tactical planes are computationally cheap (linear scan along hex axes) and provide the network with high-value preprocessed features.

---

## Recommended Architecture for Infinite Hex Tic-Tac-Toe

This section synthesizes the preceding research into a concrete architectural recommendation for a neural network-based AI for infinite hex tic-tac-toe (4-in-a-row on an unbounded hexagonal grid, 2 moves per turn after the opening move).

### Board Representation: Axial Coordinates

**Recommendation: Axial coordinates `(q, r)` with brick-wall storage layout.**

Axial coordinates are the optimal choice for this application. They provide uniform neighbor relationships (all 6 hex neighbors are at fixed offsets), storage efficiency (2 components per cell), and clean distance computation (`d = max(|dq|, |dr|, |dq + dr|)` -- derived from the cube coordinate distance with `z = -q - r`). The brick-wall layout stores the hex grid in a standard 2D numpy/torch array where odd rows are logically shifted by half a cell. This enables the use of standard 3x3 convolutions with cuDNN optimization.

Internally, the game engine should use axial coordinates for game logic (move validation, win checking). The neural network input is constructed by mapping the axial-coordinate game state onto the brick-wall 2D array within a dynamically positioned window.

### Convolution Approach: Hex-Masked Axial Convolutions with P6 Equivariance

**Recommendation: Start with standard 3x3 convolutions on the brick-wall layout. Upgrade to `p6`-equivariant hex convolutions if data efficiency is a bottleneck.**

The pragmatic path is to begin with plain 3x3 convolutions on the brick-wall embedding. This is the approach that NeuroHex (Young et al., 2016) and MoHex-CNN (Gao et al., 2017) validated for the game of Hex. Two of the nine kernel weights will correspond to non-hex-adjacent cells, but in practice the network learns to suppress these, and the computational efficiency of using standard convolutions is substantial.

If training data is limited or training compute is constrained, upgrading to `p6`-equivariant convolutions (Hoogeboom et al., 2018) provides a factor of ~6x parameter efficiency for rotation-symmetric patterns. This is implemented by masking the 3x3 kernel to a 7-element hex kernel and constructing 6 rotated copies that share weights. The `e2cnn` library (Weiler & Cesa, 2019) provides production-ready `p6` and `p6m` equivariant layers.

**Recommended backbone:**
- **Architecture:** ResNet with 10-15 residual blocks, 128 channels per layer
- **Convolution:** 3x3 on brick-wall layout (or hex-masked 3x3 for equivariance)
- **Normalization:** Batch normalization
- **Activation:** ReLU (or GELU for marginal improvement)
- **Receptive field:** With 10 residual blocks (20 conv layers), the receptive field radius is ~20 cells, more than sufficient for the 4-in-a-row win condition and strategic planning

### Handling the Infinite Board: Dynamic Windowing with CNN

**Recommendation: Dynamic windowing with a fixed 19x19 window, centered on the centroid of placed stones, with relative coordinate feature planes.**

Justification:
1. **Locality of the game.** The 4-in-a-row win condition means all threats are within radius 3 of existing stones. Strategic play (creating double threats, blocking opponent threats) extends this to perhaps radius 5-6. A 19x19 window provides a radius of 9 from center, which is generous.
2. **Computational efficiency.** A fixed-size CNN on a 19x19 grid is extremely fast (~0.1ms per forward pass on GPU with a 10-block ResNet). This enables deep MCTS with thousands of simulations per move in reasonable time.
3. **Translation invariance.** Convolutions are translationally equivariant, so patterns learned anywhere in the window generalize to any position on the infinite board.
4. **Simplicity.** No custom graph operations, no quadratic attention costs, no variable-size batching complications.

**Specific implementation details:**
- Compute the centroid `(q_c, r_c)` of all placed stones (average of axial coordinates).
- Round to the nearest integer hex coordinate.
- Extract a 19x19 axial window centered on `(q_c, r_c)`.
- Map to brick-wall layout and construct input feature planes.
- If all placed stones fit within a smaller region, the window is padded with zeros (empty cells) at the edges. Include a `WITHIN_BOARD` plane that is 1 everywhere (since the board is infinite, all cells are valid; this plane is always 1 but maintains compatibility with finite-board variants).
- Include `Q_RELATIVE` and `R_RELATIVE` planes (axial coordinates relative to window center, normalized to `[-1, 1]`), so the network can reason about absolute position within the window if needed.
- **Fallback for very spread-out games:** If the bounding box of placed stones exceeds 19x19, increase the window size to 25x25 or 31x31. Alternatively, downsample the active region or use a hierarchical approach. In practice, for hex tic-tac-toe, this is unlikely to occur before the game ends.

### Handling Two-Moves-Per-Turn: Sequential Decomposition in MCTS

**Recommendation: Decompose each two-move turn into two sequential single-move decisions in the MCTS tree. The neural network always predicts a single move.**

This is the cleanest approach and aligns with how AlphaZero's MCTS naturally operates:

1. When it is a player's turn to make 2 moves, the MCTS tree contains an intermediate node after the first move where the **same player** moves again.
2. The neural network is called twice per turn: once to evaluate the position before the first move, and once after the first move is placed (with the `MOVES_THIS_TURN` feature plane updated).
3. The network architecture is identical for first and second moves. The `IS_FIRST_MOVE_OF_TURN` input plane tells the network whether it is predicting the first or second move.
4. After the second move, the turn passes to the opponent.

**Advantages of this decomposition:**
- Branching factor is `O(N)` per tree edge, not `O(N^2)`.
- The network can learn different strategies for the first and second moves of a turn (e.g., "first move creates a threat, second move extends it").
- Standard AlphaZero MCTS code requires minimal modification: only the turn-switching logic changes.
- Policy target construction during training is straightforward: each MCTS position (including intermediate same-player positions) produces a policy target.

### Input Feature Planes

**Recommendation: 12 input feature planes.**

| # | Feature Plane | Type | Description |
|---|---|---|---|
| 1 | `MY_STONES` | Binary | Current player's stones |
| 2 | `OPP_STONES` | Binary | Opponent's stones |
| 3 | `MOVES_THIS_TURN` | Binary | Stone placed in first move of current turn (zeros during first move) |
| 4 | `IS_FIRST_MOVE` | Uniform | 1.0 if first move of turn, 0.0 if second |
| 5 | `COLOR_TO_PLAY` | Uniform | 1.0 for Player 1, 0.0 for Player 2 |
| 6 | `MOVE_RECENCY` | Float | Per-cell recency weighting (most recent = 1.0, decay 0.9/move) |
| 7 | `Q_RELATIVE` | Float | Normalized q-coordinate relative to window center |
| 8 | `R_RELATIVE` | Float | Normalized r-coordinate relative to window center |
| 9 | `DISTANCE_TO_CENTROID` | Float | Normalized hex distance to centroid of all stones |
| 10 | `MY_THREAT_3` | Binary | Cells that complete a friendly open three (immediate 4-in-a-row threat) |
| 11 | `OPP_THREAT_3` | Binary | Cells that complete an opponent open three (must-block) |
| 12 | `MY_THREAT_2` | Binary | Cells that extend a friendly open two to open three |

Planes 10-12 are "tactical hint" features computed by a fast linear scan along the 3 hex axes. They are optional (a sufficiently deep network can learn to detect these patterns) but significantly accelerate training, especially in the early phases when the network has not yet learned basic threat detection.

### Exploiting Hexagonal Symmetry

**Data augmentation (minimum viable approach):** The infinite hex board has 6-fold rotational symmetry and a reflection, giving a 12-element dihedral group `D6`. Each training sample can be augmented into 12 equivalent samples by rotating the board state by `{0, 60, 120, 180, 240, 300}` degrees and optionally reflecting. In axial coordinates, a 60-degree rotation maps `(q, r)` to `(-r, q + r)` and the other rotations are compositions of this. Reflection maps `(q, r)` to `(r, q)`. Both the board features and the policy targets must be transformed consistently.

This 12x data augmentation is free (no additional network evaluations) and is the single most impactful technique for data efficiency on hex boards.

**Equivariant network (advanced approach):** Use `p6m`-equivariant convolutions (Hoogeboom et al., 2018; Weiler & Cesa, 2019) so that the network's output is exactly equivariant to all 12 symmetries by construction. This eliminates the need for augmentation and provides stronger generalization guarantees. The tradeoff is a 12x increase in feature map channels (which can be offset by reducing the base channel count).

**Recommendation: Start with data augmentation (12-fold `D6`). If training budget allows, experiment with `p6`-equivariant convolutions (6-fold, ignoring reflection since it can be handled by augmenting the remaining factor of 2). The expected improvement is 2-5x data efficiency for equivalent playing strength.**

### Summary: Complete Architecture Specification

```
Input:  12 x 19 x 19  (feature planes on brick-wall hex layout)
        ├── Dynamic windowing centered on stone centroid
        └── Axial coordinates mapped to brick-wall 2D array

Backbone: ResNet
        ├── Initial conv: 3x3, 12 -> 128 channels, BN, ReLU
        ├── 12 Residual blocks: [3x3 conv, BN, ReLU, 3x3 conv, BN] + skip, ReLU
        └── Optional: hex-masked or p6-equivariant convolutions

Policy Head:
        ├── 1x1 conv, 128 -> 2 channels, BN, ReLU
        ├── Flatten -> FC -> 19*19 = 361 logits
        └── Softmax over valid (empty) cells in window

Value Head:
        ├── 1x1 conv, 128 -> 1 channel, BN, ReLU
        ├── Flatten -> FC(361, 256) -> ReLU -> FC(256, 1)
        └── Tanh (output in [-1, 1])

Training: AlphaZero self-play loop
        ├── MCTS with 800 simulations/move
        ├── Sequential two-move turns (intermediate same-player nodes)
        ├── D6 data augmentation (12-fold)
        └── Loss = MSE(value) + CrossEntropy(policy) + L2 regularization

Symmetry: D6 augmentation (12x) at minimum; p6-equivariant convolutions optional
```

---

## References

- Arneson, B., Hayward, R., & Henderson, P. (2010). MoHex wins Hex tournament. *ICGA Journal*, 33(3), 180-183.
- Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). Generating long sequences with sparse transformers. *arXiv:1904.10509*.
- Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML 2016*.
- Coulom, R. (2007). Efficient selectivity and backup operators in Monte-Carlo tree search. *CG 2006*, LNCS 4630.
- Czech, J., Korus, P., & Kersting, K. (2021). Improving AlphaZero using Monte-Carlo graph search. *ICML 2021*.
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., & Salinas, R. (2019). Transformer-XL: Attentive language models beyond a fixed-length context. *ACL 2019*.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
- Freeling, C. (1981). Havannah. Board game, Ravensburger.
- Gao, C., Muller, M., & Hayward, R. (2017). Focused depth-first proof number search using convolutional neural networks for the game of Hex. *IJCAI 2017*.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *ICML 2017*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Hoogeboom, E., Peters, J. W. T., Cohen, T. S., & Welling, M. (2018). HexaConv. *ICLR 2018*.
- Huang, S. C., Arneson, B., Hayward, R., Muller, M., & Pawlewicz, J. (2013). MoHex 2.0: A pattern-based MCTS Hex player. *CG 2013*, LNCS 8427.
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *ICML 2020*.
- Lorentz, R. (2011). Improving Monte-Carlo tree search in Havannah. *CG 2010*, LNCS 6515.
- Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). Growing neural cellular automata. *Distill*.
- Patel, A. (2015). Hexagonal grids. *Red Blob Games*. https://www.redblobgames.com/grids/hexagons/
- Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR 2017*.
- Red Blob Games. (2013). Hexagonal grids reference. https://www.redblobgames.com/grids/hexagons/
- Reisch, S. (1981). Hex ist PSPACE-vollstandig. *Acta Informatica*, 15(2), 167-191.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *NAACL 2018*.
- Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., ... & Hassabis, D. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.
- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.
- Steppa, C., & Holch, T. L. (2019). HexagDLy -- Processing hexagonally sampled data with CNNs. *SoftwareX*, 9, 193-198.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.
- Wang, M., Zheng, D., Ye, Z., Gan, Q., Li, M., Song, X., ... & Zhang, Z. (2019). Deep Graph Library: A graph-centric, highly-performant package for graph neural networks. *arXiv:1909.01315*.
- Weiler, M., & Cesa, G. (2019). General E(2)-equivariant steerable CNNs. *NeurIPS 2019*.
- Wu, I. C., & Huang, D. Y. (2006). A new family of k-in-a-row games. *ACG 2005*, LNCS 4250.
- Wu, I. C., & Lin, P. H. (2010). Relevance-zone-oriented proof search for Connect6. *IEEE Transactions on Computational Intelligence and AI in Games*, 2(3), 191-207.
- Young, K., Vasan, G., & Hayward, R. (2016). NeuroHex: A deep Q-learning Hex agent. *CG 2016*, LNCS 10068.

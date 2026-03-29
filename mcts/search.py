"""Monte Carlo Tree Search for Infinite Hexagonal Tic-Tac-Toe.

Implements PUCT-based search with neural network evaluation.  Handles
the interleaved two-sub-move-per-turn tree structure where each player
places two stones per turn (except Player 1's first turn which is one
stone).

The search cycle is:
    1. Select a leaf node by walking down the tree using UCB scores.
    2. Evaluate the leaf with the neural network.
    3. Expand the leaf with move priors from the policy head.
    4. Backpropagate the value estimate up to the root.

After ``num_simulations`` iterations, the visit-count distribution at
the root encodes the MCTS policy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from game.hex_grid import HexCoord, axial_to_brick, brick_to_axial
from game.rules import GameState
from mcts.node import MCTSNode
from mcts.zoi import compute_zoi, compute_zoi_mask
from nn.features import extract_features
from nn.model import HexTTTNet


class MCTS:
    """Monte Carlo Tree Search for Infinite Hex Tic-Tac-Toe.

    Implements PUCT-based search with neural network evaluation.
    Handles the interleaved two-sub-move-per-turn tree structure.

    Args:
        network: a :class:`HexTTTNet` instance for policy and value evaluation.
        config: dict of search hyperparameters (see ``__init__`` for keys).
    """

    def __init__(self, network: HexTTTNet, config: dict) -> None:
        """
        Config keys:
            num_simulations (int): number of MCTS iterations (e.g. 200, 600, 800).
            cpuct (float): exploration constant for PUCT (e.g. 2.5).
            dirichlet_alpha (float): Dirichlet noise alpha (e.g. 0.10).
            dirichlet_epsilon (float): noise mixing weight (e.g. 0.25).
            temperature (float): temperature for move selection.
            fpu_reduction (float): First Play Urgency reduction (e.g. 0.0 or -0.2).
            zoi_margin (int): Zone of Interest margin (e.g. 3).
            grid_size (int): spatial size of the NN input grid (e.g. 19).
            virtual_loss (int): virtual loss amount for parallel search (e.g. 3).
            device (str): torch device string (e.g. 'cpu', 'cuda').
        """
        self.network = network
        self.config = config
        self.device = config.get("device", "cpu")

    def search(self, game_state: GameState) -> Tuple[MCTSNode, Dict[HexCoord, float]]:
        """Run MCTS from the given game state.

        Returns:
            root: the root MCTSNode with populated search statistics.
            policy: dict mapping HexCoord -> probability (normalised visit counts).
        """
        num_simulations: int = self.config.get("num_simulations", 200)
        cpuct: float = self.config.get("cpuct", 2.5)
        fpu_reduction: float = self.config.get("fpu_reduction", 0.0)
        dirichlet_alpha: float = self.config.get("dirichlet_alpha", 0.10)
        dirichlet_epsilon: float = self.config.get("dirichlet_epsilon", 0.25)

        # 1. Create root node.
        root = MCTSNode(game_state=game_state)

        # 2. Evaluate and expand root.
        if not game_state.is_terminal:
            root_value = self._evaluate_and_expand(root)
            # Initial backup for root (just the root itself).
            root.visit_count += 1
            root.total_value += root_value

            # 3. Add Dirichlet noise to root priors for exploration.
            if root.children:
                self._add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)
        else:
            # Terminal root: no children, just set visit count.
            root.visit_count = 1
            # Value for a terminal state: +1 if current player won, -1 otherwise.
            if game_state.winner == game_state.current_player:
                root.total_value = 1.0
            elif game_state.winner is not None:
                root.total_value = -1.0
            else:
                root.total_value = 0.0

        # 4. Run simulations.
        for _ in range(num_simulations):
            # Select leaf.
            leaf = self._select_leaf(root, cpuct, fpu_reduction)

            # Evaluate terminal nodes directly.
            if leaf.is_terminal:
                # Terminal value from the perspective of the leaf's current player.
                gs = leaf.game_state
                if gs.winner is not None:
                    # Winner is set; determine from leaf's current_player perspective.
                    if gs.winner == gs.current_player:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    value = 0.0  # draw (shouldn't happen in this game, but be safe)
            else:
                # Neural network evaluation and expansion.
                value = self._evaluate_and_expand(leaf)

            # Backup the value.
            leaf.backup(value)

        # 5. Return root and visit distribution.
        policy = root.get_visit_distribution()
        return root, policy

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        """Evaluate a leaf node with the neural network and expand it.

        Steps:
            1. Extract features from the node's game_state.
            2. Compute ZoI mask for valid move filtering.
            3. Run network forward pass (no gradient computation).
            4. Extract move priors (masked to legal moves in ZoI).
            5. Expand the node with priors.
            6. Return the value estimate.

        Args:
            node: an unexpanded, non-terminal MCTSNode.

        Returns:
            The value estimate from the network (from the current player's
            perspective).
        """
        game_state = node.game_state
        grid_size: int = self.config.get("grid_size", 19)
        zoi_margin: int = self.config.get("zoi_margin", 3)
        half = grid_size // 2

        # 1. Extract features -- returns (C, H, W) tensor and window centre.
        features, (center_q, center_r) = extract_features(game_state, grid_size=grid_size)

        # 2. Compute ZoI mask.
        zoi_mask_2d = compute_zoi_mask(
            game_state, center_q, center_r, grid_size, margin=zoi_margin
        )

        # Prepare tensors for network (add batch dimension).
        features_batch = features.unsqueeze(0).to(self.device)  # (1, C, H, W)
        mask_flat = torch.from_numpy(zoi_mask_2d.reshape(1, -1)).to(self.device)  # (1, H*W)

        # 3. Forward pass.
        self.network.eval()
        with torch.no_grad():
            output = self.network(features_batch, valid_moves_mask=mask_flat)

        # 4. Extract policy probabilities and value.
        policy_probs = output["policy"].squeeze(0).cpu().numpy()  # (H*W,)
        value = output["value"].squeeze().item()  # scalar in [-1, 1]

        # 5. Map policy probabilities back to HexCoord moves.
        # Only include moves that are in the ZoI (mask > 0).
        move_priors: Dict[HexCoord, float] = {}
        zoi_mask_flat = zoi_mask_2d.reshape(-1)

        for idx in range(grid_size * grid_size):
            if zoi_mask_flat[idx] > 0.0:
                prob = float(policy_probs[idx])
                if prob > 0.0:
                    row_idx = idx // grid_size
                    col_idx = idx % grid_size
                    # Convert grid position back to axial coordinates.
                    coord = brick_to_axial(
                        row_idx - half, col_idx - half, center_q, center_r, grid_size
                    )
                    move_priors[coord] = prob

        # If no moves have probability (e.g., all masked out), assign uniform
        # priors over all ZoI cells as a fallback.
        if not move_priors:
            zoi_cells = compute_zoi(game_state, margin=zoi_margin)
            if zoi_cells:
                uniform_p = 1.0 / len(zoi_cells)
                for coord in zoi_cells:
                    move_priors[coord] = uniform_p

        # Re-normalise priors to sum to 1 (they should already be close due
        # to softmax, but masking and float rounding can introduce drift).
        if move_priors:
            total_p = sum(move_priors.values())
            if total_p > 0.0:
                inv_total = 1.0 / total_p
                move_priors = {m: p * inv_total for m, p in move_priors.items()}

        # 6. Expand the node.
        if move_priors:
            node.expand(move_priors)

        return value

    def _select_leaf(
        self,
        root: MCTSNode,
        cpuct: float,
        fpu_reduction: float,
    ) -> MCTSNode:
        """Walk down the tree using UCB scores until reaching an
        unexpanded or terminal node.

        Applies virtual loss on the traversal path to support parallel
        search (even in single-threaded mode, virtual loss is benign).

        Args:
            root: the root node to start selection from.
            cpuct: exploration constant for PUCT.
            fpu_reduction: FPU reduction for unvisited children.

        Returns:
            The selected leaf MCTSNode.
        """
        virtual_loss: int = self.config.get("virtual_loss", 3)
        node = root

        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            node = node.select_child(cpuct, fpu_reduction)
            node.add_virtual_loss(virtual_loss)

        # Remove virtual losses from the path (they will be re-applied
        # as real visits during backup).
        current = node
        while current is not root and current.parent is not None:
            current.remove_virtual_loss(virtual_loss)
            current = current.parent

        return node

    def _add_dirichlet_noise(
        self,
        node: MCTSNode,
        alpha: float,
        epsilon: float,
    ) -> None:
        """Add Dirichlet noise to the root node's children priors for exploration.

        The noise ensures that every legal move has a non-zero probability of
        being explored, preventing the search from being overly narrow due to
        a strong but potentially incorrect policy network.

        Formula::

            new_prior = (1 - epsilon) * prior + epsilon * noise

        Args:
            node: the root node whose children priors will be mixed with noise.
            alpha: Dirichlet distribution concentration parameter.
            epsilon: mixing weight (0 = no noise, 1 = pure noise).
        """
        if not node.children:
            return

        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for i, move in enumerate(moves):
            child = node.children[move]
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(noise[i])

    def get_move(
        self,
        game_state: GameState,
        temperature: Optional[float] = None,
        prev_root: Optional[MCTSNode] = None,
    ) -> Tuple[HexCoord, Dict[HexCoord, float], MCTSNode]:
        """Run search and return the best move, policy, and new root.

        Supports **tree reuse** (inspired by hexgo): if ``prev_root`` is
        provided, the search attempts to find the subtree matching the
        current game state among the previous root's children/grandchildren.
        If found, this subtree becomes the new root with its accumulated
        statistics, saving ~sims/branching_factor work for free.

        Args:
            game_state: the current game state.
            temperature: move selection temperature.  If ``None``, uses the
                value from ``self.config['temperature']``.
            prev_root: optional root from a previous search call, enabling
                tree reuse.

        Returns:
            ``(best_move, policy_dict, root_node)`` where ``root_node`` can
            be passed as ``prev_root`` to the next call.
        """
        if temperature is None:
            temperature = self.config.get("temperature", 0.0)

        # Tree reuse: try to find the current state in the previous tree
        reused = False
        if prev_root is not None and prev_root.children:
            # Search one or two levels deep (to cover both sub-moves
            # within the same turn)
            for child in prev_root.children.values():
                if child.game_state.board.hash == game_state.board.hash:
                    child.parent = None  # detach from old tree (GC)
                    root, policy = self._search_from_existing(child)
                    reused = True
                    break
                # Check grandchildren (covers opponent's response)
                if child.children:
                    for grandchild in child.children.values():
                        if grandchild.game_state.board.hash == game_state.board.hash:
                            grandchild.parent = None
                            root, policy = self._search_from_existing(grandchild)
                            reused = True
                            break
                    if reused:
                        break

        if not reused:
            root, policy = self.search(game_state)

        best_move = root.get_best_move(temperature=temperature)

        return best_move, policy, root

    def _search_from_existing(self, node: MCTSNode) -> Tuple[MCTSNode, Dict[HexCoord, float]]:
        """Continue MCTS from an existing node (tree reuse).

        Re-applies Dirichlet noise to the reused root and runs the
        remaining simulations.
        """
        num_simulations: int = self.config.get("num_simulations", 200)
        cpuct: float = self.config.get("cpuct", 2.5)
        fpu_reduction: float = self.config.get("fpu_reduction", 0.0)
        dirichlet_alpha: float = self.config.get("dirichlet_alpha", 0.10)
        dirichlet_epsilon: float = self.config.get("dirichlet_epsilon", 0.25)

        # Re-apply Dirichlet noise for fresh exploration
        if node.children:
            self._add_dirichlet_noise(node, dirichlet_alpha, dirichlet_epsilon)

        # Run additional simulations (subtract already-accumulated visits)
        remaining = max(0, num_simulations - node.visit_count)
        for _ in range(remaining):
            leaf = self._select_leaf(node, cpuct, fpu_reduction)
            if leaf.is_terminal:
                gs = leaf.game_state
                if gs.winner is not None:
                    value = 1.0 if gs.winner == gs.current_player else -1.0
                else:
                    value = 0.0
            else:
                value = self._evaluate_and_expand(leaf)
            leaf.backup(value)

        policy = node.get_visit_distribution()
        return node, policy

"""MCTS tree node for Infinite Hexagonal Tic-Tac-Toe.

Each node represents a single sub-move in the interleaved move sequence:
P1-move1 -> P1-move2 -> P2-move1 -> P2-move2 -> ...

The backup logic is critical: values are negated only at player transitions
(when the child's current_player differs from the parent's), not within
same-player sub-move pairs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from game.hex_grid import HexCoord
from game.rules import GameState


class MCTSNode:
    """A node in the MCTS search tree.

    Attributes:
        game_state: GameState at this node.
        parent: parent node (None for root).
        move: the HexCoord that led to this node from parent.
        children: dict mapping HexCoord -> MCTSNode.
        visit_count: N(s) -- number of times this node was visited.
        total_value: W(s) -- sum of backed-up values.
        prior: P(s,a) -- prior probability from the policy network.
        is_expanded: whether this node has been expanded with children.
        virtual_loss_count: number of in-flight virtual losses applied.
    """

    __slots__ = [
        "game_state",
        "parent",
        "move",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "is_expanded",
        "virtual_loss_count",
    ]

    def __init__(
        self,
        game_state: GameState,
        parent: Optional[MCTSNode] = None,
        move: Optional[HexCoord] = None,
        prior: float = 0.0,
    ) -> None:
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children: Dict[HexCoord, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False
        self.virtual_loss_count: int = 0

    # ------------------------------------------------------------------
    # Value computation
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean action value Q(s) = W(s) / N(s). Returns 0.0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(
        self,
        parent_visits: int,
        cpuct: float,
        fpu_reduction: float = 0.0,
    ) -> float:
        """PUCT score for child selection.

        Formula::

            Q(s,a) + cpuct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

        For unvisited nodes (N=0), the Q value is estimated as the parent's
        mean Q minus ``fpu_reduction`` (First Play Urgency).

        Args:
            parent_visits: N(parent) -- total visits to the parent node.
            cpuct: exploration constant.
            fpu_reduction: value subtracted from parent Q for unvisited children.

        Returns:
            The UCB/PUCT score (higher is better).
        """
        exploration = cpuct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

        if self.visit_count == 0:
            # First Play Urgency: use parent's Q - reduction.
            parent_q = self.parent.q_value if self.parent is not None else 0.0
            q = parent_q - fpu_reduction
        else:
            q = self.q_value

        return q + exploration

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_child(self, cpuct: float, fpu_reduction: float = 0.0) -> MCTSNode:
        """Select the child with the highest UCB score.

        Args:
            cpuct: exploration constant for PUCT.
            fpu_reduction: FPU reduction for unvisited nodes.

        Returns:
            The child MCTSNode with maximum UCB score.

        Raises:
            ValueError: if the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child from node with no children")

        parent_visits = self.visit_count
        best_score = -math.inf
        best_child: Optional[MCTSNode] = None

        for child in self.children.values():
            score = child.ucb_score(parent_visits, cpuct, fpu_reduction)
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(self, move_priors: Dict[HexCoord, float]) -> None:
        """Create child nodes for each move in *move_priors*.

        Each child receives the corresponding prior probability from the
        policy network.  The child's game_state is obtained by applying
        the move to this node's game_state.

        Args:
            move_priors: mapping from HexCoord to prior probability.
        """
        for move, prior in move_priors.items():
            child_state = self.game_state.apply_move(move)
            child = MCTSNode(
                game_state=child_state,
                parent=self,
                move=move,
                prior=prior,
            )
            self.children[move] = child

        self.is_expanded = True

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self, value: float) -> None:
        """Backpropagate value from this node up to the root.

        CRITICAL: The value is negated only at player transitions (when
        a child's ``current_player`` differs from its parent's
        ``current_player``).  Within same-player sub-moves, the value
        sign is preserved.

        This correctly handles the interleaved sub-move tree:
        P1-move1 -> P1-move2 -> P2-move1 -> P2-move2 -> ...

        The incoming ``value`` is from the perspective of the current
        player at this (leaf) node.

        Args:
            value: the value estimate (from the evaluating player's perspective).
        """
        node: Optional[MCTSNode] = self
        v = value

        while node is not None:
            node.visit_count += 1
            node.total_value += v

            # When traversing to the parent, check if there is a player
            # transition.  If the parent's current_player is different from
            # this node's current_player, negate the value.
            if node.parent is not None:
                if node.parent.game_state.current_player != node.game_state.current_player:
                    v = -v

            node = node.parent

    # ------------------------------------------------------------------
    # Virtual loss (for parallel MCTS)
    # ------------------------------------------------------------------

    def add_virtual_loss(self, amount: int = 1) -> None:
        """Add virtual loss for parallel MCTS.

        Virtual losses discourage other threads from exploring the same
        path by temporarily making this node appear worse.

        Args:
            amount: number of virtual losses to add.
        """
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count += amount
            node.total_value -= amount
            node.virtual_loss_count += amount
            node = node.parent

    def remove_virtual_loss(self, amount: int = 1) -> None:
        """Remove virtual loss after evaluation completes.

        Args:
            amount: number of virtual losses to remove.
        """
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visit_count -= amount
            node.total_value += amount
            node.virtual_loss_count -= amount
            node = node.parent

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def get_visit_distribution(self) -> Dict[HexCoord, float]:
        """Return normalised visit counts over children (the MCTS policy target).

        Returns:
            Dict mapping each child's move to its fraction of total child visits.
            Returns empty dict if node has no children or all visit counts are zero.
        """
        if not self.children:
            return {}

        total = sum(child.visit_count for child in self.children.values())
        if total == 0:
            return {}

        return {
            move: child.visit_count / total
            for move, child in self.children.items()
        }

    def get_best_move(self, temperature: float = 0.0) -> HexCoord:
        """Select a move based on visit counts.

        Args:
            temperature: controls exploration.
                - ``0.0``: argmax (deterministic -- pick most visited child).
                - ``> 0.0``: sample proportional to ``N^(1/temp)``.

        Returns:
            The selected HexCoord move.

        Raises:
            ValueError: if the node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select best move from node with no children")

        moves = list(self.children.keys())
        visit_counts = np.array(
            [self.children[m].visit_count for m in moves], dtype=np.float64
        )

        if temperature <= 0.0 or temperature < 1e-8:
            # Deterministic: argmax.  Break ties by highest prior.
            max_visits = visit_counts.max()
            candidates = [
                i for i, v in enumerate(visit_counts) if v == max_visits
            ]
            if len(candidates) == 1:
                return moves[candidates[0]]
            # Tie-break by prior.
            best_idx = max(candidates, key=lambda i: self.children[moves[i]].prior)
            return moves[best_idx]

        # Temperature-based sampling.
        # Apply temperature: N^(1/temp), then normalise.
        # Use log-space for numerical stability.
        log_counts = np.log(visit_counts + 1e-10)
        log_probs = log_counts / temperature
        # Subtract max for numerical stability before exp.
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()

        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]

    # ------------------------------------------------------------------
    # Terminal check
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """Whether this node's game state is terminal (game over)."""
        return self.game_state.is_terminal

"""Batched parallel evaluation support for MCTS.

Collects leaf-node evaluation requests from multiple search threads and
batches them together for efficient GPU inference.  This amortises the
per-inference overhead (kernel launch, memory transfer) across many
evaluations, which is critical for achieving high simulations-per-second
on GPU hardware.

Architecture::

    Search thread 1 ──┐
    Search thread 2 ──┤
    Search thread 3 ──┼──> Request Queue ──> Eval Thread ──> Network (batched)
         ...          │                                          │
    Search thread N ──┘                      Results ◄──────────┘

Each search thread calls :meth:`BatchedEvaluator.evaluate` which blocks
until the batch containing its request has been processed.
"""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Dict, List, Tuple

import numpy as np
import torch

from game.hex_grid import HexCoord, brick_to_axial
from game.rules import GameState
from mcts.zoi import compute_zoi, compute_zoi_mask
from nn.features import extract_features


class _EvalRequest:
    """Internal container for a single evaluation request."""

    __slots__ = ["game_state", "result_event", "value", "move_priors"]

    def __init__(self, game_state: GameState) -> None:
        self.game_state = game_state
        self.result_event = threading.Event()
        self.value: float = 0.0
        self.move_priors: Dict[HexCoord, float] = {}


class BatchedEvaluator:
    """Collects leaf nodes and evaluates them in batches on GPU.

    Usage::

        evaluator = BatchedEvaluator(network, batch_size=32, device='cuda')
        evaluator.start()

        # From search threads:
        value, priors = evaluator.evaluate(game_state)

        # When done:
        evaluator.stop()

    Args:
        network: a :class:`HexTTTNet` instance.
        grid_size: spatial dimension for feature extraction (default 19).
        batch_size: maximum number of requests to batch together.
        device: torch device for inference.
        timeout_ms: maximum time (in milliseconds) to wait for a full
            batch before processing a partial batch.
        zoi_margin: Zone of Interest margin for move masking.
    """

    def __init__(
        self,
        network: object,
        grid_size: int = 19,
        batch_size: int = 32,
        device: str = "cpu",
        timeout_ms: float = 1.0,
        zoi_margin: int = 3,
    ) -> None:
        self.network = network
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.device = device
        self.timeout_ms = timeout_ms
        self.zoi_margin = zoi_margin
        self._request_queue: Queue[_EvalRequest] = Queue()
        self._running = False
        self._eval_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background evaluation thread."""
        if self._running:
            return
        self._running = True
        self._eval_thread = threading.Thread(
            target=self._eval_loop, daemon=True, name="BatchedEvaluator"
        )
        self._eval_thread.start()

    def stop(self) -> None:
        """Stop the background evaluation thread.

        Blocks until the thread has finished processing any remaining
        requests.
        """
        self._running = False
        if self._eval_thread is not None:
            self._eval_thread.join(timeout=5.0)
            self._eval_thread = None

    def evaluate(self, game_state: GameState) -> Tuple[float, Dict[HexCoord, float]]:
        """Submit a game state for evaluation.  Blocks until the result is ready.

        Args:
            game_state: the game state to evaluate.

        Returns:
            ``(value, move_priors)`` where ``value`` is a scalar in [-1, 1]
            and ``move_priors`` maps HexCoord to prior probabilities.
        """
        request = _EvalRequest(game_state)
        self._request_queue.put(request)
        # Block until the evaluation thread has processed this request.
        request.result_event.wait()
        return request.value, request.move_priors

    def _eval_loop(self) -> None:
        """Background thread: collect requests, batch them, run network,
        distribute results."""
        timeout_s = self.timeout_ms / 1000.0

        while self._running:
            batch: List[_EvalRequest] = []

            # Collect up to batch_size requests, waiting briefly for more.
            try:
                first = self._request_queue.get(timeout=timeout_s)
                batch.append(first)
            except Empty:
                continue

            # Try to fill the rest of the batch without blocking.
            deadline = time.monotonic() + timeout_s
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._request_queue.get(timeout=max(remaining, 0.0001))
                    batch.append(req)
                except Empty:
                    break

            # Process the batch.
            results = self._batch_evaluate([r.game_state for r in batch])

            # Distribute results to waiting threads.
            for request, (value, priors) in zip(batch, results):
                request.value = value
                request.move_priors = priors
                request.result_event.set()

        # Drain any remaining requests on shutdown.
        while not self._request_queue.empty():
            try:
                req = self._request_queue.get_nowait()
                # Return neutral evaluation for stragglers.
                req.value = 0.0
                req.move_priors = {}
                req.result_event.set()
            except Empty:
                break

    def _batch_evaluate(
        self, game_states: List[GameState]
    ) -> List[Tuple[float, Dict[HexCoord, float]]]:
        """Run the network on a batch of game states.

        Extracts features and ZoI masks for each state, batches them into
        a single tensor, runs inference, and unpacks the results.

        Args:
            game_states: list of GameState instances to evaluate.

        Returns:
            List of ``(value, move_priors)`` tuples, one per input state.
        """
        grid_size = self.grid_size
        half = grid_size // 2
        batch_size = len(game_states)

        # Pre-allocate batch tensors.
        features_list: List[torch.Tensor] = []
        masks_list: List[np.ndarray] = []
        centers: List[Tuple[int, int]] = []

        for gs in game_states:
            feat, (cq, cr) = extract_features(gs, grid_size=grid_size)
            features_list.append(feat)
            centers.append((cq, cr))

            zoi_mask_2d = compute_zoi_mask(
                gs, cq, cr, grid_size, margin=self.zoi_margin
            )
            masks_list.append(zoi_mask_2d.reshape(-1))

        # Stack into batch tensors.
        features_batch = torch.stack(features_list, dim=0).to(self.device)  # (B, C, H, W)
        masks_batch = torch.from_numpy(
            np.stack(masks_list, axis=0)
        ).to(self.device)  # (B, H*W)

        # Network forward pass.
        self.network.eval()
        with torch.no_grad():
            output = self.network(features_batch, valid_moves_mask=masks_batch)

        policy_probs_batch = output["policy"].cpu().numpy()  # (B, H*W)
        values_batch = output["value"].cpu().numpy()  # (B, 1)

        # Unpack results.
        results: List[Tuple[float, Dict[HexCoord, float]]] = []

        for i in range(batch_size):
            value = float(values_batch[i, 0])
            policy_probs = policy_probs_batch[i]  # (H*W,)
            center_q, center_r = centers[i]
            zoi_mask_flat = masks_list[i]

            move_priors: Dict[HexCoord, float] = {}
            for idx in range(grid_size * grid_size):
                if zoi_mask_flat[idx] > 0.0:
                    prob = float(policy_probs[idx])
                    if prob > 0.0:
                        row_idx = idx // grid_size
                        col_idx = idx % grid_size
                        coord = brick_to_axial(
                            row_idx - half,
                            col_idx - half,
                            center_q,
                            center_r,
                            grid_size,
                        )
                        move_priors[coord] = prob

            # Fallback: uniform priors if everything got masked out.
            if not move_priors:
                zoi_cells = compute_zoi(game_states[i], margin=self.zoi_margin)
                if zoi_cells:
                    uniform_p = 1.0 / len(zoi_cells)
                    for coord in zoi_cells:
                        move_priors[coord] = uniform_p

            # Re-normalise.
            if move_priors:
                total_p = sum(move_priors.values())
                if total_p > 0.0:
                    inv_total = 1.0 / total_p
                    move_priors = {m: p * inv_total for m, p in move_priors.items()}

            results.append((value, move_priors))

        return results

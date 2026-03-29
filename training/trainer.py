"""Network trainer for AlphaZero-style learning.

Handles:
    - Optimizer creation (SGD with momentum and weight decay)
    - Learning rate scheduling (cosine, step, cosine with warm restarts)
    - Single training step execution against replay buffer batches
    - Checkpoint save/load with full optimizer and scheduler state
    - Loss tracking and logging
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from nn.model import HexTTTNet
from training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class Trainer:
    """Handles network training from replay buffer data.

    Manages:
        - SGD optimizer with momentum and weight decay (L2 regularisation)
        - Learning rate scheduling (cosine, step, cosine_warm_restarts)
        - Training step execution with multi-head loss
        - Checkpointing (save/load model, optimizer, scheduler, step counter)
        - Loss tracking
    """

    def __init__(
        self,
        network: HexTTTNet,
        config: dict,
        device: str = "cpu",
    ) -> None:
        """Initialize the trainer.

        Args:
            network: the HexTTTNet model to train.
            config: merged configuration dict (should contain ``training`` key).
            device: torch device string (``'cpu'``, ``'cuda'``, ``'mps'``).
        """
        self.network = network.to(device)
        self.device = device
        self.config = config
        self.train_config: dict = config.get("training", {})
        self.global_step: int = 0

        # Loss weight config (passed to model.loss())
        self.loss_config: Dict[str, float] = {
            "value_weight": self.train_config.get("value_weight", 1.0),
            "policy_weight": self.train_config.get("policy_weight", 1.0),
            "ownership_weight": self.train_config.get("ownership_weight", 0.15),
            "threat_weight": self.train_config.get("threat_weight", 0.15),
        }

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        logger.info(
            "Trainer initialized: device=%s, lr=%.6f, weight_decay=%.6f, "
            "schedule=%s",
            device,
            self.train_config.get("learning_rate", 0.02),
            self.train_config.get("weight_decay", 1e-4),
            self.train_config.get("lr_schedule", "cosine"),
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create SGD optimizer with momentum and weight decay (L2 regularisation).

        Returns:
            Configured ``torch.optim.SGD`` optimizer.
        """
        lr = self.train_config.get("learning_rate", 0.02)
        momentum = self.train_config.get("momentum", 0.9)
        weight_decay = self.train_config.get("weight_decay", 1e-4)

        optimizer = optim.SGD(
            self.network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        logger.debug(
            "Created SGD optimizer: lr=%.6f, momentum=%.2f, weight_decay=%.6f, nesterov=True",
            lr, momentum, weight_decay,
        )
        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Create a learning rate scheduler based on the config.

        Supported schedules:
            - ``cosine``: Cosine annealing to ``lr_min``.
            - ``step``: Step decay at specified step counts.
            - ``cosine_warm_restarts``: Cosine annealing with periodic warm restarts.

        Returns:
            An LR scheduler instance, or ``None`` if no schedule is configured.
        """
        schedule_type = self.train_config.get("lr_schedule", "cosine")

        if schedule_type == "cosine":
            # Total steps: num_iterations * training_steps_per_iteration (if available),
            # or a large default
            total_steps = self.train_config.get("num_iterations", 100) * \
                self.train_config.get("training_steps_per_iteration", 500)
            lr_min = self.train_config.get("lr_min", 2e-4)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=lr_min,
            )
            logger.debug(
                "Created CosineAnnealingLR scheduler: T_max=%d, eta_min=%.6f",
                total_steps, lr_min,
            )
            return scheduler

        elif schedule_type == "step":
            milestones = self.train_config.get("lr_steps", [50000, 150000])
            gamma = self.train_config.get("lr_gamma", 0.1)

            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma,
            )
            logger.debug(
                "Created MultiStepLR scheduler: milestones=%s, gamma=%.3f",
                milestones, gamma,
            )
            return scheduler

        elif schedule_type == "cosine_warm_restarts":
            restart_period = self.train_config.get("lr_restart_period", 50000)
            lr_min = self.train_config.get("lr_min", 5e-5)

            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=restart_period,
                T_mult=1,
                eta_min=lr_min,
            )
            logger.debug(
                "Created CosineAnnealingWarmRestarts scheduler: T_0=%d, eta_min=%.6f",
                restart_period, lr_min,
            )
            return scheduler

        else:
            logger.warning(
                "Unknown LR schedule type '%s'; no scheduler created.", schedule_type
            )
            return None

    def train_step(self, batch: dict) -> Dict[str, float]:
        """Execute one training step.

        Args:
            batch: dict from ``ReplayBuffer.sample()`` with keys
                ``features``, ``policy``, ``value``, and optionally
                ``ownership`` and ``threats``.

        Returns:
            Dict of scalar loss values:
                ``total``, ``value``, ``policy``, ``ownership``, ``threat``.
        """
        self.network.train()

        # Move tensors to the correct device
        features = batch["features"].to(self.device)        # (B, C, H, W)
        policy_target = batch["policy"].to(self.device)      # (B, H*W)
        value_target = batch["value"].to(self.device)        # (B, 1)

        targets: Dict[str, torch.Tensor] = {
            "policy": policy_target,
            "value": value_target,
        }

        if batch.get("ownership") is not None:
            targets["ownership"] = batch["ownership"].to(self.device)

        if batch.get("threats") is not None:
            targets["threats"] = batch["threats"].to(self.device)

        # Forward pass
        predictions = self.network(features)

        # Compute loss
        losses = self.network.loss(predictions, targets, config=self.loss_config)

        # Backward pass
        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping for stability
        max_grad_norm = self.train_config.get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), max_norm=max_grad_norm
        )

        self.optimizer.step()

        # Step the scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Extract scalar values for logging
        loss_values = {
            k: v.item() for k, v in losses.items()
        }

        if self.global_step % 100 == 0:
            logger.info(
                "Step %d | total=%.4f policy=%.4f value=%.4f "
                "ownership=%.4f threat=%.4f | lr=%.6f",
                self.global_step,
                loss_values["total"],
                loss_values["policy"],
                loss_values["value"],
                loss_values["ownership"],
                loss_values["threat"],
                self.learning_rate,
            )

        return loss_values

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """Save model, optimizer, scheduler, and global step to a checkpoint file.

        Args:
            path: file path for the checkpoint.
            extra: optional dict of additional data to save alongside the
                checkpoint (e.g. iteration number, replay buffer stats).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint and restore model, optimizer, scheduler, and step counter.

        Args:
            path: file path to the checkpoint.

        Returns:
            The ``extra`` dict that was saved alongside the checkpoint,
            or an empty dict if none was present.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        extra = checkpoint.get("extra", {})

        logger.info(
            "Loaded checkpoint from %s (step %d)", path, self.global_step
        )

        return extra

    @property
    def learning_rate(self) -> float:
        """Current learning rate from the optimizer."""
        return self.optimizer.param_groups[0]["lr"]

"""Training loop implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for autoencoder models.

    Features:
    - Mixed precision training (optional)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - W&B logging

    Args:
        model: The autoencoder model
        cfg: Hydra configuration
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device | str = "auto",
    ):
        self.model = model
        self.cfg = cfg

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Loss function
        self.criterion = nn.MSELoss()

        # Phase/amplitude loss weighting
        # Default: equal weights, but amplitude loss can be down-weighted if it dominates
        self.lambda_phase = getattr(cfg.training, "lambda_phase", 1.0)
        self.lambda_amplitude = getattr(cfg.training, "lambda_amplitude", 1.0)
        self.lambda_unit_circle = getattr(cfg.training, "lambda_unit_circle", 0.1)

        # Phase channels (2 for cos/sin, 3 if including amplitude)
        self.phase_channels = getattr(cfg.model.phase, "include_amplitude", False) and 3 or 2

        # Gradient clipping for training stability
        self.gradient_clip_norm = getattr(cfg.training, "gradient_clip_norm", 1.0)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.initial_val_loss = None  # Track for preflight check

        # W&B
        self.use_wandb = cfg.logging.wandb.enabled and WANDB_AVAILABLE

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Configure optimizer from config."""
        opt_cfg = self.cfg.training.optimizer
        if opt_cfg.name.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Configure learning rate scheduler from config."""
        sched_cfg = self.cfg.training.scheduler
        if sched_cfg.name is None:
            return None
        elif sched_cfg.name.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs - sched_cfg.warmup_epochs,
                eta_min=sched_cfg.min_lr,
            )
        elif sched_cfg.name.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            data = batch["data"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            reconstruction, latent = self.model(data)

            # Compute masked loss
            loss = self._compute_masked_loss(data, reconstruction, mask)

            # NaN/Inf detection - stop immediately if training is unstable
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {n_batches}! Training is unstable.")
                raise RuntimeError(
                    "Training failed: NaN/Inf loss detected. "
                    "Try lowering learning rate or increasing gradient clipping."
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / n_batches
        return {"train_loss": avg_loss}

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                data = batch["data"].to(self.device)
                mask = batch["mask"].to(self.device)

                reconstruction, latent = self.model(data)
                loss = self._compute_masked_loss(data, reconstruction, mask)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return {"val_loss": avg_loss}

    def _compute_masked_loss(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss only on valid (non-padded) samples.

        Includes:
        - Separate weighting for phase (cos/sin) vs amplitude
        - Unit-circle regularization: penalize (cos² + sin² - 1)²

        Args:
            target: Ground truth (batch, features, time)
            prediction: Model output (batch, features, time)
            mask: Validity mask (batch, time)

        Returns:
            Total loss (reconstruction + regularization)
        """
        batch_size, total_features, time = target.shape

        # Expand mask to feature dimension
        mask_expanded = mask.unsqueeze(1).expand_as(target)
        valid_count = mask_expanded.sum()

        # Infer n_channels from total features and phase_channels
        n_channels = total_features // self.phase_channels

        # Reshape to (batch, n_channels, phase_channels, time)
        target_reshaped = target.view(batch_size, n_channels, self.phase_channels, time)
        pred_reshaped = prediction.view(batch_size, n_channels, self.phase_channels, time)
        mask_phase = mask.unsqueeze(1).unsqueeze(2).expand(batch_size, n_channels, self.phase_channels, time)

        # Compute phase loss (cos/sin channels: indices 0 and 1)
        cos_target, sin_target = target_reshaped[:, :, 0, :], target_reshaped[:, :, 1, :]
        cos_pred, sin_pred = pred_reshaped[:, :, 0, :], pred_reshaped[:, :, 1, :]

        phase_error = ((cos_target - cos_pred) ** 2 + (sin_target - sin_pred) ** 2)
        phase_mask = mask.unsqueeze(1).expand(batch_size, n_channels, time)
        phase_loss = (phase_error * phase_mask).sum() / (phase_mask.sum() * 2 + 1e-8)

        total_loss = self.lambda_phase * phase_loss

        # Compute amplitude loss if present (index 2)
        if self.phase_channels == 3:
            amp_target = target_reshaped[:, :, 2, :]
            amp_pred = pred_reshaped[:, :, 2, :]
            amp_error = (amp_target - amp_pred) ** 2
            amp_loss = (amp_error * phase_mask).sum() / (phase_mask.sum() + 1e-8)
            total_loss = total_loss + self.lambda_amplitude * amp_loss

        # Unit-circle regularization: penalize (cos² + sin² - 1)²
        # This prevents the model from "cheating" by shrinking cos/sin magnitudes
        if self.lambda_unit_circle > 0:
            unit_violation = (cos_pred ** 2 + sin_pred ** 2 - 1) ** 2
            unit_loss = (unit_violation * phase_mask).sum() / (phase_mask.sum() + 1e-8)
            total_loss = total_loss + self.lambda_unit_circle * unit_loss

        return total_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints

        Returns:
            History dictionary with losses per epoch
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])

                # Track initial val loss for preflight check
                if self.initial_val_loss is None:
                    self.initial_val_loss = val_metrics["val_loss"]
                    logger.info(f"Initial validation loss: {self.initial_val_loss:.4f}")

                # Check for improvement
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.epochs_without_improvement = 0

                    # Save best model
                    if checkpoint_dir and self.cfg.training.checkpointing.save_best:
                        self.save_checkpoint(checkpoint_dir / "best.pt")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.cfg.training.early_stopping.enabled:
                    if self.epochs_without_improvement >= self.cfg.training.early_stopping.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau) and val_loader:
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Periodic checkpointing
            save_every = self.cfg.training.checkpointing.save_every_n_epochs
            if checkpoint_dir and save_every and (epoch + 1) % save_every == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

            # Logging
            log_msg = f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}"
            if val_loader:
                log_msg += f", val_loss={val_metrics['val_loss']:.4f}"
            logger.info(log_msg)

            if self.use_wandb:
                log_dict = {"epoch": epoch, **train_metrics}
                if val_loader:
                    log_dict.update(val_metrics)
                wandb.log(log_dict)

        # Save final model
        if checkpoint_dir and self.cfg.training.checkpointing.save_last:
            self.save_checkpoint(checkpoint_dir / "last.pt")

        # Preflight checks: verify training actually learned something
        if val_loader is not None and self.initial_val_loss is not None:
            improvement_pct = (self.initial_val_loss - self.best_val_loss) / self.initial_val_loss * 100
            logger.info(f"Training improvement: {improvement_pct:.1f}% (initial: {self.initial_val_loss:.4f}, best: {self.best_val_loss:.4f})")

            if improvement_pct < 5.0:
                logger.warning(
                    f"PREFLIGHT WARNING: Validation loss only improved by {improvement_pct:.1f}% "
                    f"(threshold: 5%). The model may not have learned meaningful representations. "
                    f"Consider training longer or adjusting hyperparameters."
                )

        return history

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": dict(self.cfg),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def check_latent_collapse(self, dataloader: DataLoader, threshold: float = 0.01) -> tuple[bool, float]:
        """
        Check if latent space has collapsed (all outputs nearly identical).

        Args:
            dataloader: Data loader to check
            threshold: Std threshold below which we consider collapsed

        Returns:
            Tuple of (is_collapsed, mean_std)
        """
        self.model.eval()
        all_latents = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch["data"].to(self.device)
                _, latent = self.model(data)
                # Take mean over time dimension to get segment-level embedding
                latent_mean = latent.mean(dim=1)
                all_latents.append(latent_mean.cpu())

        all_latents = torch.cat(all_latents, dim=0)
        mean_std = all_latents.std(dim=0).mean().item()

        is_collapsed = mean_std < threshold
        if is_collapsed:
            logger.warning(
                f"PREFLIGHT WARNING: Latent space may have collapsed! "
                f"Mean std={mean_std:.6f} (threshold: {threshold}). "
                f"The model is not learning diverse representations."
            )
        else:
            logger.info(f"Latent space check: mean std={mean_std:.4f} (OK)")

        return is_collapsed, mean_std

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

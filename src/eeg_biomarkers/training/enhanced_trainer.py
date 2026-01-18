"""Enhanced training loop with contrastive loss."""

from __future__ import annotations

import logging
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for learning discriminative representations.

    Pulls together samples from the same class (same subject or same condition)
    and pushes apart samples from different classes.

    This helps the autoencoder learn features that are actually useful for
    distinguishing MCI from HC, not just good at reconstruction.
    """

    def __init__(self, temperature: float = 0.07, mode: str = "subject"):
        """
        Args:
            temperature: Temperature for softmax scaling (lower = harder contrasts)
            mode: "subject" (same subject = positive) or "condition" (same HC/MCI = positive)
        """
        super().__init__()
        self.temperature = temperature
        self.mode = mode

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Latent representations (batch, hidden_size)
            labels: Class labels (batch,) - subject IDs or condition labels

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Count positive pairs for each sample
        pos_count = mask.sum(dim=1)

        # For samples with no positive pairs, skip them
        valid_samples = pos_count > 0

        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Compute log-softmax for each row
        # Subtract max for numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

        # Mask out self-similarity
        self_mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))

        # Log-sum-exp over all negatives + positives
        log_sum_exp = torch.logsumexp(sim_matrix, dim=1)

        # Sum of positive similarities
        pos_sim = (sim_matrix * mask).sum(dim=1)

        # Contrastive loss: -log(sum of positives / sum of all)
        # = -sum_pos + log_sum_exp
        loss_per_sample = -pos_sim / (pos_count + 1e-8) + log_sum_exp

        # Average over valid samples only
        loss = loss_per_sample[valid_samples].mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for learning discriminative representations.

    For each anchor, finds a positive (same class) and negative (different class),
    and ensures the anchor is closer to the positive than the negative by a margin.
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Minimum margin between positive and negative distances
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with hard negative mining.

        Args:
            embeddings: Latent representations (batch, hidden_size)
            labels: Class labels (batch,)

        Returns:
            Triplet loss
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create mask for positive and negative pairs
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float()
        neg_mask = 1.0 - pos_mask

        # Remove diagonal
        pos_mask = pos_mask - torch.eye(batch_size, device=pos_mask.device)

        # Hard positive: furthest positive
        pos_distances = distances * pos_mask
        pos_distances[pos_mask == 0] = 0
        hardest_pos, _ = pos_distances.max(dim=1)

        # Hard negative: closest negative
        neg_distances = distances * neg_mask
        neg_distances[neg_mask == 0] = float('inf')
        hardest_neg, _ = neg_distances.min(dim=1)

        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only count valid triplets (where we have both pos and neg)
        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        if not valid.any():
            return torch.tensor(0.0, device=embeddings.device)

        return loss[valid].mean()


class EnhancedTrainer:
    """
    Enhanced training loop with contrastive loss.

    Features:
    - Reconstruction loss (MSE on phase/amplitude)
    - Contrastive loss (pulls same-class together, pushes different apart)
    - Unit-circle regularization
    - Mixed precision training (optional)
    - Learning rate scheduling
    - Early stopping
    - Checkpointing

    Args:
        model: The autoencoder model
        cfg: Hydra configuration
        device: Device to train on
        contrastive_mode: "subject" or "condition" for contrastive loss
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        device: torch.device | str = "auto",
        contrastive_mode: str = "condition",
    ):
        self.model = model
        self.cfg = cfg
        self.contrastive_mode = contrastive_mode

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

        # Loss functions
        self.reconstruction_criterion = nn.MSELoss(reduction='none')
        self.contrastive_loss = ContrastiveLoss(
            temperature=getattr(cfg.training, 'contrastive_temperature', 0.07),
            mode=contrastive_mode,
        )
        self.triplet_loss = TripletLoss(
            margin=getattr(cfg.training, 'triplet_margin', 0.5),
        )

        # Loss weights
        self.lambda_reconstruction = getattr(cfg.training, 'lambda_reconstruction', 1.0)
        self.lambda_phase = getattr(cfg.training, 'lambda_phase', 1.0)
        self.lambda_amplitude = getattr(cfg.training, 'lambda_amplitude', 1.0)
        self.lambda_unit_circle = getattr(cfg.training, 'lambda_unit_circle', 0.1)
        self.lambda_contrastive = getattr(cfg.training, 'lambda_contrastive', 0.1)
        self.lambda_triplet = getattr(cfg.training, 'lambda_triplet', 0.0)  # Off by default

        # Phase channels
        self.phase_channels = getattr(cfg.model.phase, 'include_amplitude', False) and 3 or 2

        # Gradient clipping
        self.gradient_clip_norm = getattr(cfg.training, 'gradient_clip_norm', 1.0)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.initial_val_loss = None

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

    def _compute_reconstruction_loss(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute reconstruction loss with separate phase/amplitude components.

        Returns:
            total_loss: Total reconstruction loss
            loss_dict: Dictionary of individual loss components
        """
        batch_size, total_features, time = target.shape
        n_channels = total_features // self.phase_channels

        # Reshape to (batch, n_channels, phase_channels, time)
        target_reshaped = target.view(batch_size, n_channels, self.phase_channels, time)
        pred_reshaped = prediction.view(batch_size, n_channels, self.phase_channels, time)

        # Phase mask
        phase_mask = mask.unsqueeze(1).expand(batch_size, n_channels, time)

        # Phase loss (cos/sin)
        cos_target, sin_target = target_reshaped[:, :, 0, :], target_reshaped[:, :, 1, :]
        cos_pred, sin_pred = pred_reshaped[:, :, 0, :], pred_reshaped[:, :, 1, :]

        phase_error = ((cos_target - cos_pred) ** 2 + (sin_target - sin_pred) ** 2)
        phase_loss = (phase_error * phase_mask).sum() / (phase_mask.sum() * 2 + 1e-8)

        total_loss = self.lambda_phase * phase_loss
        loss_dict = {'phase_loss': phase_loss.item()}

        # Amplitude loss if present
        if self.phase_channels == 3:
            amp_target = target_reshaped[:, :, 2, :]
            amp_pred = pred_reshaped[:, :, 2, :]
            amp_error = (amp_target - amp_pred) ** 2
            amp_loss = (amp_error * phase_mask).sum() / (phase_mask.sum() + 1e-8)
            total_loss = total_loss + self.lambda_amplitude * amp_loss
            loss_dict['amplitude_loss'] = amp_loss.item()

        # Unit-circle regularization
        if self.lambda_unit_circle > 0:
            unit_violation = (cos_pred ** 2 + sin_pred ** 2 - 1) ** 2
            unit_loss = (unit_violation * phase_mask).sum() / (phase_mask.sum() + 1e-8)
            total_loss = total_loss + self.lambda_unit_circle * unit_loss
            loss_dict['unit_circle_loss'] = unit_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in pbar:
            data = batch["data"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Get labels for contrastive loss
            if self.contrastive_mode == "subject":
                # Use subject_id as contrastive target
                labels = batch.get("subject_id", None)
                if labels is not None and isinstance(labels[0], str):
                    # Convert string subject IDs to integers
                    unique_subjects = list(set(labels))
                    labels = torch.tensor([unique_subjects.index(s) for s in labels])
            else:
                # Use condition label (HC=0, MCI=1)
                labels = batch.get("label", None)

            if labels is not None:
                labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else labels

            # Forward pass
            reconstruction, latent = self.model(data)

            # Reconstruction loss
            recon_loss, loss_dict = self._compute_reconstruction_loss(data, reconstruction, mask)

            # Contrastive loss on mean latent embedding
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if labels is not None and self.lambda_contrastive > 0:
                # Use mean latent over time as embedding
                mean_latent = latent.mean(dim=1)  # (batch, hidden_size)
                contrastive_loss = self.contrastive_loss(mean_latent, labels)
                loss_dict['contrastive_loss'] = contrastive_loss.item()

            # Triplet loss (optional)
            triplet_loss = torch.tensor(0.0, device=self.device)
            if labels is not None and self.lambda_triplet > 0:
                mean_latent = latent.mean(dim=1)
                triplet_loss = self.triplet_loss(mean_latent, labels)
                loss_dict['triplet_loss'] = triplet_loss.item()

            # Total loss
            loss = (
                self.lambda_reconstruction * recon_loss +
                self.lambda_contrastive * contrastive_loss +
                self.lambda_triplet * triplet_loss
            )

            # NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {n_batches}!")
                raise RuntimeError("Training failed: NaN/Inf loss detected.")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'contr': contrastive_loss.item(),
            })

        return {
            'train_loss': total_loss / n_batches,
            'train_recon_loss': total_recon_loss / n_batches,
            'train_contrastive_loss': total_contrastive_loss / n_batches,
        }

    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                data = batch["data"].to(self.device)
                mask = batch["mask"].to(self.device)

                reconstruction, latent = self.model(data)
                recon_loss, _ = self._compute_reconstruction_loss(data, reconstruction, mask)

                total_loss += recon_loss.item()
                total_recon_loss += recon_loss.item()
                n_batches += 1

        return {
            'val_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches,
        }

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

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "train_contrastive_loss": [],
        }

        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_recon_loss"].append(train_metrics["train_recon_loss"])
            history["train_contrastive_loss"].append(train_metrics["train_contrastive_loss"])

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])

                if self.initial_val_loss is None:
                    self.initial_val_loss = val_metrics["val_loss"]
                    logger.info(f"Initial validation loss: {self.initial_val_loss:.4f}")

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.epochs_without_improvement = 0

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
            log_msg = (
                f"Epoch {epoch}: loss={train_metrics['train_loss']:.4f}, "
                f"recon={train_metrics['train_recon_loss']:.4f}, "
                f"contr={train_metrics['train_contrastive_loss']:.4f}"
            )
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

        # Preflight check
        if val_loader is not None and self.initial_val_loss is not None:
            improvement_pct = (self.initial_val_loss - self.best_val_loss) / self.initial_val_loss * 100
            logger.info(
                f"Training improvement: {improvement_pct:.1f}% "
                f"(initial: {self.initial_val_loss:.4f}, best: {self.best_val_loss:.4f})"
            )

            if improvement_pct < 5.0:
                logger.warning(
                    f"PREFLIGHT WARNING: Validation loss only improved by {improvement_pct:.1f}%"
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

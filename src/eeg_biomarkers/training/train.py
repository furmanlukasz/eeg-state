"""Main training entry point with Hydra configuration."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from eeg_biomarkers.models import ConvLSTMAutoencoder, TransformerAutoencoder
from eeg_biomarkers.data import EEGDataModule
from eeg_biomarkers.training.trainer import Trainer
from eeg_biomarkers.training.enhanced_trainer import EnhancedTrainer

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def setup_wandb(cfg: DictConfig) -> bool:
    """Initialize Weights & Biases.

    Returns:
        True if WandB was successfully initialized, False otherwise.
    """
    if not cfg.logging.wandb.enabled:
        logger.info("WandB logging disabled")
        return False

    if not WANDB_AVAILABLE:
        logger.warning("WandB requested but not installed")
        return False

    try:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return True
    except Exception as e:
        logger.warning(f"WandB initialization failed: {e}")
        logger.info("Continuing without WandB logging")
        return False


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Usage:
        # Default training
        python -m eeg_biomarkers.training.train

        # Override config
        python -m eeg_biomarkers.training.train model=complex training.epochs=200

        # Multi-run sweep
        python -m eeg_biomarkers.training.train --multirun model=base,complex
    """
    # Setup
    setup_logging(cfg)
    set_seed(cfg.experiment.seed)
    wandb_enabled = setup_wandb(cfg)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Resolve paths
    data_dir = Path(cfg.paths.data_dir)
    output_dir = Path(cfg.paths.output_dir)
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_module = EEGDataModule(cfg, data_dir)
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create model based on config
    logger.info("Creating model...")
    n_channels = data_module.n_channels
    model_name = getattr(cfg.model, 'name', 'convlstm_autoencoder')

    if model_name == "transformer_autoencoder":
        model = TransformerAutoencoder.from_config(cfg.model, n_channels)
        logger.info("Using TransformerAutoencoder (self-attention)")
    else:
        model = ConvLSTMAutoencoder.from_config(cfg.model, n_channels)
        logger.info("Using ConvLSTMAutoencoder (LSTM)")

    logger.info(f"Model: {model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Create trainer - use EnhancedTrainer for transformer or if contrastive loss enabled
    use_enhanced = (
        model_name == "transformer_autoencoder" or
        getattr(cfg.training, 'lambda_contrastive', 0.0) > 0
    )

    if use_enhanced:
        contrastive_mode = getattr(cfg.training, 'contrastive_mode', 'condition')
        trainer = EnhancedTrainer(model, cfg, device=cfg.experiment.device, contrastive_mode=contrastive_mode)
        logger.info(f"Using EnhancedTrainer with contrastive loss (mode={contrastive_mode})")
    else:
        trainer = Trainer(model, cfg, device=cfg.experiment.device)
        logger.info("Using standard Trainer")

    # Resume from checkpoint if specified
    resume_from = getattr(cfg.training, 'resume_from', None)
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=cfg.experiment.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model weights loaded from checkpoint")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")

    # Train
    logger.info("Starting training...")
    history = trainer.fit(
        train_loader,
        val_loader,
        checkpoint_dir=model_dir,
    )

    # Log final metrics
    logger.info(f"Training complete. Best val loss: {trainer.best_val_loss:.4f}")

    if wandb_enabled:
        wandb.finish()

    return trainer.best_val_loss


if __name__ == "__main__":
    main()

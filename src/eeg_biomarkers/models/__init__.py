"""Model architectures for EEG representation learning."""

from eeg_biomarkers.models.autoencoder import ConvLSTMAutoencoder
from eeg_biomarkers.models.encoder import ConvLSTMEncoder
from eeg_biomarkers.models.decoder import ConvLSTMDecoder
from eeg_biomarkers.models.transformer_autoencoder import TransformerAutoencoder
from eeg_biomarkers.models.transformer_encoder import TemporalTransformerEncoder
from eeg_biomarkers.models.transformer_decoder import TemporalTransformerDecoder

__all__ = [
    "ConvLSTMAutoencoder",
    "ConvLSTMEncoder",
    "ConvLSTMDecoder",
    "TransformerAutoencoder",
    "TemporalTransformerEncoder",
    "TemporalTransformerDecoder",
]

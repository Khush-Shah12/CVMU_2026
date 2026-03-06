"""
models/anomaly_model.py
=======================
Anomaly-detection models for the Synthetic Data Validator:

1. IsolationForestDetector  – sklearn-based, fast, interpretable
2. AutoencoderDetector      – PyTorch-based, captures non-linear patterns
3. AnomalyEnsemble          – weighted combination of both detectors

Workflow:
  fit(training_data)  →  predict(new_data)  →  anomaly labels + scores
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config


# ============================================================================
# 1. Isolation Forest Detector
# ============================================================================

class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest.

    Scores are normalised to [0, 1] where 1 = most anomalous.
    """

    def __init__(
        self,
        contamination: float = config.ISOLATION_FOREST_CONTAMINATION,
        n_estimators: int = config.ISOLATION_FOREST_N_ESTIMATORS,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, data: np.ndarray) -> "IsolationForestDetector":
        """Fit on *normal* transaction data (n_samples × n_features)."""
        scaled = self.scaler.fit_transform(data)
        self.model.fit(scaled)
        self._fitted = True
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        scaled = self.scaler.transform(data)
        preds = self.model.predict(scaled)  # -1 = anomaly, 1 = normal
        return (preds == -1).astype(int)

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores in [0, 1] for each sample.
        Higher = more anomalous.
        """
        scaled = self.scaler.transform(data)
        raw = self.model.decision_function(scaled)  # lower = more anomalous
        # Normalise: flip sign, then min-max scale to [0, 1]
        flipped = -raw
        mn, mx = flipped.min(), flipped.max()
        if mx - mn < 1e-9:
            return np.zeros(len(data))
        return (flipped - mn) / (mx - mn)


# ============================================================================
# 2. Autoencoder Detector (PyTorch)
# ============================================================================

class _AENetwork(nn.Module):
    """Simple symmetric Autoencoder network."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """
    Anomaly detection via reconstruction error.

    Trained on *normal* data; anomalies produce higher reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = config.VAE_INPUT_DIM,
        hidden_dim: int = config.AUTOENCODER_HIDDEN_DIM,
        latent_dim: int = config.AUTOENCODER_LATENT_DIM,
        epochs: int = config.AUTOENCODER_EPOCHS,
        lr: float = config.AUTOENCODER_LR,
        batch_size: int = config.AUTOENCODER_BATCH_SIZE,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model: _AENetwork | None = None
        self._threshold: float = 0.0  # anomaly threshold (set after fit)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, data: np.ndarray) -> "AutoencoderDetector":
        """Train autoencoder on normal transaction data."""
        scaled = self.scaler.fit_transform(data)
        tensor = torch.FloatTensor(scaled).to(self._device)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)

        self.model = _AENetwork(self.input_dim, self.hidden_dim, self.latent_dim).to(self._device)
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="none")

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch).mean()
                loss.backward()
                optimiser.step()

        # Compute threshold as the 95th percentile of training errors
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor)
            errors = ((recon - tensor) ** 2).mean(dim=1).cpu().numpy()
        self._threshold = float(np.percentile(errors, 95))
        return self

    def _reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """Return per-sample mean squared reconstruction error."""
        scaled = self.scaler.transform(data)
        tensor = torch.FloatTensor(scaled).to(self._device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor)
            errors = ((recon - tensor) ** 2).mean(dim=1).cpu().numpy()
        return errors

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        errors = self._reconstruction_error(data)
        return (errors > self._threshold).astype(int)

    def score(self, data: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]. Higher = more anomalous."""
        errors = self._reconstruction_error(data)
        mn, mx = errors.min(), errors.max()
        if mx - mn < 1e-9:
            return np.zeros(len(data))
        return (errors - mn) / (mx - mn)


# ============================================================================
# 3. Ensemble detector
# ============================================================================

class AnomalyEnsemble:
    """
    Combines Isolation Forest and Autoencoder scores with configurable weights.
    """

    def __init__(
        self,
        input_dim: int = config.VAE_INPUT_DIM,
        if_weight: float = config.ANOMALY_ENSEMBLE_IF_WEIGHT,
        ae_weight: float = config.ANOMALY_ENSEMBLE_AE_WEIGHT,
    ):
        self.if_detector = IsolationForestDetector()
        self.ae_detector = AutoencoderDetector(input_dim=input_dim)
        self.if_weight = if_weight
        self.ae_weight = ae_weight

    def fit(self, data: np.ndarray) -> "AnomalyEnsemble":
        """Fit both detectors on standard (normal) data."""
        print("  [Anomaly] Training Isolation Forest ...")
        self.if_detector.fit(data)
        print("  [Anomaly] Training Autoencoder ...")
        self.ae_detector.fit(data)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return binary labels (1 = anomaly) using combined score > 0.5."""
        combined = self.score(data)
        return (combined > 0.5).astype(int)

    def score(self, data: np.ndarray) -> np.ndarray:
        """Return weighted ensemble anomaly score in [0, 1]."""
        if_scores = self.if_detector.score(data)
        ae_scores = self.ae_detector.score(data)
        return self.if_weight * if_scores + self.ae_weight * ae_scores

    def summary(self, data: np.ndarray) -> dict:
        """Return a human-readable anomaly summary dict."""
        labels = self.predict(data)
        scores = self.score(data)
        n_anomalies = int(labels.sum())
        return {
            "total_samples": len(data),
            "anomalies_detected": n_anomalies,
            "anomaly_rate": round(n_anomalies / max(len(data), 1) * 100, 2),
            "mean_anomaly_score": round(float(scores.mean()), 4),
            "max_anomaly_score": round(float(scores.max()), 4),
        }

"""
models/gan_model.py
===================
Variational Autoencoder (VAE) for synthetic financial data generation.

Architecture:
  Encoder  →  μ, log σ²  →  reparameterise  →  Decoder  →  reconstructed row

The VAE learns the latent distribution of normalised financial features so we
can sample new, realistic rows from the latent space.

Why VAE over GAN:
  • More stable training (no mode-collapse risk).
  • Explicit latent space makes it easy to interpolate / control output.
  • Reconstruction loss gives a clear training signal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config


# ============================================================================
# Encoder
# ============================================================================

class Encoder(nn.Module):
    """Maps input features → (μ, log σ²) of the latent Gaussian."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
        )
        # Separate heads for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """Maps latent vector z → reconstructed input features."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # output in [0, 1] (data is min-max normalised)
        )

    def forward(self, z: torch.Tensor):
        return self.net(z)


# ============================================================================
# VAE (full model)
# ============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder combining Encoder + Decoder.

    Loss = Reconstruction (BCE) + KL divergence
    """

    def __init__(
        self,
        input_dim: int = config.VAE_INPUT_DIM,
        hidden_dim: int = config.VAE_HIDDEN_DIM,
        latent_dim: int = config.VAE_LATENT_DIM,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # Reparameterisation trick: z = μ + σ * ε,  ε ~ N(0, 1)
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    # ------------------------------------------------------------------
    # Sample new rows from the latent space
    # ------------------------------------------------------------------
    def sample(self, n: int, device: str = "cpu") -> np.ndarray:
        """Generate *n* synthetic rows by sampling z ~ N(0, I)."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            generated = self.decoder(z)
        return generated.cpu().numpy()


# ============================================================================
# Loss function
# ============================================================================

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    ELBO loss = Reconstruction (MSE) + KL divergence.
    Using MSE instead of BCE for continuous financial features.
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


# ============================================================================
# Training routine
# ============================================================================

def train_vae(
    data: np.ndarray,
    input_dim: int = config.VAE_INPUT_DIM,
    hidden_dim: int = config.VAE_HIDDEN_DIM,
    latent_dim: int = config.VAE_LATENT_DIM,
    epochs: int = config.VAE_EPOCHS,
    batch_size: int = config.VAE_BATCH_SIZE,
    lr: float = config.VAE_LEARNING_RATE,
) -> VAE:
    """
    Train a VAE on the supplied numeric data matrix.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, input_dim)
        Min-max normalised feature matrix.

    Returns
    -------
    VAE : trained model (in eval mode).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare DataLoader
    tensor_data = torch.FloatTensor(data).to(device)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            x_hat, mu, logvar = model(batch)
            loss = vae_loss(batch, x_hat, mu, logvar)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(dataset)
            print(f"  [VAE] Epoch {epoch:>3}/{epochs}  Loss: {avg:.4f}")

    model.eval()
    return model


# ============================================================================
# Generation helper
# ============================================================================

def generate_from_vae(model: VAE, n: int) -> np.ndarray:
    """
    Generate *n* synthetic rows using a trained VAE.

    Returns
    -------
    np.ndarray of shape (n, input_dim) with values in [0, 1].
    Caller is responsible for denormalising back to original feature ranges.
    """
    device = next(model.parameters()).device
    return model.sample(n, device=str(device))

"""Variational Autoencoder (VAE) module."""
from torch import nn

from vae.components import Encoder
from vae.components import Decoder

class VAE(nn.Module):
    """Variational Autoencoder (VAE) module."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """Forward pass for the VAE
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (N, 3, H, W)
            torch.Tensor: Encoded tensor of shape (N, 8, H/8, W/8)
        """
        # x: (N, 3, H, W)

        # (N, 3, H, W) -> (N, 8, H/8, W/8)
        encoded, mean, log_variance = self.encoder(x)
        # (N, 8, H/8, W/8) -> (N, 3, H, W)
        decoded = self.decoder(encoded)
        # (N, 3, H, W), (N, 8, H/8, W/8)
        return decoded, encoded

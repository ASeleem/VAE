"""Decoder module for a Variational Autoencoder (VAE)."""
from torch import nn

from vae.attention import AttentionBlock
from vae.components import ResidualBlock

class Decoder(nn.Sequential):
    """Decoder module for a Variational Autoencoder (VAE)."""
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """Forward pass for the Decoder
        Args:
            x (torch.Tensor): Input tensor of shape (N, 4, H/8, W/8)
        Returns:
            torch.Tensor: Output tensor of shape (N, 3, H, W)
        """
        # x: (N, 4, H/8, W/8)
        x /= 0.18215
        for module in self:
            x = module(x)
        # (N, 3, H, W)
        return x

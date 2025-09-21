"""Encoder module for the VAE."""
import torch
from torch import nn
import torch.nn.functional as F

from vae.attention import AttentionBlock
from vae.components import ResidualBlock

class Encoder(nn.Sequential):
    """Encoder module for the VAE."""
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x):
        """Forward pass for the Encoder
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, 8, H/8, W/8)
        """
        # x: (N, 3, H, W)
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)

        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)

        x = mean + eps * std
        x *= 0.18215

        # (N, 8, H/8, W/8)
        return x, mean, log_variance

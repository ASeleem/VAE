"""Residual Block Module for VAE
This module implements a Residual Block with Group Normalization
and Convolutional Layers, commonly used in Variational Autoencoders (VAEs)."""
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual Block with Group Normalization and Convolutional Layers
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Forward pass for the Residual Block
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W)
        """
        # x: (N, C, H, W)
        residue = x.clone()
        # (N, C, H, W) -> (N, C, H, W)
        x = self.groupnorm1(x)
        # (N, C, H, W) -> (N, C, H, W)
        x = F.selu(x)
        # (N, C, H, W) -> (N, C, H, W)
        x = self.conv1(x)
        # (N, C, H, W) -> (N, C, H, W)
        x = self.groupnorm2(x)
        # (N, C, H, W) -> (N, C, H, W)
        x = self.conv2(x)
        # (N, C, H, W) + (N, C, H, W) -> (N, C, H, W)
        x = x + self.residual_layer(residue)
        return x

"""Attention Block that combines Residual Connections,
Group Normalization, and Self-Attention
to process the input features effectively."""
from torch import nn

from vae.attention import SelfAttention


class AttentionBlock(nn.Module):
    """Attention Block that combines Residual Connections,
    Group Normalization, and Self-Attention
    to process the input features effectively.
    Args:
        channels (int): Number of input channels
    """
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        """Forward pass for the Attention Block
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W)
        """
        # x: (N, C, H, W)
        residual = x.clone()
        # (N, C, H, W) -> (N, C, H, W)
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # (N, C, H, W) -> (N, C, H*W)
        x = x.view((n, c, h * w))
        # (N, C, H*W) -> (N, H*W, C)
        x = x.transpose(-1, -2)
        # (N, H*W, C) -> (N, H*W, C)
        x = self.attention(x)
        # (N, H*W, C) -> (N, C, H*W)
        x = x.transpose(-1, -2)
        # (N, C, H*W) -> (N, C, H, W)
        x = x.view((n, c, h, w))
        # (N, C, H, W) + (N, C, H, W) -> (N, C, H, W)
        x += residual
        # (N, C, H, W)
        return x

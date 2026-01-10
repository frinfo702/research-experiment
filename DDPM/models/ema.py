"""
Exponential Moving Average (EMA) for model parameters.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that are updated with
    exponential moving average at each training step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
        """
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            ema_param.data.lerp_(model_param.data, 1 - self.decay)

        # Also update buffers (e.g., batch norm running stats)
        for ema_buf, model_buf in zip(
            self.shadow.buffers(), model.buffers()
        ):
            ema_buf.data.copy_(model_buf.data)

    def state_dict(self):
        """Return EMA state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        """Forward pass using EMA model."""
        return self.shadow(*args, **kwargs)

    def eval(self):
        """Set EMA model to eval mode."""
        self.shadow.eval()
        return self

    def train(self):
        """EMA model should always be in eval mode."""
        return self

    def to(self, device):
        """Move EMA model to device."""
        self.shadow.to(device)
        return self


"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implemented from scratch using Bernoulli sampling.

    Does NOT use torch.nn.Dropout or torch.nn.functional.dropout.
    During training, each neuron is zeroed with probability p and the remaining
    activations are scaled by 1/(1-p) (inverted dropout) so that the expected
    value of the output is unchanged — no rescaling is needed at inference time.
    During eval mode the layer is a pure identity (standard behaviour).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype))

        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
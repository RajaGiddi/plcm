"""
Feed-forward MLP backbone — the architecture brittleness control.

Drop-in replacement for the LSTM encoder used by PLCM, exposing the same
interface (a `lstm`-shaped call returning `(output, (h_n, c_n))`) so the
trainer, adapters, and readout paths are untouched. Only the backbone changes,
which is what makes the brittleness comparison single-variable.

See docs/BRITTLENESS_prereg.md. Note R1 there: an MLP's first layer is itself an
unconstrained per-input-unit linear map, so an input adapter is partially
redundant with it in a way it is not for a recurrent accumulator.
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """784 -> hidden -> hidden, ReLU, returned in LSTM-compatible shape.

    The PLCM forward path consumes `lstm_out[:, -1, :]` and `c_n[-1]`. This
    module returns tensors of those shapes so no downstream code branches on
    architecture:
        output: [batch, 1, hidden]      (single "timestep")
        h_n, c_n: [1, batch, hidden]    (single "layer")

    Args:
        input_dim: flattened input size (784 for MNIST)
        hidden_size: width of both hidden layers and of the returned state
    """

    def __init__(self, input_dim: int = 784, hidden_size: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x: [batch, seq_len, feat] (flattened internally) or [batch, input_dim]
            hidden: ignored; present for interface compatibility

        Returns:
            output: [batch, 1, hidden]
            (h_n, c_n): each [1, batch, hidden]
        """
        b = x.shape[0]
        h = self.net(x.reshape(b, -1))          # [batch, hidden]
        # `c` is the state PLCM classifies from; `h` is the "hidden output".
        # For a feed-forward net they are the same vector.
        return h.unsqueeze(1), (h.unsqueeze(0), h.unsqueeze(0))

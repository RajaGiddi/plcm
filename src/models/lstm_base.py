"""
Vanilla LSTM baseline for continual learning comparison.

Standard LSTM with a classification head. No memory augmentation.
This is the model that will exhibit full catastrophic forgetting,
providing the baseline against which PLCM is measured.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMBaseline(nn.Module):
    """
    Standard LSTM for sequence classification.

    Processes input as a sequence (e.g., rows of an image), takes the
    final hidden state, and classifies through a linear head.

    Args:
        input_size: Dimension of each input timestep
        hidden_size: LSTM hidden/cell state dimension
        num_classes: Number of output classes
        num_layers: Number of stacked LSTM layers
        dropout: Dropout between LSTM layers (only if num_layers > 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input sequence [batch, seq_len, input_size]
            hidden: Optional initial (h_0, c_0)

        Returns:
            logits: Classification output [batch, num_classes]
            hidden: Final (h_n, c_n)
        """
        # Process sequence
        output, (h_n, c_n) = self.lstm(x, hidden)
        # output: [batch, seq_len, hidden]
        # h_n: [num_layers, batch, hidden]

        # Use final hidden state from last layer
        final_hidden = h_n[-1]  # [batch, hidden]

        # Classify
        logits = self.classifier(final_hidden)  # [batch, num_classes]

        return logits, (h_n, c_n)

    def get_cell_states(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cell states for analysis.

        Args:
            x: Input sequence [batch, seq_len, input_size]

        Returns:
            cell_states: Final cell state [batch, hidden]
        """
        _, (_, c_n) = self.lstm(x)
        return c_n[-1]  # [batch, hidden]

"""Basic MLP model."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MLP(nn.Module):
    """MLP base class."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_features: int,
    ) -> None:
        """Initialize MLP with size parameters."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size * num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.size(0)
        x_reshaped = torch.reshape(
            x,
            (batch_size, -1),
        )  # Manipulations to deal with time series format
        output = self.flatten(x_reshaped)
        output = F.sigmoid(self.fc1(output))
        output = self.fc2(output)
        output = F.sigmoid(output)
        output = self.fc3(output)
        return torch.reshape(
            output,
            (batch_size, -1, 1),
        )  # Manipulations to deal with time series format

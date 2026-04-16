from typing import Iterable, List
import torch
import torch.nn as nn


class ANNClassifier(nn.Module):
    """Fully-connected ANN classifier with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")

        layer_dims: List[int] = [input_dim, *hidden_dims]

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        layers.append(nn.Linear(layer_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

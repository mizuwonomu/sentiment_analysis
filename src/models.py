from typing import Iterable, List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


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


class LSTMClassifier(nn.Module):
    """Frozen-embedding LSTM classifier for padded token id sequences."""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        bidirectional: bool = False,
        freeze_embeddings: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(),
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction_multiplier, output_dim)
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(
            embedded,
            lengths.detach().cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed_embedded)

        if self.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final_hidden = hidden[-1]

        return self.classifier(self.dropout(final_hidden))

"""Простой Baseline-класс с одним линейным слоем (логистическая регрессия)."""

import torch
from torch import nn


class LogisticRegression(nn.Module):
    """
    Логистическая регрессия как линейный классификатор.

    Parameters
    ----------
    in_features : int
        Размер входных признаков.
    num_classes : int
        Количество выходных классов.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.model = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через линейный слой, возвращает логиты.

        Parameters
        ----------
        embeddings : torch.Tensor
            Входные embedding'и.

        Returns
        -------
        torch.Tensor
            Логиты размера (N, num_classes).
        """
        return self.model(embeddings)

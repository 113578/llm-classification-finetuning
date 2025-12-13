"""Реализация глубокого MLP с residual-блоками для классификации.

Содержит ResidualBlock и DeepMLP.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """
    Одиночный residual-блок: два Linear слоя, GELU, LayerNorm и Dropout.

    Parameters
    ----------
    dimension : int
        Размерность входного и скрытого пространства.
    dropout_probability : float
        Вероятность включения dropout.
    """

    def __init__(self, dimension: int, dropout_probability: float):
        super().__init__()

        self.fc1 = nn.Linear(in_features=dimension, out_features=dimension)
        self.fc2 = nn.Linear(in_features=dimension, out_features=dimension)
        self.layer_norm = nn.LayerNorm(normalized_shape=dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через residual-блок и добавление residual connection.

        Parameters
        ----------
        embeddings : torch.Tensor
            Входной тензор shape (N, dimension).

        Returns
        -------
        torch.Tensor
            Выходной тензор той же размерности.
        """
        residual = embeddings

        embeddings = self.fc1(embeddings)
        embeddings = F.gelu(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.fc2(embeddings)

        embeddings = self.layer_norm(embeddings + residual)

        return embeddings


class DeepMLP(nn.Module):
    """
    Глубокая MLP-модель для классификации.

    Parameters
    ----------
    in_features : int
        Размерность входных признаков.
    num_classes : int
        Количество выходных классов.
    hidden_dim : int
        Размер скрытого слоя.
    depth : int
        Число residual-блоков.
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int, depth: int):
        super().__init__()

        self.input_projection = nn.Linear(in_features=in_features, out_features=hidden_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.blocks = nn.Sequential(
            *[ResidualBlock(dimension=hidden_dim, dropout_probability=0.1) for _ in range(depth)]
        )

        self.output_layer = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход и возвращает логиты (без softmax).

        Parameters
        ----------
        embeddings : torch.Tensor
            Входной тензор shape (N, in_features).

        Returns
        -------
        torch.Tensor
            Логиты размера (N, num_classes).
        """
        embeddings = self.input_projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.blocks(embeddings)

        return self.output_layer(embeddings)

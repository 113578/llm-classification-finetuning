"""Модуль с реализацией Dataset для тензорных embeddings и опциональных меток."""

import torch
from torch.utils.data import Dataset


class LCFDataset(Dataset):
    """
    Простой Dataset, оборачивающий embedding'и и опциональные метки.

    Parameters
    ----------
    embeddings : torch.Tensor
        Матрица embedding'ов (N x D).
    labels : torch.Tensor | None
        Метки в виде тензора длины N, или None (только embeddings).
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """
        Возвращает количество примеров (N).

        Returns
        -------
        int
            Число примеров.
        """
        return self.embeddings.size(dim=0)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает embeddings или (embeddings, label) для заданного индекса.

        Parameters
        ----------
        index : int
            Индекс элемента.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            embeddings либо кортеж (embeddings, label).
        """
        embeddings = self.embeddings[index]

        if self.labels is None:
            return embeddings

        return embeddings, self.labels[index]

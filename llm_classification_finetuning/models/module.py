"""Lightning-модуль, оборачивающий PyTorch-модель для задачи классификации.

Реализует шаги обучения, валидации, предсказания и конфигурацию оптимизатора.
"""

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, F1Score, Precision, Recall


class LCFModelModule(L.LightningModule):
    """
    LightningModule для тренировки и предсказания классификаторов.

    Parameters
    ----------
    model : nn.Module
        PyTorch модель для классификации.
    lr : float
        Скорость обучения для оптимизатора.
    """

    def __init__(self, model: nn.Module, lr: float):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.train_f1 = F1Score(task='multiclass', num_classes=3)
        self.train_precision = Precision(task='multiclass', num_classes=3)
        self.train_recall = Recall(task='multiclass', num_classes=3)

        self.val_accuracy = Accuracy(task='multiclass', num_classes=3)
        self.val_f1 = F1Score(task='multiclass', num_classes=3)
        self.val_precision = Precision(task='multiclass', num_classes=3)
        self.val_recall = Recall(task='multiclass', num_classes=3)

        self.lr = lr

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через внутреннюю модель.

        Parameters
        ----------
        embeddings : torch.Tensor
            Входные embedding'и.

        Returns
        -------
        torch.Tensor
            Логиты, полученные из модели.
        """
        return self.model(embeddings)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Один шаг обучения: вычисляет loss и логирует метрики.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Кортеж (embeddings, labels).

        Returns
        -------
        torch.Tensor
            Loss для backprop.
        """
        embeddings, labels = batch

        logits = self.model(embeddings)
        loss = self.criterion(logits, labels)

        probabilities = torch.softmax(input=logits, dim=1)
        predictions = torch.argmax(input=probabilities, dim=1)

        self.log(name='train_loss', value=loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(
            name='train_accuracy',
            value=self.train_accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='train_f1',
            value=self.train_f1(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='train_precision',
            value=self.train_precision(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='train_f1',
            value=self.train_recall(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Один шаг валидации: вычисляет loss и логирует метрики.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor]
            Кортеж (embeddings, labels).

        Returns
        -------
        None
        """
        embeddings, labels = batch

        logits = self.model(embeddings)
        loss = self.criterion(logits, labels)

        probabilities = torch.softmax(input=logits, dim=1)
        predictions = torch.argmax(input=probabilities, dim=1)

        self.log(name='val_loss', value=loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(
            name='val_accuracy',
            value=self.val_accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='val_f1',
            value=self.val_f1(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='val_precision',
            value=self.val_precision(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name='train_f1',
            value=self.val_recall(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Шаг предсказания, возвращает предсказанные индексы классов.

        Parameters
        ----------
        batch : torch.Tensor
            Батч embeddings для предсказания.

        Returns
        -------
        torch.Tensor
            Предсказанные индексы классов.
        """
        logits = self.model(batch)
        predictions = torch.argmax(input=logits, dim=1)

        return predictions

    def configure_optimizers(self) -> AdamW:
        """
        Создаёт оптимизатор (AdamW) для тренировки.

        Returns
        -------
        AdamW
            Оптимизатор для параметров модели.
        """
        return AdamW(params=self.model.parameters(), lr=self.lr)

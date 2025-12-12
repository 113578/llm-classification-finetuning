import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.optim import AdamW


class LCFModelModule(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=4)

        self.lr = lr

    def forward(self, embeddings):
        return self.model(embeddings)

    def training_step(self, batch):
        embeddings, labels = batch

        logits = self.model(embeddings)
        loss = self.criterion(logits, labels)

        probabilities = torch.softmax(input=logits, dim=1)
        predictions = torch.argmax(input=probabilities, dim=1)

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            name="train_accuracy",
            value=self.train_accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch):
        embeddings, labels = batch

        logits = self.model(embeddings)
        loss = self.criterion(logits, labels)

        probabilities = torch.softmax(input=logits, dim=1)
        predictions = torch.argmax(input=probabilities, dim=1)

        self.log(name="val_loss", value=loss, prog_bar=True, logger=True, on_epoch=True)
        self.log(
            name="val_accuracy",
            value=self.val_accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

    def test_step(self, batch):
        embeddings, labels = batch

        logits = self.model(embeddings)
        probabilities = torch.softmax(input=logits, dim=1)
        predictions = torch.argmax(input=probabilities, dim=1)

        self.log(
            name="test_accuracy",
            value=self.test_accuracy(predictions, labels),
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return AdamW(params=self.model.parameters(), lr=self.lr)

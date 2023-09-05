import torch.optim
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy

from lightning_lstm import LightningLSTM
from torch import nn
from torch.optim import Adam


class LSTMPredictor(LightningModule):
    def __init__(self, n_features: int, n_classes: int, learning_rate: int = 0.0001):
        super().__init__()
        self.model = LightningLSTM(n_features, n_classes)
        self.lr = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = BinaryAccuracy()

    def forward(self, x, labels=None):
        out = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(out, labels)
        return loss, out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        step_acc = self.accuracy(outputs, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc}

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        step_acc = self.accuracy(outputs, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc}

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        step_acc = self.accuracy(outputs, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_acc, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc}

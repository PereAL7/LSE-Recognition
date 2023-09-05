import torch
import torch.nn as nn
import lightning.pytorch as pl


class LSEPredictor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSEPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, 128, batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.relu(x[:, -1, :].unsqueeze(1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x.squeeze(dim=1))
        return x

    def training_step(self, batch, batch_idx):
        x = batch['sequence']
        y = batch['label']

        y_pred = self.forward(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = self.calculate_accuracy(y_pred, y)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sequence']
        y = batch['label']

        y_pred = self.forward(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = self.calculate_accuracy(y_pred, y)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['sequence']
        y = batch['label']

        y_pred = self.forward(x)
        loss = nn.functional.cross_entropy(y_pred, y)
        accuracy = self.calculate_accuracy(y_pred, y)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)

    @staticmethod
    def calculate_accuracy(outputs, labels):
        _, outputs = torch.max(outputs, dim=1)
        _, labels = torch.max(labels, dim=1)
        correct = (outputs == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total * 100
        return accuracy


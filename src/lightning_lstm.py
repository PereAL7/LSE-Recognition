from lightning.pytorch import LightningModule
from torch import nn
from torch import FloatTensor
class LightningLSTM(LightningModule):
    def __init__(self, n_features: int, n_classes: int, n_hidden: int = 128,
                 n_layers: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.7
        )
        self.linear1 = nn.Linear(n_hidden, 128)
        self.linear2 = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: FloatTensor):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return self.softmax(out)

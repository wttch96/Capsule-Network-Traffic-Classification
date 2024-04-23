from typing import Optional, Literal

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torchmetrics import F1Score, Accuracy, Recall

from metrics import Metrics


class CNN1d(pl.LightningModule):
    def __init__(self, out_features: int):
        super(CNN1d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=25, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=25, stride=1, padding='same'), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            # batch, 64, 2222 --> batch, 64, 2222
            nn.Flatten(),
            nn.Linear(in_features=64 * 2222, out_features=1000)
        )

        self.fc = nn.Linear(in_features=1000, out_features=out_features)

        self.f1 = F1Score(num_classes=out_features, task='multiclass')
        self.acc = Accuracy(num_classes=out_features, task='multiclass')
        self.rec = Recall(num_classes=out_features, task='multiclass')

        self.loss_fn = nn.CrossEntropyLoss()

        self.metrics = Metrics(num_classes=out_features)

    def forward(self, x):
        # batch, 1, 20, 1000 --> batch, 1, 20000
        x = x.reshape(x.shape[0], 1, -1)
        x = self.conv1(x)
        x = self.fc(x)  # type: Tensor
        return x

    def training_step(self, batch: Tensor, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)

        self.log('train_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        self.metrics.train_step_metrics(self, x, y)
        return loss

    def on_train_epoch_end(self):
        self.metrics.train_epoch_metrics(self)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

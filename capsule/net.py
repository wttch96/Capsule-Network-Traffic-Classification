from os import abort

import pytorch_lightning as pl
import torch
from torch import nn

from capsule.layers import Conv1dLayer, ConvCapsuleLayer, DenseCapsuleLayer


class CapsuleNet(pl.LightningModule):
    def __init__(self, out_features):
        super(CapsuleNet, self).__init__()

        # [batch, 1, N, _] --> [batch, 64, N, _]
        self.conv1d = Conv1dLayer(1, 64, kernel_size=3, stride=1)

        # [batch, 64, N, _] --> [batch, 128, N, _] --> [batch, 16, 8]
        self.conv_capsule = ConvCapsuleLayer(64, 128, num_caps=16, dim_caps=8, kernel_size=2, stride=1)

        # [batch, 16, 8] --> [batch, 2, 8]
        self.dense_capsule = DenseCapsuleLayer(16, 8, out_features, 8)

        # [batch, 16, 8] --> [batch, 32]
        self.lstm = nn.LSTM(out_features * 8, 32, batch_first=True)

        # [batch, 32] --> [batch, out_features]
        self.output = nn.Sequential(
            nn.Linear(32, out_features)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = 0.001

    def forward(self, x: torch.Tensor):
        # batch, 20, 1100 -> batch, 1, 20, 1100
        # x = x.reshape((x.shape[0], -1) + x.shape[1:])
        # x = torch.squeeze(x, dim=1)
        # ImageFolder 读取到的灰度 Tensor 刚好 shape: [batch, 1, 20, 1100]
        # [batch, 1, 20, _] -> [batch, 64, 20, _]
        x = self.conv1d(x)
        # [batch, 64, 20, _] -> [batch, 16, 8]
        x = self.conv_capsule(x)
        # -> [batch, 16, 8]
        x = self.dense_capsule(x)
        # -> [batch, 32]
        x, _ = self.lstm(x.reshape(x.shape[0], -1))
        # -> [batch, out_features]
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('train_acc', acc, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('test_acc', acc, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

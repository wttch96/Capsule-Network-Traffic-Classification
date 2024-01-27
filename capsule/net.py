import torch

from capsule.layers import Conv1dLayer, ConvCapsuleLayer, DenseCapsuleLayer
from torch import nn


class CapsuleNet(nn.Module):
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
            nn.Linear(32, out_features),
            nn.Softmax(dim=-1)
        )

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

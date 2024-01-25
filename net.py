import torch
from wttch.train.torch.utils import get_device_local
from wttch.train.utils import cache_wrapper

from capsule.layers import Conv2dCapsule, DenseCapsuleLayer
from util.data.USTC import USTCDataset

from torch import nn


class CapsuleNet(nn.Module):
    def __init__(self, out_features):
        super(CapsuleNet, self).__init__()

        self.conv2d = nn.Conv2d(1, 64, kernel_size=3, stride=1)

        # shape: [batch, 64, _, _] --> [batch, 128, _, _] --> [batch, _, 8] --> [batch, 16, 8]
        self.conv2d_capsule = Conv2dCapsule(64, 128, caps_num=16, dim_caps=8, kernel_size=2, stride=1)

        self.full_connect_capsule = DenseCapsuleLayer(16, 8, 2, 8)
        self.lstm = nn.LSTM(2 * 8, 32, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(32, out_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        # batch_size, 20, 1100 -> batch_size, 1, 20, 1100
        x = x.reshape((x.shape[0], -1) + x.shape[1:])
        # batch_size, 1, 20, 1100 -> batch_size, 64, 18, 1098
        x = self.conv2d(x)
        # batch_size, 64, 18, 1098 -> batch_size, 16,  298384, 8
        x = self.conv2d_capsule(x)
        # -> [batch_size, out_caps_num, out_caps_dim]
        x = self.full_connect_capsule(x)
        # -> [batch_size, hidden_size]
        x, _ = self.lstm(x.reshape(x.shape[0], -1))
        #
        x = self.output(x)
        return x

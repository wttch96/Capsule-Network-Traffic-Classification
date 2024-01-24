from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from wttch.train.torch.utils import get_device_local

from capsule.function import squash


class DenseCapsuleLayer(nn.Module):

    def __init__(self, in_caps_dim, out_caps_num, out_caps_dim, routing=3):
        super(DenseCapsuleLayer, self).__init__()

        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.routing = routing
        self.W = None
        self.in_caps_num = 0

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, in_caps_num, in_caps_dim]
        if self.W is None:
            self.in_caps_num = x.shape[1]
            self.W = nn.Parameter(
                torch.zeros(self.out_caps_num, x.shape[1], self.out_caps_dim, self.in_caps_dim,
                            device=get_device_local()))
            xavier_uniform_(self.W)
        # -> [batch_size, 1, in_caps_num, in_caps_dim, 1]
        x = x[:, None, :, :, None]
        # W shape: [out_caps_num, in_caps_num, out_caps_dim, in_caps_dim]
        # u_hat shape: [batch_size, out_caps_num, in_caps_num, out_caps_dim, 1]
        # -> [batch_size, out_caps_num, in_caps_num, out_caps_dim]
        u_hat = torch.squeeze(self.W @ x, dim=-1)

        # 动态路由最后一次才使用 u_hat 进行梯度
        u_hat_detached = u_hat.detach()

        b = torch.zeros((x.size(0), self.out_caps_num, self.in_caps_num), device=get_device_local())

        outputs = None

        for i in range(self.routing):
            # c = softmax(b) 在 in_caps_num 上找 softmax(out_caps)
            # c shape: [batch_size, out_caps_num, in_caps_num]
            c = F.softmax(b, dim=1)

            # c     shape: [batch_size, out_caps_num, in_caps_num]
            # c_t   shape: [batch_size, out_caps_num, 1, in_caps_num]
            c_t = c[:, :, None, :]
            if i == self.routing - 1:
                # 此时才传递梯度
                # c_t       shape: [batch_size, out_caps_num, 1,           in_caps_num]
                # u_hat     shape: [batch_size, out_caps_num, in_caps_num, out_caps_dim]
                # outputs   shape: [1, in_caps_num] @ [in_caps_num, out_caps_dim]
                # outputs   shape: [batch_size, out_caps_num, 1,           out_caps_dim]
                outputs = squash(c_t @ u_hat)
            else:
                # 不传递梯度, 使用 u_hat_detached
                outputs = squash(c_t @ u_hat_detached)
                # outputs * u_hat_detached shape: [batch_size, out_caps_num, in_caps_num, out_caps_dim]
                # a                        shape: [batch_size, out_caps_num, in_caps_dim]
                a = torch.sum(outputs * u_hat_detached, dim=-1)
                b = b + a

        # outputs   shape: [batch_size, out_caps_num, 1,           out_caps_dim]
        return torch.squeeze(outputs, dim=-2)


class Conv2dCapsuleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps, caps_num, kernel_size, stride=1, padding=0):
        super(Conv2dCapsuleLayer, self).__init__()

        self.caps_num = caps_num

        def create_capsule(idx):
            capsule = Conv2dCapsule(in_channels, out_channels, dim_caps=dim_caps, kernel_size=kernel_size,
                                    stride=stride, padding=padding)

            self.add_module("capsule_{}".format(idx), capsule)
            return capsule

        self.modules = [create_capsule(i) for i in range(caps_num)]

    def forward(self, x):
        x = [self.modules[i](x) for i in range(self.caps_num)]  # type: list[torch.Tensor]
        output = torch.stack(x, dim=1)
        return output


class Conv2dCapsule(nn.Module):
    """
    单个卷积胶囊单元。
    卷积一次后，将数据转换成 [batch_size, -1, dim_caps] 大小。
    然后将数据进行一次 squash 激活函数。
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):
        super(Conv2dCapsule, self).__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # batch_size * in_channels *
        # H *
        # W
        # ->
        # batch_size * out_channels *
        # ((H + 2 * padding - (kernel_size - 1) - 1) / stride + 1) *
        # ((W + 2 * padding - (kernel_size - 1) - 1) / stride + 1)
        x = self.conv2d(x)  # type: torch.Tensor
        # batch_size * out_channels * 17 * 1098
        # ->
        # batch_size * -1 * 8
        x = x.view(x.size(0), -1, self.dim_caps)

        return squash(x)

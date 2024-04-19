import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, uniform_

from capsule.function import squash
import pytorch_lightning as pl

class Conv1dLayer(nn.Module):
    """
    One-Dimensional CNN Layer 一维卷积层。
    只一维卷积，提取数据包的特征，随后根据论文说明调用一次 Relu 激活函数。

    输入数据维度 [batch, in_channels, N, M]
    输出数据维度 [batch, out_channels, N, _]
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """
        构造函数。
        Args:
            in_channels: 输入通道数，由于数据已经预处理成图片了，读取的灰度图像，这里就是1，而且不用再处理数据的维度。
            out_channels: 输出通道数
            kernel_size: 卷积核大小，在包方向上，流方向上自动设置为1。
            stride: 步幅，在包方向上，流方向上自动设置为1。
        """
        super(Conv1dLayer, self).__init__()
        # 因为输入数据是 [batch, 1, N, H] 格式的，直接使用一维卷积会报错，
        # 所以使用二维卷积来代替，但是 kernel_size, stride 第一个值设为1。
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x


class ConvCapsuleLayer(pl.LightningModule):
    """
    Convolutional Capsule Network Layer, 单个卷积胶囊单元。

    原文和意思这里只进行卷积（说明了卷积的参数），但是不能将其转换为 [batch, num_caps, dim_caps]。
    所以在卷积后添加了一步权重和偏置运算，将其转换为 [batch, num_caps, dim_caps] 大小。

    最后将数据进行一次 squash 激活函数。

    输入维度: [batch, in_channels, N, _]
    输出维度: [batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, num_caps, dim_caps, kernel_size=2, stride=1):
        """
        构造函数。
        Args:
            in_channels: 卷积输入通道数
            out_channels: 卷积输出通道数
            num_caps: 输出胶囊数量
            dim_caps: 输出胶囊向量的维度
            kernel_size: 卷积核大小，在包方向上，流方向上自动设置为1。
            stride: 步幅，在包方向上，流方向上自动设置为1。
        """
        super(ConvCapsuleLayer, self).__init__()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))

        self.W = None
        self.b = None

    def _try_init_params(self, x: torch.Tensor):
        """
        根据第一次输入数据的维度生成参数。
        """
        if self.W is None:
            # x      shape: [batch, _, dim_caps]
            # output shape: [batch, num_caps, dim_caps]
            # 所以
            # W      shape: [num_caps, _]
            # 这样的话
            # W @ x  shape: [num_caps, _] @ [batch, _, dim_caps] = [batch, num_caps, dim_caps]
            self.W = nn.Parameter(torch.zeros((self.num_caps, x.shape[1]), device=self.device))
            self.b = nn.Parameter(torch.zeros((self.num_caps, self.dim_caps), device=self.device))
            xavier_uniform_(self.W)
            uniform_(self.b)

    def forward(self, x):
        # x shape: [batch, in_channels,  N, _]
        # ------>: [batch, out_channels, N, _]
        x = self.conv2d(x)  # type: torch.Tensor
        # ------>: [batch, _, dim_caps]
        x = x.view(x.size(0), -1, self.dim_caps)

        self._try_init_params(x)

        # W @ x shape: [batch, caps_num, dim_caps]
        x = squash(self.W @ x + self.b)

        return squash(x)


class DenseCapsuleLayer(pl.LightningModule):
    """
    Fully-Connected Capsule Network Layer, 全连接胶囊网络层。
    主要是动态路由的实现。
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routing=3):
        """
        构造函数。
        Args:
            in_num_caps: 输入胶囊数量
            in_dim_caps: 输入胶囊向量维度
            out_num_caps: 输出胶囊数量
            out_dim_caps: 输出胶囊向量维度
            routing: 动态路由迭代次数
        """
        super(DenseCapsuleLayer, self).__init__()

        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routing = routing

        # 计算 u_hat 的权重 W
        # 已知:
        #   x                                   [batch, in_num_caps, in_dim_caps]   输入
        #   b       = 0                         [batch, in_num_caps, out_num_caps]  底层胶囊和高层胶囊的相关度
        #   b       = b + a
        #   u_hat   = u_hat_W @ x + u_hat_b                                         胶囊输入
        #   c       = softmax(b, )              [batch, in_num_caps, out_num_caps]  底层胶囊和高层胶囊的相关度
        #   s       = c @ u_hat                 [batch, out_num_caps, out_dim_caps] 相似度权重加权求和
        #   v       = squash(s)                 [batch, out_num_caps, out_dim_caps]
        #   a       = v * u_hat                 [batch, in_num_caps, out_num_caps]
        # 由于 s = c @ u_hat:
        #   c: [batch, in_num_caps, out_num_caps]
        #   s: [batch, out_num_caps, out_dim_caps]
        # 无法通过 s = c @ u_hat 矩阵乘法生成 s
        # 考虑变换
        #   c:      [batch, out_num_caps, 1, in_num_caps]
        #   s:      [batch, out_num_caps, 1, out_dim_caps]
        # 则:
        #   u_hat:  [in_num_caps, out_dim_caps]
        #           [1, in_num_caps] @ [in_num_caps, out_dim_caps] = [1, out_dim_caps]
        # 又
        #   x:      [batch, in_num_caps, in_dim_caps]
        # 无法通过 u_hat = u_hat_W @ x 矩阵乘法生成 u_hat
        # 考虑变换
        #   x:      [batch, in_num_caps, in_dim_caps,  1]
        #   u_hat:  [       in_num_caps, out_dim_caps, 1]
        # 则:
        # 其实论文写的挺清楚：向量之间的关系
        #   u_hat_W:[out_dim_caps, in_dim_caps]
        #   u_hat_b:[out_dim_caps]
        self.u_hat_W = nn.Parameter(torch.zeros(self.out_dim_caps, self.in_dim_caps, device=self.device))
        self.u_hat_b = nn.Parameter(torch.zeros(self.out_dim_caps, device=self.device))

        xavier_uniform_(self.u_hat_W)
        uniform_(self.u_hat_b)

    def forward(self, x: torch.Tensor):
        # x         shape: [batch, in_num_caps, in_dim_caps]
        # x           -->: [batch, in_num_caps, in_dim_caps, 1]
        x = x[:, :, :, None]
        # u_hat_X   shape: [out_dim_caps, in_dim_caps]
        # u_hat     shape: [out_dim_caps, in_dim_caps] @ [batch, in_num_caps, in_dim_caps, 1]
        #             -->: [batch, in_num_caps, out_dim_caps, 1]
        u_hat = self.u_hat_W @ x
        #             -->: [batch, in_num_caps, out_dim_caps]
        u_hat = torch.squeeze(u_hat, dim=-1)
        u_hat = u_hat + self.u_hat_b
        #             -->: [batch, 1, in_num_caps, out_dim_caps]
        u_hat = u_hat[:, None, :, :]

        # 动态路由最后一次才使用 u_hat 进行梯度
        u_hat_detached = u_hat.detach()

        # u_hat 和 输出v 的耦合系数
        # shape: [batch, in_num_caps, out_num_caps]
        # b = torch.zeros((x.size(0), self.in_num_caps, self.out_num_caps),
        #                 dtype=get_dtype_local(),
        #                 device=get_device_local())
        # 为了方便运算, 使用 [batch, out_num_caps, in_num_caps]
        b = torch.zeros((x.size(0), self.out_num_caps, self.in_num_caps), device=self.device)

        v = None

        for i in range(self.routing):
            # c = softmax(b) 在 in_num_caps 上找 softmax(out_caps)
            # c shape: [batch_size, in_num_caps, out_num_caps]
            # c = F.softmax(b, dim=2)
            # 为了方便计算
            # c shape: [batch_size, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # c     shape: [batch, in_num_caps, out_num_caps]
            #         -->: [batch, out_num_caps, 1, in_num_caps]
            # c = torch.permute(c, [0, 2, 1])[:, :, None, :]
            # 方便计算，不再转换轴
            # c     shape: [batch, out_num_caps, 1, in_num_caps]
            c = c[:, :, None, :]
            if i == self.routing - 1:
                # 此时才传递梯度
                # c         shape: [batch, out_num_caps, 1,           in_num_caps]
                # u_hat     shape: [batch, 1,            in_num_caps, out_dim_caps]
                # s         shape: [batch, out_num_caps, 1, out_dim_caps]
                s = c @ u_hat
                # v         shape: [batch, out_num_caps, 1, out_dim_caps]
                v = squash(s)
            else:
                # 不传递梯度, 使用 u_hat_detached
                s = c @ u_hat_detached
                v = squash(s)
                # v * u_hat_detached shape: [batch, out_num_caps, in_num_caps, out_dim_caps]
                # a                  shape: [batch, out_num_caps, in_num_caps]
                # a = torch.permute(torch.sum(v * u_hat_detached, dim=-1), [0, 2, 1])
                # 方便计算
                a = torch.sum(v * u_hat_detached, dim=-1)
                b = b + a

        # outputs   shape: [batch, out_num_caps, 1, out_dim_caps]
        #             -->: [batch, out_num_caps, out_dim_caps]
        return torch.squeeze(v, dim=-2)

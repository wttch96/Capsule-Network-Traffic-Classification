import pytorch_lightning as pl
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Recall, Precision

from context.dataset import DatasetContext


class Metrics(pl.LightningModule):
    """
    本文的测量方式: 分类的精确度; 单个分类的精确度、召回率、F1.

    继承 pl.LightningModule 是为了保证测量和模型在同一个 devcie
    """

    def __init__(self, num_classes: int):
        """
        构造函数
        Args:
            num_classes: 多分类的个数
        """
        super().__init__()
        # 总的分类精度
        self.acc = Accuracy(num_classes=num_classes, task="multiclass")
        self.f1 = F1Score(num_classes=num_classes, task="multiclass")
        self.rec = Recall(num_classes=num_classes, task="multiclass")

        # 多分类的 f1, 精确度, 召回率
        self.multi_f1 = F1Score(num_classes=num_classes, task="multiclass", average="none")
        self.multi_pre = Precision(num_classes=num_classes, task="multiclass", average="none")
        self.multi_rec = Recall(num_classes=num_classes, task="multiclass", average="none")

        self.metrics = [
            self.acc, self.rec, self.f1, self.multi_pre, self.multi_rec, self.multi_f1
        ]

    def train_step_metrics(self, module: pl.LightningModule, y_pred: Tensor, y: Tensor):
        """
        训练阶段测量数据记录。
        Args:
            module: 模型, 主要是为了调用模型的 log 函数记录测量数据
            y_pred: 模型预测值
            y: 样本目标值
        """
        for metric in self.metrics:
            metric(y_pred, y)

        module.log('train_acc', self.acc, on_step=True, prog_bar=True, sync_dist=True)
        module.log('train_rec', self.rec, on_step=True, prog_bar=True, sync_dist=True)
        module.log('train_f1', self.f1, on_step=True, prog_bar=True, sync_dist=True)

    def train_epoch_metrics(self, module: pl.LightningModule):
        """
        记录每个训练 epoch 结束的测量数据。
        Args:
            module: 模型, 主要是为了调用模型的 log 函数记录测量数据
        """
        self._epoch_metrics(module, 'train')

    def validation_step_metrics(self, y_pred: Tensor, y: Tensor):
        for metric in self.metrics:
            metric(y_pred, y)

    def validation_epoch_metrics(self, module: pl.LightningModule):
        """
        记录每个验证 epoch 结束的测量数据。
        Args:
            module: 模型, 主要是为了调用模型的 log 函数记录测量数据
        """
        self._epoch_metrics(module, 'val')

    def test_step_metrics(self, y_pred: Tensor, y: Tensor):
        for metric in self.metrics:
            metric(y_pred, y)

    def test_epoch_metrics(self, module: pl.LightningModule):
        """
        记录每个测试 epoch 结束的测量数据。
        Args:
            module: 模型, 主要是为了调用模型的 log 函数记录测量数据
        """
        self._epoch_metrics(module, 'test')

    def _epoch_metrics(self, module: pl.LightningModule, prefix: str):
        module.log(f'{prefix}_epoch_acc', self.acc.compute(), sync_dist=True)
        module.log(f'{prefix}_epoch_rec', self.rec.compute(), sync_dist=True)
        module.log(f'{prefix}_epoch_f1', self.f1.compute(), sync_dist=True)

        # 分别记录各个分类
        for i, pre in enumerate(self.multi_pre.compute()):
            module.log(f'{prefix}_pre_{DatasetContext.classes[i]}', pre, sync_dist=True)
        for i, rec in enumerate(self.multi_rec.compute()):
            module.log(f'{prefix}_rec_{DatasetContext.classes[i]}', rec, sync_dist=True)
        for i, f1 in enumerate(self.multi_rec.compute()):
            module.log(f'{prefix}_f1_{DatasetContext.classes[i]}', f1, sync_dist=True)

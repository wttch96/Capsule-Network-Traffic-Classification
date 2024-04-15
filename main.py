import argparse

from pytorch_lightning.loggers import CSVLogger
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl

from config import Config
from dataset import DatasetContext
from baseline.cnn1d import CNN1d

import importlib

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--base-line', type=str, choices=['cnn1d-2', 'cnn1d-10', 'cnn1d-20'], help='选择基线任务')
    arg.add_argument('--show-datasets', action='store_true', help='显示所有的数据集')
    arg.add_argument('--show-trainers', action='store_true', help='显示所有的训练任务')
    arg.add_argument('--show-models', action='store_true', help='显示所有的模型')

    args = arg.parse_args()

    config = Config()

    module = importlib.import_module('baseline.cnn1d')
    cls = getattr(module, 'CNN1d')
    print(cls(out_features=2))

    # 显示所有的数据集
    if args.show_datasets:
        config.show_datasets()

    if args.show_trainers:
        config.show_trainers()

    if args.show_models:
        config.show_models()


def tmp():
    config = Config()

    devices = config['pl']['devices']

    dataset_context = DatasetContext(config)

    train = pl.Trainer(max_epochs=20, devices=devices, logger=CSVLogger("logs/train.csv", version=1))

    classes_size = len(dataset_context.dataset.classes)
    print(f"out_features: {classes_size}")
    model = CNN1d(classes_size)
    train.fit(model,
              train_dataloaders=dataset_context.train,
              val_dataloaders=dataset_context.val)

    train.test(model, dataset_context.test)

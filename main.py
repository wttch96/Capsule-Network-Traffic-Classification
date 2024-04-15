from pytorch_lightning.loggers import CSVLogger
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
from wth.utils.config import Config
import pytorch_lightning as pl
from dataset import DatasetContext
from baseline.cnn1d import CNN1d

if __name__ == '__main__':
    config = Config()

    dataset_context = DatasetContext(config)

    train = pl.Trainer(max_epochs=20, devices=[2, 3], logger=CSVLogger("logs/train.csv"))

    train.fit(CNN1d(),
              train_dataloaders=dataset_context.binary_classification.train,
              val_dataloaders=dataset_context.binary_classification.val)

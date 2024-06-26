from abc import abstractmethod

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataset import Dataset, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Lambda, ToTensor, Grayscale, ToImage, ToDtype
from .config import Config
import torch.nn.functional as F


class TrainAndTestDataset:

    def __init__(self, dataset: Dataset, val_rate: float = 0.1, test_rate: float = 0.1, batch_size: int = 64,
                 num_workers: int = 2):
        self.dateset = dataset
        image_folder_len = len(dataset)
        val_size = int(val_rate * image_folder_len)
        test_size = int(test_rate * image_folder_len)
        train_size = image_folder_len - val_size - test_size

        train, val, test = random_split(dataset, [train_size, val_size, test_size])  # type: Subset, Subset, Subset

        self.train = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(test, batch_size=batch_size, num_workers=num_workers)
        self.val = DataLoader(val, batch_size=batch_size, num_workers=num_workers)


def _scale(x):
    return x / 255.0


class DatasetContext:
    """
    数据集都放在这里
    """

    def __init__(self, config: Config):
        self.class_weights = None
        self.config = config

        self.transform = Compose([
            Grayscale(),
            ToImage(),
            ToDtype(torch.float32),
            Lambda(_scale),
        ])

    def get_dataset(self, dataset: str):
        dataset_config = self.config.get_dataset(dataset)
        root_path = dataset_config['root-path']
        batch_size = dataset_config['batch-size']
        num_workers = dataset_config['num-workers']

        dataset = ImageFolder(root_path, transform=self.transform)

        class_count = torch.tensor(dataset.targets).bincount()
        class_weights = 1.0 / class_count.float()

        self.class_weights = F.softmax(class_weights)

        train, val, test = random_split(dataset, [0.8, 0.1, 0.1])  # type: Subset, Subset, Subset

        train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)
        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader

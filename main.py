import time

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch import nn

from config import config
from net import CapsuleNet
from wttch.train.torch.utils import get_device_local, set_dtype_local, get_dtype_local
import torch
from util.data.USTC_loader import USTCDataset, USTCDataloader
import torch.nn.functional as F
from wttch.train.utils import cache_wrapper
from torch.nn.functional import one_hot
from wttch.train.notification import DingtalkNotification

if __name__ == '__main__':
    notification = DingtalkNotification(config.webhook, config.secret)
    dataset = USTCDataloader('USTC-TFC2016', N=20, M=1100)
    dataset.load_data()
    train_x, train_y, test_x, test_y = dataset.split_data()

    set_dtype_local(torch.float32)

    dataset = USTCDataset(train_x, train_y)
    test_dataset = USTCDataset(test_x, test_y)

    epochs = 36
    net = CapsuleNet(20).to(get_device_local())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=10e-3)

    for i in range(epochs):
        net.train()
        start = time.time()
        dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        dataset_loader_len = len(dataset_loader)
        for i, (x, y) in enumerate(dataset_loader):  # type: int, tuple[torch.Tensor, torch.Tensor]
            x = x.to(dtype=get_dtype_local(), device=get_device_local())
            y = y.to(dtype=torch.int64, device=get_device_local())
            y = F.one_hot(y, num_classes=20).to(dtype=get_dtype_local())
            optimizer.zero_grad()
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = 0
                total = 0
                for test_x, test_y in dataset_loader:  # type: torch.Tensor, torch.Tensor
                    test_x = test_x.to(dtype=get_dtype_local(), device=get_device_local())
                    test_y = test_y.to(dtype=torch.int64, device=get_device_local())
                    test_pred = torch.argmax(net(test_x), -1)
                    acc += sum(test_pred == test_y)
                    total += len(test_x)
                print(f'{i + 1}/{dataset_loader_len} loss: {loss.item()} acc: {acc / total * 100: .2f}%')
            # test

        with torch.no_grad():
            acc = 0
            total = 0
            for test_x, test_y in dataset_loader:  # type: torch.Tensor, torch.Tensor
                test_x = test_x.to(dtype=get_dtype_local(), device=get_device_local())
                test_y = test_y.to(dtype=torch.int64, device=get_device_local())
                test_pred = torch.argmax(net(test_x), -1)
                acc += sum(test_pred == test_y)
                total += len(test_x)
            print(f"Epoch: {i + 1}/{epochs} use: {time.time() - start: .02f} acc: {acc / total * 100: .2f}%")
            notification.send_markdown(f"# Epoch: {i + 1}/{epochs} \n\n"
                                       f"> use: {time.time() - start: .02f} \n\n"
                                       f"> acc: {acc / total * 100: .2f}%", title=f"Epoch: {i + 1}/{epochs}")

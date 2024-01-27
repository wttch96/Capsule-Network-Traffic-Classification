import time

from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Grayscale, ToTensor
from wttch.train.utils.progress import Progress

from config import config
from capsule.net import CapsuleNet
from wttch.train.torch.utils import get_device_local, set_dtype_local, get_dtype_local, set_device_local, try_gpu
import torch
from util.data.USTC_loader import USTCDataset, USTCDataloader
import torch.nn.functional as F
from wttch.train.notification import DingtalkNotification

if __name__ == '__main__':
    notification = DingtalkNotification(config.webhook, config.secret)

    transform = Compose([
        Grayscale(),
        ToTensor()
    ])

    set_device_local(try_gpu(3))

    image_folder = ImageFolder("USTC-TFC2016-IMAGE", transform=transform)

    image_folder_len = len(image_folder)
    train_len = int(image_folder_len * 0.8)
    test_len = image_folder_len - train_len

    train, test = random_split(image_folder, [train_len, test_len])  # type: Subset, Subset

    epochs = 36
    set_dtype_local(torch.float32)
    net = CapsuleNet(20).to(get_device_local(), dtype=get_dtype_local())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=10e-3)

    train_loader = DataLoader(train, shuffle=True, batch_size=64, num_workers=4)
    test_loader = DataLoader(test, shuffle=True, batch_size=64, num_workers=4)

    for i in range(epochs):
        print(f"Epoch: {i + 1}/{epochs}")
        net.train()
        start = time.time()
        dataset_loader_len = len(train_loader)
        progress = Progress(train_loader)
        for _, (x, y) in enumerate(progress):  # type: int, tuple[torch.Tensor, torch.Tensor]
            x = x.to(dtype=get_dtype_local(), device=get_device_local())
            y = y.to(dtype=torch.int64, device=get_device_local())
            y_hot = F.one_hot(y, num_classes=20).to(dtype=get_dtype_local())
            optimizer.zero_grad()
            y_pred = net(x)
            loss = loss_fn(y_pred, y_hot)
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(y_pred, dim=1)
            acc = torch.sum(y_pred == y)
            progress.train_result(loss.item(), acc.item() / y_pred.shape[0])

        print("Finished Training, Start Testing...")
        with torch.no_grad():
            acc = 0
            total = 0
            for test_x, test_y in test_loader:  # type: torch.Tensor, torch.Tensor
                test_x = test_x.to(dtype=get_dtype_local(), device=get_device_local())
                test_y = test_y.to(dtype=torch.int64, device=get_device_local())
                test_pred = torch.argmax(net(test_x), -1)
                acc += sum(test_pred == test_y)
                total += len(test_x)
            print(f"Epoch: {i + 1}/{epochs} use: {time.time() - start: .02f} acc: {acc / total * 100: .2f}%")
            notification.send_markdown(f"# Epoch: {i + 1}/{epochs} \n\n"
                                       f"> use: {time.time() - start: .02f} \n\n"
                                       f"> acc: {acc / total * 100: .2f}%", title=f"Epoch: {i + 1}/{epochs}")

    torch.save(net.state_dict(), 'capsule_module.pkl')
    notification.send_text("胶囊网络已保存!")

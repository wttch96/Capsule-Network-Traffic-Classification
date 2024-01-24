from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch import nn
from net import CapsuleNet
from wttch.train.torch.utils import get_device_local
import torch
from util.data.USTC import USTCDataset, USTCDataloader
import torch.nn.functional as F
from wttch.train.utils import cache_wrapper
from torch.nn.functional import one_hot

if __name__ == '__main__':
    dataset = USTCDataloader('USTC-TFC2016', N=20, M=1100)
    dataset.load_data()
    train_x, train_y, test_x, test_y = dataset.split_data()

    dataset = USTCDataset(train_x, train_y)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    epochs = 10
    net = CapsuleNet(20).to(get_device_local())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=10e-3)

    for i in range(epochs):
        net.train()
        for x, y in dataset_loader:  # type: torch.Tensor, torch.Tensor
            x = x.to(dtype=torch.float32, device=get_device_local())
            y = y.to(dtype=torch.int64, device=get_device_local())
            y = F.one_hot(y, num_classes=20).to(dtype=torch.float32)
            x = x.reshape((x.shape[0], -1) + x.shape[1:])
            optimizer.zero_grad()
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            print(loss)

            optimizer.step()
            break
            # test

#%%

import os
from math import floor

import numpy as np
from torch.utils.data import Dataset
from wttch.train.utils import StopWatch, cache_wrapper

from util.data.USTC import all_types
from util.data.USTC_preprocess import load_pcap_raw_data


class USTCDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@cache_wrapper(prefix="ustc_numpy", save_path="caches")
def load_ustc(file_path, M, N):
    loader = USTCDataloader(file_path, M, N)
    loader.load_data()

    return loader.data, loader.label


class USTCDataloader:

    def __init__(self, file_path, M, N):
        self.file_path = file_path
        self.M = M
        self.N = N

        self.stopwatch = StopWatch()

        self.data = None
        self.label = None

    def split_data(self, test_rate=0.2):
        data_len = len(self.data)
        test_len = int(data_len * test_rate)
        indices = np.random.permutation(data_len)
        test_indices = indices[:test_len]
        train_indices = indices[test_len:]

        return self.data[train_indices], self.label[train_indices], self.data[test_indices], self.label[test_indices]

    def load_data(self):
        """加载数据"""
        self.data = None
        self.label = None
        self.stopwatch.start("load preprocessed data")
        raw_data_dict = load_pcap_raw_data(self.file_path, max_len=self.M)
        self.stopwatch.stop()

        for label, data in raw_data_dict.items():
            self.stopwatch.start(f"Parse raw data for {label}: {all_types[label]}")

            # 能生成多少个 N * M 的包
            data_count = floor(len(data) / self.N)

            if data_count == 0:
                continue

            # N * M 个包一共多长
            data_len = data_count * self.N
            pck_data = np.zeros((data_len, self.M), dtype=np.uint8)

            for i in range(data_len):
                raw_data = data[i]
                # 包长度
                raw_data_len = len(raw_data)
                # 因为预处理过，所以 不会有大于的
                raw_data_array = np.frombuffer(raw_data, dtype=np.uint8)
                if raw_data_len < self.M:
                    pck_data[i, :raw_data_len] = raw_data_array
                else:
                    pck_data[i] = raw_data_array
            pck_data = pck_data.reshape((data_count, self.N, self.M))
            label_array = np.array([label] * data_count)
            if self.data is None:
                self.data = pck_data
                self.label = label_array
            else:
                self.data = np.append(self.data, pck_data, axis=0)
                self.label = np.append(self.label, label_array)

            print(f'解析 {all_types[label]}, 用时: {self.stopwatch.stop()}')
        print()
        self.stopwatch.display()

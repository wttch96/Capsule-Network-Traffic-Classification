from math import ceil

from scapy.utils import rdpcap
import numpy as np
from torch.utils.data import Dataset
from wttch.train.utils import cache_wrapper, StopWatch
import os

# from py7zr import SevenZipFile

benign_types = ['SMB', 'Weibo', 'BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype',
                'WorldOfWarcraft']
benign_type2idx = {bt: i for i, bt in enumerate(benign_types)}
malware_types = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
all_types = benign_types + malware_types
malware_type2idx = {bt: i + 10 for i, bt in enumerate(malware_types)}
type2idx = {bt: i for i, bt in enumerate(all_types)}


# def _try_unzip_file(path):
#     """
#     尝试解压指定目录下的 7z 文件
#     """
#     for filename in os.listdir(f'{path}'):
#         if filename.endswith('.7z'):
#             full_path = f'{path}/{filename.replace(".7z", "")}'
#             if os.path.exists(full_path) or os.path.exists(f'{full_path}.pcap'):
#                 # 已经解压
#                 print(f'{filename} 已经解压.')
#             else:
#                 # 解压
#                 print(f'{filename} 开始解压...')
#                 archive = SevenZipFile(f'{path}/{filename}', mode='r')
#                 archive.extractall(f'{path}')
#                 archive.close()
#                 print(f'{filename} 已解压完成!')


class USTCDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class USTCDataloader:
    data: np.ndarray
    label: np.ndarray

    def __init__(self, file_path, M, N):
        self.M = M
        self.N = N
        self._stopwatch = StopWatch()
        self._origin_file_path = file_path

        self._data = []
        self._label = []

    def split_data(self, test_rate=0.2):
        data_len = len(self.data)
        test_len = int(data_len * test_rate)
        indices = np.random.permutation(data_len)
        test_indices = indices[:test_len]
        train_indices = indices[test_len:]

        return self.data[train_indices], self.label[train_indices], self.data[test_indices], self.label[test_indices]

    def load_data(self):
        """加载数据"""
        self._data = []
        self._label = []
        # 尝试解压
        # self._stopwatch.start('Unzip .7z file')
        # _try_unzip_file(f'{self._origin_file_path}/Benign')
        # _try_unzip_file(f'{self._origin_file_path}/Malware')
        # self._stopwatch.stop()

        self._load_pcap_original_by_fold(f'{self._origin_file_path}/Benign')
        self._load_pcap_original_by_fold(f'{self._origin_file_path}/Malware')

        self.data = np.concatenate(self._data)
        self.label = np.array(self._label)
        self._data = []
        self._label = []
        self._stopwatch.display()

    def _load_pcap_original_by_fold(self, fold):
        """遍历加载pcap文件"""
        for filename in os.listdir(f'{fold}'):
            if filename.endswith('.pcap'):
                # 实际解析
                self._load_pcap_original(f'{fold}/{filename}')
            elif filename.endswith('.7z'):
                pass
            elif os.path.isdir(f'{fold}/{filename}'):
                # 递归调用
                self._load_pcap_original_by_fold(f'{fold}/{filename}')
            else:
                print(f'忽略文件: {filename}')

    def _load_pcap_original(self, filename):
        """实际读取pcap文件"""
        self._stopwatch.start(filename)
        pcap = rdpcap(filename)
        data_count = ceil(len(pcap) / self.N)
        data_len = data_count * self.N
        pck_data = np.zeros((data_len, self.M))

        label = None
        for t in all_types:
            if t.lower() in filename.lower():
                label = type2idx[t]
                break

        if label is None:
            raise ValueError(f'未知类型: {filename}')

        for i, data in enumerate(pcap):
            arr_len = len(data.original)
            if arr_len >= self.M:
                pck_data[i, :self.M] = np.array(bytearray(data.original[:self.M])) / 255
            else:
                pck_data[i, :arr_len] = np.array(bytearray(data.original)) / 255
        pck_data = pck_data.reshape((data_count, self.N, self.M))

        for _ in range(data_count):
            self._label.append(label)
        self._data.append(pck_data)
        self._stopwatch.stop()

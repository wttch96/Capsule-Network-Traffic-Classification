import os
from math import ceil

from scapy.utils import rdpcap
from wttch.train.utils import StopWatch

from util.data.USTC import all_types, type2idx


class USTCPreprocess:
    def __init__(self, file_path, max_len=1100):
        self.file_path = file_path
        self.stopwatch = StopWatch()
        self.max_len = max_len

        self.data = {}  # type: dict[int, list[bytes]]

    def load_data(self) -> dict[int, list[bytes]]:
        """加载数据"""
        self.data = {}

        # 初始化
        for v in type2idx.values():
            self.data[v] = []

        self._load_pcap_original_by_fold(f'{self.file_path}/Benign')
        self._load_pcap_original_by_fold(f'{self.file_path}/Malware')

        self.stopwatch.display()

        return self.data

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
        self.stopwatch.start(filename)
        pcap = rdpcap(filename)

        label = None
        for t in all_types:
            if t.lower() in filename.lower():
                label = type2idx[t]
                break

        if label is None:
            raise ValueError(f'未知类型: {filename}')

        for i, data in enumerate(pcap):
            arr_len = len(data.original)
            if arr_len >= self.max_len:
                self.data[label].append(data.original[:self.max_len])
            else:
                self.data[label].append(data.original)

        self.stopwatch.stop()

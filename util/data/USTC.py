import os
from math import ceil, floor

import numpy as np
from scapy.utils import rdpcap
from wttch.train.utils import StopWatch, cache_wrapper

from PIL import Image

benign_types = ['SMB', 'Weibo', 'BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype',
                'WorldOfWarcraft']
benign_type2idx = {bt: i for i, bt in enumerate(benign_types)}
malware_types = ['Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
all_types = benign_types + malware_types
malware_type2idx = {bt: i + 10 for i, bt in enumerate(malware_types)}
type2idx = {bt: i for i, bt in enumerate(all_types)}


class USTCPreprocess:
    def __init__(self, file_path, img_save_path, pack_width=20, pack_len=1100, img_w=200, img_h=110):
        self.file_path = file_path
        self.img_save_path = img_save_path
        self.stopwatch = StopWatch()
        self.pack_width = pack_width
        self.max_len = pack_len

        self.img_w = img_w
        self.img_h = img_h

        self.type_count = {}

    def preprocess(self):
        """加载数据"""
        # 计数
        self.type_count = {}
        for t in all_types:
            self.type_count[t] = 0

        self._load_pcap_original_by_fold(f'{self.file_path}/Benign')
        self._load_pcap_original_by_fold(f'{self.file_path}/Malware')

        self.stopwatch.display()

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
        print(f"Begin loading {filename}...")
        pcap = rdpcap(filename)

        label = None
        for t in all_types:
            if t.lower() in filename.lower():
                label = t
                break
        if label is None:
            raise ValueError(f'未知类型: {filename}')

        # 能生成多少个 N * M 的包
        data_count = floor(len(pcap) / self.pack_width)

        if data_count == 0:
            self.stopwatch.stop()
            return

        # N * M 个包一共多长
        data_len = data_count * self.pack_width
        pck_data = np.zeros((data_len, self.max_len), dtype=np.uint8)

        for i, raw_data in enumerate(pcap):
            if i >= data_count:
                break
            raw_data = raw_data.original
            # 包长度
            raw_data_len = len(raw_data)
            if raw_data_len < self.max_len:
                raw_data_array = np.frombuffer(raw_data, dtype=np.uint8)
                pck_data[i, :raw_data_len] = raw_data_array
            else:
                raw_data_array = np.frombuffer(raw_data[:self.max_len], dtype=np.uint8)
                pck_data[i] = raw_data_array
        pck_data = pck_data.reshape((data_count, self.pack_width, self.max_len))

        for i in range(pck_data.shape[0]):
            # 写入图片
            img_data = pck_data[i].reshape(self.img_w, self.img_h)
            image = Image.fromarray(img_data)
            filename = f'{label}/{self.type_count[label]}.png'
            label_path = f'{self.img_save_path}/{label}'

            if not os.path.exists(label_path):
                os.makedirs(label_path)

            image.save(f'{self.img_save_path}/{filename}')
            print(f'Saved {filename}.', end='\r')
            self.type_count[label] = self.type_count[label] + 1
        print()
        print(f"{filename}: pack count: {len(pcap)}, saved img count: {pck_data.shape[0]}")
        self.stopwatch.stop()

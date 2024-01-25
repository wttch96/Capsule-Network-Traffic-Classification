import os

from wttch.train.utils import cache_wrapper

from util.data.USTC import USTCDataloader
from util.data.USTC_preprocess import USTCPreprocess
from util.data.USTC import all_types


@cache_wrapper(prefix='pcap_raw_data', save_path='caches')
def load_pcap_raw_data(file_path):
    preprocess = USTCPreprocess(file_path, max_len=1100)

    data = preprocess.load_data()

    return data


if __name__ == '__main__':
    data = load_pcap_raw_data("USTC-TFC2016")

    for k, v in data.items():
        print(f'{k}: {all_types[k]} length: {len(v)}')

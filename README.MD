论文代码复现。

Capsule Network Assisted IoT Traffic Classification Mechanism for Smart Citie

[原文链接](https://ieeexplore.ieee.org/abstract/document/8651277)

# 数据集

[USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016)

## 数据预处理

本仓库不管数据预处理，数据预处理请参见[预处理器](https://github.com/wttch96/PcapPreprocess)

# 训练

模型、数据集、训练器定义在 config.yml 中。

## 显示所有的数据集
```bash
# 显示数据集
python main.py --show-datasets
```
示例输出:
```text
数据集:ustc-all:
    参数:{'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000', 'batch-size': 2, 'num-workers': 2}
数据集:ustc-benign:
    参数:{'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000/Benign', 'batch-size': 2, 'num-workers': 2}
数据集:ustc-malware:
    参数:{'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000/Malware', 'batch-size': 2, 'num-workers': 2}
```
## 显示所有的模型定义
```bash
python main.py --show-models
```

示例输出:
```text
cnn1d
    定义: {'module': 'baseline.cnn1d', 'cls': 'CNN1d'}
capsule
    定义: {'module': 'capsule.net', 'cls': 'CapsuleNet'}
```

## 显示所有的训练器
```bash
python main.py --show-trainers
```

示例输出:
```text
训练器:capsule-all
    参数: {'max_epochs': 20, 'devices': 'auto'}
    数据集: ustc-all
        参数: {'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000', 'batch-size': 2, 'num-workers': 2}
训练器:capsule-benign
    参数: {'max_epochs': 20, 'devices': 'auto'}
    数据集: ustc-benign
        参数: {'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000/Benign', 'batch-size': 2, 'num-workers': 2}
训练器:capsule-malware
    参数: {'max_epochs': 20, 'devices': 'auto'}
    数据集: ustc-malware
        参数: {'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000/Malware', 'batch-size': 2, 'num-workers': 2}
训练器:cnn1d-all
    参数: {'max_epochs': 20, 'devices': 'auto'}
    数据集: ustc-all
        参数: {'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000', 'batch-size': 2, 'num-workers': 2}
训练器:cnn1d-benign
    参数: {'max_epochs': 20, 'devices': 'auto'}
    数据集: ustc-benign
        参数: {'root-path': '/Users/wttch/workspace/PG/PcapPreprocess/PcapPreprocess/datasets/USTC-example-capsule-20-1000/Benign', 'batch-size': 2, 'num-workers': 2}
```

## 开始训练
```bash
# 使用 capsule-benign 开始训练
python main.py --start-trainers capsule-benign
```
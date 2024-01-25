论文代码复现。

Capsule Network Assisted IoT Traffic Classification Mechanism for Smart Citie

[原文链接](https://ieeexplore.ieee.org/abstract/document/8651277)

# 数据集

[USTC-TFC2016](https://github.com/yungshenglu/USTC-TFC2016)

# 数据预处理

> 1. 加载这样的二进制数据，全部读入内存显然不太合适（尝试过，速度太慢了）；
> 2. 所以想着怎么处理下，使用过读取、处理后缓存，不理想，还是很慢；
> 3. 原数据集有[处理工具](https://github.com/yungshenglu/USTC-TK2016), 可惜只是 windom 版本的；
> 4. 参考该工具，自己写个处理工具，思路就是每个包 N * M 的将其转换为一个照片，按分类分文件夹保存；
> 5. 使用 `torchvision.datasets.ImageFolder` 加载数据。
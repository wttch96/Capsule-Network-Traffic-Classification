from typing import Optional

from wth.utils.config import Config as _Config


class Config(_Config):
    """
    扩展 config.yml 的读取类。

    解析本项目使用的相关数据。
    """

    def __init__(self, file: str = 'config.yml'):
        super().__init__(config_file=file)

        # 所有的训练器
        self.trainers = self['trainers']  # type: dict[str, dict]
        self.datasets = self['datasets']  # type: dict[str, dict]
        self.models = self['models']  # type: dict[str, dict]

    def _get_dataset(self, dataset: str) -> Optional[dict]:
        return self.datasets[dataset]

    def show_models(self):
        for k, v in self.models.items():
            print(f'{k}')
            print(f'    定义: {v}')

    def show_datasets(self):
        """打印配置文件中所有定义的数据集。"""
        for k, v in self.datasets.items():
            print(f"数据集:{k}:")
            print(f"    参数:{v}")

    def show_trainers(self):
        """打印配置文件中所有定义的训练器。"""
        for k, v in self.trainers.items():
            print(f"训练器:{k}")
            print(f"    参数: {v['parameters']}")
            dataset = v['dataset']
            if dataset in self.datasets:
                print(f"    数据集: {dataset}")
                print(f"        参数: {self.datasets[dataset]}")
            else:
                print(f"    数据集: {dataset} (⚠️:未在 config.yml 中定义)")

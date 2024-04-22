from pytorch_lightning.loggers import CSVLogger

from .config import Config
import pytorch_lightning as pl

from .dataset import DatasetContext
from .model import ModelContext


class TrainerContext:
    """
    训练器都放在这里。
    """

    def __init__(self, config: Config):
        self.config = config
        self.dataset_ctx = DatasetContext(config)
        self.model_ctx = ModelContext(config)

    def start_train(self, trainer_name: str):
        """
        开始执行训练任务。
        :param trainer_name: 训练任务的名称
        """
        print(f"开始配置训练器 [{trainer_name}]...")
        trainer_config = self.config.get_trainer(trainer_name)
        dataset_name = trainer_config['dataset']
        # 生成训练、验证、测试的 DataLoader
        print(f"开始加载数据集 [{dataset_name}]...")
        train, val, test = self.dataset_ctx.get_dataset(dataset_name)
        print(
            f"已加载数据集 [{dataset_name}]. 训练集长度: {len(train)}, 验证集长度: {len(val)}, 测试集长度: {len(test)}")
        # 初始化模型
        model_config = trainer_config['model']
        model = self.model_ctx.get_model(model_config)

        # 训练器
        init_parameters = trainer_config['init-parameters']
        log = CSVLogger(trainer_config['csv-logger-path'])
        print(f"训练器参数: {init_parameters}")
        trainer = pl.Trainer(**init_parameters, logger=log)  # type: pl.Trainer
        print(f"训练器配置完成!")
        # 训练
        trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
        # 测试
        trainer.test(model, test)

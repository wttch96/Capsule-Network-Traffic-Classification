from .config import Config

import importlib


class ModelContext:
    """
    模型都放在这里。

    config.yml 中的定义格式：
    名字:
      module: 模型所在的 python 包
      cls: 模型的类名

    例如

    cnn1d:
      module: baseline.cnn1d
      cls: CNN1d

    模型的使用时格式（需要放在 trainer 的子级， 即生命训练器使用的模型）:
    model:
      name: 模型的名称, 和定义的名字一致
      init-parameters: 初始化参数, 必须和指定的 module, cls 的构造函数的 kwargs 格式一致。
    """

    def __init__(self, config: Config):
        self.config = config

        self.models = self.config['models']

    def get_model(self, model_params: dict):
        """
        根据模型声明的参数来构造模型。
        :param model_params: trainer 使用的字典, 包含模型的名称和初始化参数
        :return:
        """
        model_name = model_params['name']
        model_init_parameters = model_params['init-parameters']

        model_config = self.models[model_name]
        model_module = model_config['module']
        model_cls = model_config['cls']

        print(f"尝试导入模块和初始化模型: [{model_module}.{model_cls}]({model_init_parameters})...")

        module = importlib.import_module(model_module)
        cls = getattr(module, model_cls)

        model = cls(**model_init_parameters)
        print(f"初始化模型成功, type: {type(model)}")
        print(f"模型定义: {model}")
        return model

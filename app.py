import argparse

from wth import App
from wth.notification import DingtalkNotification
from context.config import Config

from context.trainer import TrainerContext


class CapsuleNetApp(App):
    def __init__(self):
        config = Config()
        notification = DingtalkNotification(
            webhook_url=config['dingtalk']['webhook'],
            secret=config['dingtalk']['secret']
        )
        super(CapsuleNetApp, self).__init__(notification, name="CapsuleNetApp")
        self.config = config

    def _run(self):
        config = self.config
        arg = argparse.ArgumentParser()
        # arg.add_argument('--base-line', type=str, choices=['cnn1d-2', 'cnn1d-10', 'cnn1d-20'], help='选择基线任务')
        arg.add_argument('--show-datasets', action='store_true', help='显示所有的数据集')
        arg.add_argument('--show-trainers', action='store_true', help='显示所有的训练任务')
        arg.add_argument('--show-models', action='store_true', help='显示所有的模型')
        arg.add_argument('--start-train', type=str, choices=config.trainers.keys(), help='选择训练器开始训练.')

        args = arg.parse_args()

        # 显示所有的数据集
        if args.show_datasets:
            config.display_datasets()

        if args.show_trainers:
            config.display_trainers()

        if args.show_models:
            config.display_models()

        if args.show_trainers or args.show_models or args.show_datasets:
            return

        trainer_context = TrainerContext(config)
        trainer_context.start_train(args.start_train)

import warnings
from typing import Type

import lightning as L

from finetune_minilm.tensorboard import TensorBoardWork


class TrainerWithTensorboard(L.LightningFlow):
    def __init__(self, work_cls: Type[L.LightningWork], cloud_compute: L.CloudCompute):
        super().__init__()
        tb_drive = L.app.storage.Drive("lit://tb_drive")
        warn_if_drive_not_empty(tb_drive)
        self.tensorboard_work = TensorBoardWork(drive=tb_drive)
        self.trainer_work = work_cls(cloud_compute=cloud_compute, tb_drive=tb_drive)

    def run(self, *args, **kwargs) -> None:
        self.tensorboard_work.run()
        self.trainer_work.run()

    def configure_layout(self):
        return [{"name": "Training Logs", "content": self.tensorboard_work.url}]


def warn_if_drive_not_empty(drive: L.app.storage.Drive):
    if drive.list():
        warnings.warn(
            "Drive is not empty! This may result in wrong logging behaviour if your app doesn't have a built-in resume"
            " mechanism. Consider deleting the .lightning file and restarting the app or giving it a new name with the"
            " --name flag of 'lightning run app'."
        )

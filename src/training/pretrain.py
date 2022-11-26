# third party imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import src.utils.utils
from src.utils.datasets import PretrainDataset
from src.modules.lit_detection import LitDetectionModule

if __name__ == "__main__":

    # defining callbacks
    # 1. checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="pretrain_loss",
        filename="detection-{epoch:02d}-{pretrain_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # defining the model
    detection_model = LitDetectionModule(
        pretrain_set=PretrainDataset(split_file=["train.txt"]),
        batch_size=16,
        num_points=4,
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=300,
        callbacks=[checkpoint_callback, lr_logger],
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group="pretraining",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 5, 6, 7],
    )
    trainer.fit(model=detection_model)

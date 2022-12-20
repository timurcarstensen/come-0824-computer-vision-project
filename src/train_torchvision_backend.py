# standard library imports
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import utilities.setup_utils
from src.modules.torchvision_backend.recognition_module import (
    RecognitionModule,
)
from src.utilities.datasets import TrainDataset, TestDataset


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision",
        filename="recognition-{epoch:02d}-{val_precision:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # defining the model
    recognition_module = RecognitionModule(
        batch_size=8,
        pretrained_model_path="resnet_backend.ckpt",
        transformer=False,
        num_dataloader_workers=16,
    )

    # print(detection_model)

    trainer = pl.Trainer(
        max_epochs=100,
        precision=16,
        callbacks=[checkpoint_callback, lr_logger],
        log_every_n_steps=1,
        logger=WandbLogger(  # initialise WandbLogger, modify the group based on your current experiment
            entity="default-entity-name",
            project="default-project-name",
            group="default-grou-name",
            save_dir=os.getenv("LOG_DIR"),
            log_model="all",
        ),
        accelerator="gpu",
        # devices=[0, 1, 2, 3],
    )

    trainer.fit(model=recognition_module)

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
        num_dataloader_workers=4,
    )

    # print(detection_model)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=100,
        # precision=16,
        # num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_logger],
        # limit_train_batches=0.001,
        # limit_test_batches=0.05,
        # limit_val_batches=0.001,
        log_every_n_steps=1,
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            name="no-transformer-fixed",
            group="training-resnet-backbone",
            log_model="all",
        ),
        # auto_scale_batch_size=True,
        # auto_lr_find=True,
        accelerator="gpu",
        devices=[0,1,2],#[1, 2, 3],
    )

    trainer.fit(model=recognition_module)

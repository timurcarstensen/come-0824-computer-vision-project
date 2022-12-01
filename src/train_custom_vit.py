# standard library imports
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import utils.utils
from modules.custom_vit.lit_vit import LitEnd2EndViT
from utils.datasets import TrainDataset, TestDataset


if __name__ == "__main__":

    model = LitEnd2EndViT(
        train_set=TrainDataset(),
        batch_size=2,  # 16 is the default, modify according to available GPU memory
    )

    # initialise ModelCheckpoint Callback, which saves the top 3 models based
    # on validation precision (no need to modify)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision",
        filename="recognition-{epoch:02d}-{val_precision:.2f}",
        save_top_k=3,
        mode="min",
    )

    # init learning rate callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    trainer = pl.Trainer(
        # fast_dev_run=True,  # uncomment this line to run a quick test of the model
        max_epochs=100,  # number of epochs to train for
        callbacks=[
            checkpoint_callback,
            lr_logger,
        ],  # add the checkpoint callback, no need to modify
        default_root_dir=os.getenv(
            "LOG_DIR"
        ),  # path to the logs folder, no need to modify
        strategy="ddp_find_unused_parameters_false",  # no need to modify
        log_every_n_steps=1,  # logging interval, no need to modify
        limit_val_batches=0.05,  # validation set size, decrease for increased performance (% of the validation set)
        limit_train_batches=0.05,  # analogous to limit_val_batches, no need to modify
        logger=WandbLogger(  # initialise WandbLogger, modify the group based on your current experiment
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group="custom_vit_pytorch_lightning",
            save_dir=os.getenv("LOG_DIR"),
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",  # modify this based on the machine you're running on
        devices=[2, 5],  # device indices for the GPUs
    )

    trainer.fit(model=model)

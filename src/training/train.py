# standard library imports
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

# local imports (i.e. our own code)
import src.utils.utils
from src.modules.lit_recognition import LitRecognitionModule
from src.datasets.datasets import TrainDataset, TestDataset


if __name__ == "__main__":

    # initialise model (the train and val dataset are passed in with default args)
    model = LitRecognitionModule(
        train_set=TrainDataset(),
        val_set=TestDataset(["val.txt"]),
        batch_size=16,
        pretrained_model_path="model.ckpt",
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        max_epochs=100,
        default_root_dir=os.getenv("LOG_DIR"),
        strategy="ddp_find_unused_parameters_false",
        log_every_n_steps=1,
        limit_val_batches=0.05,
        logger=WandbLogger(
            entity="timurcarstensen",
            project="cv-project",
            group="validation-checking",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        # accelerator="gpu",
        # devices=[0, 1],
    )

    trainer.fit(model=model)

# standard library imports
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import src.utils.utils
from src.modules.lit_recognition import LitRecognitionModule
from src.utils.datasets import TrainDataset, TestDataset


if __name__ == "__main__":

    # initialise model (the train dataset is initialised with the default args)

    model = LitRecognitionModule(
        train_set=TrainDataset(),
        val_set=TestDataset(["val.txt"]),
        batch_size=16,  # 16 is the default, modify according to available GPU memory
        pretrained_model_path="pretrained_model.ckpt",  # filename of the pretrained model (in src/logs),
        # make sure this file exists in the logs folder on the machine you're running on
    )

    # initialise ModelCheckpoint Callback, which saves the top 3 models based
    # on validation precision (no need to modify)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_precision",
        filename="recognition-{epoch:02d}-{val_precision:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        # fast_dev_run=True, # uncomment this line to run a quick test of the model
        max_epochs=150,  # number of epochs to train for
        callbacks=[checkpoint_callback],  # add the checkpoint callback, no need to modify
        default_root_dir=os.getenv(
            "LOG_DIR"
        ),  # path to the logs folder, no need to modify
        strategy="ddp_find_unused_parameters_false",  # no need to modify
        log_every_n_steps=1,  # logging interval, no need to modify
        limit_val_batches=1.0,  # validation set size, decrease for increased performance (% of the validation set)
        limit_train_batches=1.0,  # analogous to limit_val_batches, no need to modify
        logger=WandbLogger(  # initialise WandbLogger, modify the group based on your current experiment
            entity="timurcarstensen",
            project="cv-project",
            group="training-recognition",
            save_dir=os.getenv("LOG_DIR"),
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",  # modify this based on the machine you're running on
        devices=[0, 1],  # device indices for the GPUs
    )

    trainer.fit(model=model)

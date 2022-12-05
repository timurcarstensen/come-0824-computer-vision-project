# standard library imports
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import utilities.setup_utils
# from modules.pl_original_models.lit_recognition import LitRecognitionModule
from modules.lit_recognition_transformer import LitRecognitionModule_Transformer
from utilities.datasets import TrainDataset, TestDataset

if __name__ == "__main__":
    # TODO: start using the same hyperparameters as the authors for comparability (i.e. optimizer, batch_size, etc.)
    # TODO: check performance when using pretrained model from the git repo! (https://github.com/detectRecog/CCPD#demo)
    # initialise model (the train dataset is initialised with the default args)

    model = LitRecognitionModule_Transformer(
        train_set=TrainDataset(),
        val_set=TestDataset(["val.txt"]),
        batch_size=16,  # 16 is the default, modify according to available GPU memory
        # pretrained_model_path="pretrained_model.ckpt",  # filename of the pretrained model (in src/logs),
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

    # init learning rate callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    trainer = pl.Trainer(
        # fast_dev_run=True, # uncomment this line to run a quick test of the model
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
        # limit_val_batches=0.05,  # validation set size, decrease for increased performance (% of the validation set)
        # limit_train_batches=0.05,  # analogous to limit_val_batches, no need to modify
        logger=WandbLogger(  # initialise WandbLogger, modify the group based on your current experiment
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group="recognition_with_transformer",
            save_dir=os.getenv("LOG_DIR"),
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",  # modify this based on the machine you're running on
        devices=[0, 1, 2, 3, 4, 5, 6, 7],  # device indices for the GPUs
    )

    trainer.fit(model=model)

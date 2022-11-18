# third party imports
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# local imports (i.e. our own code)
import src.data_handlers.data_handlers
import src.utils.utils
from src.data_loaders.data_loaders import DataLoaderPreTrain
from src.modules.lit_detection import LitDetectionModule

if __name__ == "__main__":

    # TODO: add ArgParser
    # TODO: add DataParallel for multi-GPU training (cf. wR2.py)
    split_directories = ["train.txt"]

    batch_size = 256

    pretrain_loader = DataLoader(
        dataset=DataLoaderPreTrain(
            split_file=split_directories, img_size=(480, 480), test_mode=False
        ),
        num_workers=8,
    )

    num_points = 4
    detection_model = LitDetectionModule(num_points=num_points)

    trainer = pl.Trainer(
        max_epochs=100,
        logger=WandbLogger(
            project="cv-project",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 5, 6, 7],
    )
    trainer.fit(model=detection_model, train_dataloaders=pretrain_loader)

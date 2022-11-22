# third party imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# local imports (i.e. our own code)
import src.utils.utils
from src.datasets.datasets import PretrainDataset
from src.modules.lit_detection import LitDetectionModule

if __name__ == "__main__":

    batch_size = 256

    pretrain_loader = DataLoader(
        dataset=PretrainDataset(split_file=["train.txt"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    detection_model = LitDetectionModule(num_points=4)

    trainer = pl.Trainer(
        max_epochs=100,
        logger=WandbLogger(
            entity="timurcarstensen",
            project="cv-project",
            group="pretraining",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 5, 6, 7],
    )
    trainer.fit(model=detection_model, train_dataloaders=pretrain_loader)

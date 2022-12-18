# standard library imports
from typing import Optional
import sys

# 3rd party imports
from torchvision.io.image import read_image
from torchvision.models import resnet18, resnet50, resnet152

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
from src.utilities.datasets import TrainDataset, PretrainDataset
from src.utilities import setup_utils
from src.modules.utils import iou_and_gen_iou


class DetectionModule(pl.LightningModule):
    def __init__(self, batch_size: Optional[int] = 4):
        super(DetectionModule, self).__init__()

        self.batch_size = batch_size

        self.backbone = resnet50(weights="ResNet50_Weights.DEFAULT")

        self.box_regressor = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feature_map = self.backbone(x)
        x = self.box_regressor(feature_map)
        return x

    def train_dataloader(self):
        # returns a DataLoader for pretraining samples given the pretraining dataset
        return DataLoader(
            dataset=PretrainDataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def training_step(self, batch, batch_idx):
        """
        processes a batch of pretraining samples and returns the loss
        :param batch: a batch of pretraining samples of size self.batch_size
        :param batch_idx: index of the batch
        :return: loss
        """
        x, y = batch
        y = torch.stack(tensors=y).T

        y_pred = self(x)

        # TODO: I can't find this in the paper
        loss1 = 0.8 * nn.L1Loss()(y_pred[:, :2], y[:, :2])
        loss2 = 0.2 * nn.L1Loss()(y_pred[:, 2:], y[:, 2:])
        loss = loss1 + loss2

        iou, gen_iou = iou_and_gen_iou(y=y, y_pred=y_pred)

        self.log("IoU", {"train-IoU": iou, "train-gIoU": gen_iou})
        self.log("pretrain_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        returns the optimizer and learning rate scheduler used for pretraining
        :return:
        """
        # TODO: check that the parameters here are the same as used in the original paper!
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # return [optimizer], [lr_scheduler]

        # TODO: try adam
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor="pretrain_loss",
        filename="detection-{epoch:02d}-{pretrain_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # defining the model
    detection_model = DetectionModule(batch_size=16)

    # for i in detection_model.modules():
    #     print(i)

    print(detection_model)
    #
    # x = torch.randn((1, 64, 480, 480))
    # x1 = detection_model.backbone.layer1(x)
    # x2 = detection_model.backbone.layer2(x1)
    # x3 = detection_model.backbone.layer3(x2)
    #
    #
    # # print sizes of x to x3
    # print(x.shape)
    # print(x1.shape)
    # print(x2.shape)
    # print(x3.shape)
    #
    #
    # sys.exit()

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=200,
        precision=16,
        callbacks=[checkpoint_callback, lr_logger],
        # limit_train_batches=0.05,
        # limit_test_batches=0.05,
        # limit_val_batches=0.05,
        log_every_n_steps=1,
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group="pretraining-resnet-backbone",
            name="resnet152",
            log_model="all",
        ),
        # auto_scale_batch_size=True,
        # auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 3],
    )

    trainer.fit(model=detection_model)

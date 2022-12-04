# 3rd party imports
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.ops.boxes import box_iou, box_convert, generalized_box_iou

import pytorch_lightning as pl

# local imports (i.e. our own code)
from ..utils import iou_and_gen_iou


class LitDetectionModule(pl.LightningModule):
    def __init__(
        self,
        pretrain_set: torch.utils.data.Dataset = None,
        batch_size: int = 16,
        num_dataloader_workers: int = 8,
        num_points: int = 4,
    ):
        """
        Initializer for the LitDetectionModule class.
        :param pretrain_set: the pretraining dataset to use
        :param batch_size: batch size to use for training
        :param num_dataloader_workers: number of workers to use for the dataloader
        :param num_points: number of points to use for the detection model (this should always be 4, i.e. [x,y,w,h])]
        """
        super().__init__()

        self.pretrain_set = pretrain_set
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.num_points = num_points

        # TODO: log hyper-parameters to Weights & Biases
        self.save_hyperparameters()

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10,
        )

        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, self.num_points),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x

    def train_dataloader(self):
        # returns a DataLoader for pretraining samples given the pretraining dataset
        return DataLoader(
            dataset=self.pretrain_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
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

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


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
        self.pretrain_set = pretrain_set
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.num_points = num_points

        # TODO: log hyperparameters to Weights & Biases
        self.save_hyperparameters()

        super().__init__()

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

        # TODO: Why are we not using non-linearities here? (i.e. why are the ReLUs missing?)
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
        ios = []
        for i, box in enumerate(y_pred):
            a, b, c, d = box
            boxA = [a, b, a + c, b + d]
            ap, bp, cp, dp = y[i]
            boxB = [ap, bp, ap + cp, bp + dp]
            # print(bb_intersection_over_union(boxA, boxB).item())
            ios.append(bb_intersection_over_union(boxA, boxB).item())

        # TODO: I can't find this in the paper
        loss1 = 0.8 * nn.L1Loss()(y_pred[:, :2], y[:, :2])
        loss2 = 0.2 * nn.L1Loss()(y_pred[:, 2:], y[:, 2:])
        loss = loss1 + loss2

        # general loss:
        # loss = nn.L1Loss()(y_pred, y)
        # TODO: implement Intersection over Union (IoU) metric logging
        #  (cf. https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html)
        self.log("pretrain_loss", loss)
        self.log("Intersection over Union", np.mean(ios))
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


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
# standard library imports
from typing import Optional, Tuple
import os
import itertools
from statistics import mean

# 3rd party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.modules.custom_vit.lit_vit import LitEnd2EndViT
from src.modules.custom_vit.vit_utils import Transformer

# local imports (i.e. our own code)
from src.utilities.datasets import TrainDataset, TestDataset
from src.modules.utils import iou_and_gen_iou
from src.modules.torchvision_backend.detection_module import DetectionModule
from src.modules.torchvision_backend.crop_module import CropModule


class RecognitionModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_path: Optional[str] = None,
        fine_tuning: Optional[bool] = False,
        transformer: Optional[bool] = True,
        batch_size: Optional[int] = 4,
        crop_size: Tuple[int, int] = (128, 128),
        province_num: Optional[int] = 38,
        alphabet_num: Optional[int] = 25,
        alphabet_numbers_num: Optional[int] = 35,
        num_dataloader_workers: Optional[int] = 4,
        plate_character_criterion: Optional[nn.Module] = nn.CrossEntropyLoss(),
    ):
        """
        Initialises a RecognitionModule
        :param pretrained_model_path: path to pretrained weights for DetectionModule
        :param fine_tuning: whether to activate gradient updates for the DetectionModule
        :param batch_size: batch_size
        :param crop_size: tuple, size of the cropped images passed to the classifiers
        :param plate_character_criterion: loss criterion for the LP recognition
        :param province_num: int, num classes (ignore)
        :param alphabet_num: int, num classes (ignore)
        :param alphabet_numbers_num: int, num classes (ignore)
        """

        super(RecognitionModule, self).__init__()

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.use_transformer = transformer
        self.num_dataloader_workers = num_dataloader_workers
        self.plate_character_criterion = plate_character_criterion

        # initialise detection module
        self.detection_module = DetectionModule()

        # load pretrained weights if provided
        if pretrained_model_path:
            print("Loading pretrained model from: ", pretrained_model_path)
            self.detection_module.load_from_checkpoint(
                checkpoint_path=f"{os.getenv('LOG_DIR')}{pretrained_model_path}",
            )

        # if not fine_tuning, disable gradient updates for the detection module
        if not fine_tuning:
            for param in self.detection_module.parameters():
                param.requires_grad = False

        # initialise crop module
        self.crop = CropModule(size=crop_size)

        # use one convolutional layer
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        # if transformer then use vision transformer after cropping and before classifier

        self.vit = LitEnd2EndViT(
            train_set=None,
            image_size=64,
            patch_size=8,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            channels=8,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(57 * 57 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, province_num),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_num),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier5 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )

        self.classifier7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 64 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )

    def forward(self, x):
        detect_output = self.detection_module(x)
        cropped = self.crop((x, detect_output))

        # use one convolutional layer
        convolved = self.conv_layer(
            cropped.view(-1, 3, self.crop_size[0], self.crop_size[1])
        )

        if not self.use_transformer:
            y0 = self.classifier1(convolved.view(self.batch_size, -1))
            y1 = self.classifier2(convolved.view(self.batch_size, -1))
            y2 = self.classifier3(convolved.view(self.batch_size, -1))
            y3 = self.classifier4(convolved.view(self.batch_size, -1))
            y4 = self.classifier5(convolved.view(self.batch_size, -1))
            y5 = self.classifier6(convolved.view(self.batch_size, -1))
            y6 = self.classifier7(convolved.view(self.batch_size, -1))

        else:
            y0, y1, y2, y3, y4, y5, y6 = self.vit(convolved)

        return detect_output, [y0, y1, y2, y3, y4, y5, y6]

    def train_dataloader(self):
        # returns a DataLoader for pretraining samples given the pretraining dataset
        return DataLoader(
            dataset=TrainDataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TestDataset(split_file=["val.txt"]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
        )

    def validation_step(self, batch, batch_idx):

        x, labels, ims = batch

        # convert the labels to tensor of shape (7, batch_size)
        y_i = torch.tensor(
            data=[[int(elem) for elem in label.split("_")[:7]] for label in labels],
            device=self.device,
        ).T

        # TODO: modify implementation of the TestDataset s.t. that the bounding boxes are returned too
        _, y_pred = self(x)

        # getting the argmax for each of the 7 digits for each element in the batch (also shape of (7, batch_size))
        y_pred = torch.stack(tensors=[torch.argmax(input=elem, dim=1) for elem in y_pred])

        # getting the element wise equality of the predictions and the labels
        per_sample_performance = torch.sum(
            input=torch.eq(input=y_pred, other=y_i), dim=0, dtype=torch.float32
        )

        # summing over the correct predictions for each classifier
        per_classifier_performance = torch.sum(
            input=torch.eq(input=y_pred, other=y_i), dim=1, dtype=torch.float32
        )

        # reshaping s.t. we have tensors of shape (1, num_classifiers)
        per_classifier_performance = per_classifier_performance.view(1, -1)

        # calculating the mean accuracy for each classifier (for the current validation batch)
        per_classifier_performance = torch.div(
            input=per_classifier_performance, other=len(x)
        )

        return {
            "correct_samples": torch.where(
                per_sample_performance == 7, 1.0, 0.0
            ).tolist(),
            "classifier_performance": per_classifier_performance,
            "correct_label_predictions": torch.mean(per_sample_performance),
        }

    def validation_epoch_end(self, outputs) -> None:

        # TODO: is this accuracy or precision? (should be precision afaik)
        # mean of accuracy for each classifier over the entire validation set
        per_classifier_performance = torch.mean(
            torch.stack(tensors=[elem["classifier_performance"] for elem in outputs]),
            dim=0,
        )

        # logging per classifier performance
        self.log_dict(
            dictionary={
                f"classifier_{str(i)}": per_classifier_performance[:, i] for i in range(7)
            },
            sync_dist=True,
        )

        # logging validation precision
        self.log(
            "val_precision",
            mean(
                list(
                    itertools.chain.from_iterable(
                        elem["correct_samples"] for elem in outputs
                    )
                )
            ),
            sync_dist=True,
        )

        # logging the average number of correctly predicted labels per classifier
        self.log(
            "avg_correct_labels",
            sum([elem["correct_label_predictions"] for elem in outputs]) / len(outputs),
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        x, box_gt, labels, ims = batch

        labels = [[int(elem) for elem in label.split("_")[:7]] for label in labels]

        box_gt = torch.stack(tensors=box_gt).T

        x = x.clone().detach()

        box_pred, lp_char_pred = self(x)

        bounding_loss = torch.tensor(data=[0.0], device=self.device)
        bounding_loss += 0.8 * nn.L1Loss()(box_pred[:, :2], box_gt[:, :2])
        bounding_loss += 0.2 * nn.L1Loss()(box_pred[:, 2:], box_gt[:, 2:])

        character_loss = torch.tensor(data=[0.0], device=self.device)
        for j in range(7):
            l = torch.tensor(
                data=[elem[j] for elem in labels], dtype=torch.long, device=self.device
            )
            character_loss += self.plate_character_criterion(lp_char_pred[j], l)

        loss = bounding_loss + character_loss

        iou, gen_iou = iou_and_gen_iou(y=box_gt, y_pred=box_pred)

        self.log("IoU", {"train-IoU": iou, "train-gIoU": gen_iou})
        self.log("train_loss", loss)

        self.log("bounding_loss", bounding_loss)
        self.log("character_loss", character_loss)
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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

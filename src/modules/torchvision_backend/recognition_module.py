# standard library imports
from typing import Optional, Tuple
import os

# 3rd party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
from src.utilities.datasets import TrainDataset
from src.modules.utils import iou_and_gen_iou
from src.modules.torchvision_backend.detection_module import DetectionModule
from src.modules.torchvision_backend.crop_module import CropModule


class RecognitionModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_path: Optional[str] = None,
        fine_tuning: Optional[bool] = False,
        batch_size: Optional[int] = 4,
        crop_size: Tuple[int, int] = (224, 224),
        province_num: Optional[int] = 38,
        alphabet_num: Optional[int] = 25,
        alphabet_numbers_num: Optional[int] = 35,
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

        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(
                crop_size[0] * crop_size[1] * 3, 128
            ),  # insert size of cropped image
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, province_num),
        )
        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_num),
        )
        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )

        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(crop_size[0] * crop_size[1] * 3, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphabet_numbers_num),
        )

    def forward(self, x):
        detect_output = self.detection_module(x)
        cropped = self.crop((x, detect_output))

        y0 = self.classifier1(cropped)  # input to each classifier is cropped image
        y1 = self.classifier2(cropped)
        y2 = self.classifier3(cropped)
        y3 = self.classifier4(cropped)
        y4 = self.classifier5(cropped)
        y5 = self.classifier6(cropped)
        y6 = self.classifier7(cropped)

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
    recognition_module = RecognitionModule(
        batch_size=4, pretrained_model_path="resnet_backend.ckpt"
    )

    # print(detection_model)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=100,
        callbacks=[checkpoint_callback, lr_logger],
        # limit_train_batches=0.05,
        # limit_test_batches=0.05,
        # limit_val_batches=0.05,
        log_every_n_steps=1,
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group="pretraining-resnet-backbone",
            log_model="all",
        ),
        # auto_scale_batch_size=True,
        # auto_lr_find=True,
        # accelerator="gpu",
        # devices=[0, 1],
    )

    trainer.fit(model=recognition_module)

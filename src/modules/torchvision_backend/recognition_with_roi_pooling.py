# standard library imports
from typing import Optional
import os
import itertools
from statistics import mean

# 3rd party imports
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
from .detection_module import DetectionModule
from ..utils import roi_pooling_ims, iou_and_gen_iou


class Squeeze(pl.LightningModule):
    def forward(self, x):
        return x.squeeze()


class RecognitionModule(pl.LightningModule):
    def __init__(
        self,
        train_set: torch.utils.data.Dataset,
        val_set: torch.utils.data.Dataset,
        num_dataloader_workers: Optional[int] = 4,
        batch_size: Optional[int] = 2,
        pretrained_model_path: Optional[str] = None,
        province_num: Optional[int] = 38,
        alphabet_num: Optional[int] = 25,
        alphabet_numbers_num: Optional[int] = 35,
        plate_character_criterion=nn.CrossEntropyLoss(),  # TODO: WHY ARE WE USING CROSS ENTROPY LOSS HERE?
        limit_train_batches: Optional[float] = 1.0,
        limit_val_batches: Optional[float] = 1.0,
    ):
        """
        Initialize the recognition module
        :param train_set: “torch.utils.data.Dataset” for training
        :param val_set: “torch.utils.data.Dataset” for validation
        :param num_dataloader_workers: number of workers for the dataloader
        :param batch_size: batch size
        :param pretrained_model_path: path to the pretrained model
        :param province_num: number of provinces
        :param alphabet_num: number of characters in the alphabet
        :param alphabet_numbers_num: number of characters in the alphabet and numbers
        :param plate_character_criterion: loss function for the plate character classifier
        """
        super().__init__()

        # TODO: log hyper-parameters to Weights & Biases
        self.save_hyperparameters(ignore=["plate_character_criterion"])

        # 1. setting variables

        # init the detection module
        self.detection_module = DetectionModule()

        # load the pretrained detection module if a path is provided
        if pretrained_model_path:
            self.detection_module.load_from_checkpoint(
                checkpoint_path=f"{os.getenv('LOG_DIR')}{pretrained_model_path}",
                strict=False,
            )

        self.plate_character_criterion = plate_character_criterion

        self.train_set = train_set
        self.val_set = val_set
        self.num_dataloader_workers = num_dataloader_workers
        self.batch_size = batch_size
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        # 2. defining the model
        self.downsample1 = nn.Sequential(
            self.detection_module.backbone.avgpool,
            Squeeze(),
            self.detection_module.backbone.fc,
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(1792, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )

        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, province_num),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_num),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier6 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier7 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16640, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )

    def forward(self, x):
        x0 = self.detection_module.backbone.conv1(x)
        _x1 = self.detection_module.backbone.layer1(x0)
        _x2 = self.detection_module.backbone.layer2(_x1)
        _x3 = self.detection_module.backbone.layer3(_x2)
        _x4 = self.detection_module.backbone.layer4(_x3)
        x5 = self.downsample1(_x4)

        # x5 = _x5.view(_x5.size(0), -1)
        box_loc = self.detection_module.box_regressor(x5)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = torch.tensor(
            data=[[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        h2, w2 = _x2.data.size()[2], _x2.data.size()[3]
        p2 = torch.tensor(
            data=[[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        h3, w3 = _x3.data.size()[2], _x3.data.size()[3]
        p3 = torch.tensor(
            data=[[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        # x, y, w, h --> x1, y1, x2, y2
        if not box_loc.data.size()[1] == 4:
            raise ValueError(f"box_loc.data.size()[1] != 4 but {box_loc.data.size()[1]}")

        postfix = torch.tensor(
            data=[[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        box_new = box_loc.mm(postfix).clamp(min=0, max=1)

        roi1 = roi_pooling_ims(
            t=_x1, rois=box_new.mm(p1), size=(8, 16), device=self.device
        )
        roi2 = roi_pooling_ims(
            t=_x2, rois=box_new.mm(p2), size=(8, 16), device=self.device
        )
        roi3 = roi_pooling_ims(
            t=_x3, rois=box_new.mm(p3), size=(8, 16), device=self.device
        )
        rois = torch.cat((roi1, roi2, roi3), 1)

        rois = self.downsample2(rois)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        return box_loc, [y0, y1, y2, y3, y4, y5, y6]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=16,
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

        y_i = [[int(elem) for elem in label.split("_")[:7]] for label in labels]

        box_gt = torch.stack(tensors=box_gt).T

        x = x.clone().detach()

        box_pred, lp_char_pred = self(x)

        bounding_loss = torch.tensor(data=[0.0], device=self.device)
        bounding_loss += 0.8 * nn.L1Loss()(box_pred[:, :2], box_gt[:, :2])
        bounding_loss += 0.2 * nn.L1Loss()(box_pred[:, 2:], box_gt[:, 2:])

        character_loss = torch.tensor(data=[0.0], device=self.device)
        for j in range(7):
            l = torch.tensor(
                data=[elem[j] for elem in y_i], dtype=torch.long, device=self.device
            )
            character_loss += self.plate_character_criterion(lp_char_pred[j], l)

        loss = 10 * bounding_loss + character_loss

        iou, gen_iou = iou_and_gen_iou(y=box_gt, y_pred=box_pred)

        self.log("IoU", {"train-IoU": iou, "train-gIoU": gen_iou})
        self.log("train_loss", loss)

        self.log("bounding_loss", bounding_loss)
        self.log("character_loss", character_loss)
        return loss

    def configure_optimizers(self):
        # TODO: are we actually using the correct optimizer? i.e. the one from the paper with the correct HPs?
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # return [optimizer], [lr_scheduler]
        #
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=10, gamma=0.1
        )
        return [optimizer], [lr_scheduler]


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
    detection_model = RecognitionModule(batch_size=16)

    # for i in detection_model.modules():
    #     print(i)

    # print(detection_model)
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

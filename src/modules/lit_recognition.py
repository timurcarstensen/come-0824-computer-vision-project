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

# local imports (i.e. our own code)
from src.modules.lit_detection import LitDetectionModule
from src.modules.roi_pooling import roi_pooling_ims


class LitRecognitionModule(pl.LightningModule):
    def __init__(
        self,
        train_set: torch.utils.data.Dataset,
        val_set: torch.utils.data.Dataset,
        num_dataloader_workers: Optional[int] = 4,
        batch_size: Optional[int] = 2,
        num_points: Optional[int] = 4,
        pretrained_model_path: Optional[str] = None,
        province_num: Optional[int] = 38,
        alphabet_num: Optional[int] = 25,
        alphabet_numbers_num: Optional[int] = 35,
        plate_character_criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()

        # 1. setting variables

        # init the detection module
        self.detection_module = LitDetectionModule(num_points=num_points)

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

        self.correct: int = 0
        self.incorrect: int = 0

        # 2. defining the model
        self.classifier1 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, province_num),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_num),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier6 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier7 = nn.Sequential(
            nn.Linear(53248, 128),
            nn.Linear(128, alphabet_numbers_num),
        )

    def forward(self, x):
        x0 = self.detection_module.features[0](x)
        _x1 = self.detection_module.features[1](x0)
        x2 = self.detection_module.features[2](_x1)
        _x3 = self.detection_module.features[3](x2)
        x4 = self.detection_module.features[4](_x3)
        _x5 = self.detection_module.features[5](x4)

        x6 = self.detection_module.features[6](_x5)
        x7 = self.detection_module.features[7](x6)
        x8 = self.detection_module.features[8](x7)
        x9 = self.detection_module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        box_loc = self.detection_module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = torch.tensor(
            data=[[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]

        p2 = torch.tensor(
            data=[[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = torch.tensor(
            data=[[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]],
            requires_grad=False,
            dtype=torch.float32,
            device=self.device,
        )

        # x, y, w, h --> x1, y1, x2, y2
        if not box_loc.data.size()[1] == 4:
            raise ValueError("box_loc.data.size()[1] != 4")

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
            t=_x3, rois=box_new.mm(p2), size=(8, 16), device=self.device
        )
        roi3 = roi_pooling_ims(
            t=_x5, rois=box_new.mm(p3), size=(8, 16), device=self.device
        )
        rois = torch.cat((roi1, roi2, roi3), 1)

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

        _, y_pred = self(x)

        # getting the argmax for each of the 7 digits for each element in the batch (also shape of (7, batch_size))
        y_pred = torch.stack(tensors=[torch.argmax(input=elem, dim=1) for elem in y_pred])

        # getting the element wise equality of the predictions and the labels
        equality = torch.sum(
            input=torch.eq(input=y_pred, other=y_i), dim=0, dtype=torch.float32
        )

        return {
            "correct": torch.where(equality == 7, 1.0, 0.0).tolist(),
            "correct_label_predictions": torch.mean(equality),
        }

    def validation_epoch_end(self, outputs) -> None:
        self.log(
            "val_precision",
            mean(
                list(itertools.chain.from_iterable(elem["correct"] for elem in outputs))
            ),
            sync_dist=True,
        )

        self.log(
            "avg_correct_labels",
            sum([elem["correct_label_predictions"] for elem in outputs]) / len(outputs),
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        x, y, labels, ims = batch

        y_i = [[int(elem) for elem in label.split("_")[:7]] for label in labels]

        y = torch.stack(tensors=y).T

        x = x.clone().detach()

        fps_pred, y_pred = self(x)

        loss = torch.tensor(data=[0.0], device=self.device)
        loss += 0.8 * nn.L1Loss()(fps_pred[:, :2], y[:, :2])
        loss += 0.2 * nn.L1Loss()(fps_pred[:, 2:], y[:, 2:])

        for j in range(7):
            l = torch.tensor(
                data=[elem[j] for elem in y_i], dtype=torch.long, device=self.device
            )
            loss += self.plate_character_criterion(y_pred[j], l)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=5, gamma=0.1
        )
        return [optimizer], [lr_scheduler]

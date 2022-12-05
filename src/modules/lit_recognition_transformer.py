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
# from vit_pytorch import ViT
from torchvision.transforms.functional import crop
from torchvision.transforms import Resize
from torchvision.ops import box_convert

# local imports (i.e. our own code)
from src.modules.pl_original_models.lit_detection import LitDetectionModule
from .utils import roi_pooling_ims
from .custom_vit.lit_vit import LitEnd2EndViT


class LitRecognitionModule_Transformer(pl.LightningModule):
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
            plate_character_criterion=nn.CrossEntropyLoss(),  # TODO: WHY ARE WE USING CROSS ENTROPY LOSS HERE?
            resize_size=(64, 128),
    ):
        """
        Initialize the recognition module
        :param train_set: “torch.utils.data.Dataset” for training
        :param val_set: “torch.utils.data.Dataset” for validation
        :param num_dataloader_workers: number of workers for the dataloader
        :param batch_size: batch size
        :param num_points: number of points for coordinates of the bounding box, should always be 4 (x, y, w, h)
        :param pretrained_model_path: path to the pretrained model
        :param province_num: number of provinces
        :param alphabet_num: number of characters in the alphabet
        :param alphabet_numbers_num: number of characters in the alphabet and numbers
        :param plate_character_criterion: loss function for the plate character classifier
        :param resize_size: size to which the images should be resized
        """
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

        # Use ViT as classifier

        self.vit = LitEnd2EndViT(
            train_set=None,
            image_size=resize_size[0] * resize_size[1],
            patch_size=32,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            channels=3,
            dim_head=64,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.resize = Resize(size=resize_size)

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

        # crop x with box_loc
        box_loc = box_convert(box_loc, in_fmt="cxcywh", out_fmt="xywh")
        print(box_loc)
        x_cropped = crop(x, top=box_loc[:, 0], left=box_loc[:, 1], height=box_loc[:, 2], width=box_loc[:, 3])

        # resize x_cropped to 64*128
        x_resized = self.resize(x_cropped)

        # feed x_resized through ViT
        lp_prediction = self.vit(x_resized)

        return box_loc, lp_prediction

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
        x, y, labels, ims = batch

        y_i = [[int(elem) for elem in label.split("_")[:7]] for label in labels]

        y = torch.stack(tensors=y).T

        x = x.clone().detach()

        fps_pred, y_pred = self(x)

        bounding_loss = torch.tensor(data=[0.0], device=self.device)
        bounding_loss += 0.8 * nn.L1Loss()(fps_pred[:, :2], y[:, :2])
        bounding_loss += 0.2 * nn.L1Loss()(fps_pred[:, 2:], y[:, 2:])

        character_loss = torch.tensor(data=[0.0], device=self.device)
        for j in range(7):
            l = torch.tensor(
                data=[elem[j] for elem in y_i], dtype=torch.long, device=self.device
            )
            character_loss += self.plate_character_criterion(y_pred[j], l)

        loss = bounding_loss + character_loss

        self.log("train_loss", loss)
        self.log("bounding_loss", bounding_loss)
        self.log("character_loss", character_loss)
        return loss

    def configure_optimizers(self):
        # TODO: are we actually using the correct optimizer? i.e. the one from the paper with the correct HPs?
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #    optimizer=optimizer, step_size=5, gamma=0.1
        # )
        # return [optimizer], [lr_scheduler]
        return optimizer

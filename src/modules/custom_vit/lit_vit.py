# standard library imports
from typing import Optional

# 3rd party imports
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import repeat
from einops.layers.torch import Rearrange
from torchmetrics.classification.accuracy import MulticlassAccuracy

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import src.utils.utils
from .utils import Transformer, pair


class LitEnd2EndViT(pl.LightningModule):
    def __init__(
        self,
        train_set: torch.utils.data.Dataset,
        num_dataloader_workers: Optional[int] = 8,
        batch_size: Optional[int] = 16,
        image_size: Optional[int] = 480,
        patch_size: Optional[int] = 32,
        dim: Optional[int] = 1024,
        depth: Optional[int] = 6,
        heads: Optional[int] = 8,
        mlp_dim: Optional[int] = 2048,
        pool: Optional[str] = "cls",
        channels: Optional[int] = 3,
        dim_head: Optional[int] = 64,
        dropout: Optional[float] = 0.1,
        emb_dropout: Optional[float] = 0.1,
        province_num: Optional[int] = 38,
        alphabet_num: Optional[int] = 25,
        alphabet_numbers_num: Optional[int] = 35,
        plate_character_criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # setting
        self.batch_size = batch_size
        self.train_set = train_set
        self.num_dataloader_workers = num_dataloader_workers
        self.plate_character_criterion = plate_character_criterion

        if not image_height % patch_height == 0 and image_width % patch_width == 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        if not pool in {"cls", "mean"}:
            raise ValueError(
                "pool type must be either cls (cls token) or mean (mean pooling)"
            )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, province_num),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_num),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier6 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )
        self.classifier7 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, alphabet_numbers_num),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        y0 = self.classifier1(x)
        y1 = self.classifier2(x)
        y2 = self.classifier3(x)
        y3 = self.classifier4(x)
        y4 = self.classifier5(x)
        y5 = self.classifier6(x)
        y6 = self.classifier7(x)
        return [y0, y1, y2, y3, y4, y5, y6]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=True,
        )

    def training_step(self, batch, batch_idx):
        image, lp_box, lp_char_cls, img_name = batch

        lp_char_cls = [
            [int(elem) for elem in label.split("_")[:7]] for label in lp_char_cls
        ]
        lp_char_cls = torch.tensor(lp_char_cls)

        image = image.clone().detach()

        char_prediction = self(image)

        # noinspection DuplicatedCode
        loss = torch.tensor(data=[0.0], device=self.device)

        test = [25, 38, 35, 35, 35, 35, 35]
        for j in range(7):
            char_gt = torch.tensor(
                data=[elem[j] for elem in lp_char_cls],
                dtype=torch.long,
                device=self.device,
            )

            if not (max([elem[j] for elem in lp_char_cls]) <= test[j]):
                raise ValueError(f"Character class {j} must be less than {test[j]}.")

            loss += self.plate_character_criterion(char_prediction[j], char_gt)

            acc = MulticlassAccuracy(num_classes=char_prediction[j].size()[1]).to(
                self.device
            )
            self.log(f"classifier_{j}_accuracy", acc(char_prediction[j], char_gt))

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # return [optimizer], [lr_scheduler]
        return optimizer


if __name__ == "__main__":
    pass

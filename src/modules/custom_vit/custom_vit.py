# TODO: this can be deleted, is implemented in pytorch lightning in lit_vit and utils

import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import src.utils.utils

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.utils.datasets import TrainDataset, TestDataset
import wandb


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Custom_ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, num_classes)
        # )

        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 25),
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 38),
        )
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 35),
        )
        self.classifier4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 35),
        )
        self.classifier5 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 35),
        )
        self.classifier6 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 35),
        )
        self.classifier7 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 35),
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
        # return [self.classifier1, self.classifier2, self.classifier3]#self.mlp_head(x)


def train_model(model, num_epochs=25, batch_size=32, use_gpu=True):
    # since = time.time()
    if use_gpu:
        # model = torch.nn.DataParallel(
        #    model, device_ids=range(torch.cuda.device_count())
        # )
        model = model.cuda()

    print("Creating Dataset...")
    dataset = TrainDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Dataset created.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        lossAver = []
        model.train(True)
        # lrScheduler.step()
        # start = time()
        print("Epoch: ", epoch)
        for i, (XI, Y, labels, ims) in enumerate(tqdm(train_loader)):
            # print("Batch: ", i)
            if not len(XI) == batch_size:
                continue

            YI = [[int(ee) for ee in el.split("_")[:7]] for el in labels]
            Y = np.array([el.numpy() for el in Y]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(Y).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                y = Variable(torch.FloatTensor(Y), requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            y_pred = model(x)

            # Compute and print loss
            loss = 0.0
            # loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
            # loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
            for j in range(7):
                if use_gpu:
                    l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                else:
                    l = Variable(torch.LongTensor([el[j] for el in YI]))
                loss += nn.CrossEntropyLoss()(y_pred[j], l)
            # print("Loss: ", loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                lossAver.append(loss.data[0])
            except:
                pass

            wandb.log({"Loss": loss})
        print("%s %s\n" % (epoch, sum(lossAver) / len(lossAver)))
        model.eval()
        # count, correct, error, precision, avgTime = eval(model, testDirs)
        # torch.save(model.state_dict(), storeName + str(epoch))
        # save trained model to file
    return model


wandb.login(key="e5e086aa1769f55b05d2cb71ba817e392d6c1e40")
wandb.init(
    entity="mtp-ai-board-game-engine",
    project="cv-project",
    group="vit_training",
)
model = Custom_ViT(
    image_size=480,
    patch_size=32,
    num_classes=0,
    dim=128,
    depth=3,
    heads=4,
    mlp_dim=512,
    pool="cls",
    channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
)

mod = train_model(model, num_epochs=100, batch_size=4, use_gpu=False)
print("Finished")

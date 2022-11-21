# standard library imports
import os
from time import time
from typing import Optional, Tuple, List

# 3rd party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

# local imports (i.e. our own code)
import src.data_handlers.data_handlers
import src.utils.utils
from src.modules.lit_recognition import LitRecognitionModule
from src.datasets.datasets import TrainDataset, TestDataset


def evaluate(
    model: nn.Module,
    test_dirs: List[str] | str,
    img_size: Tuple[int, int],
    use_gpu: Optional[bool] = False,
    standalone: Optional[bool] = False,
    model_path: Optional[str] = None,
):
    if standalone:
        if not model_path:
            raise ValueError("model_path must be provided when standalone is True")

        print("Loading rpnet model weights...")
        model.load_state_dict(torch.load(f=f"{os.getenv('MODEL_DIR')}{model_path}"))
        print("Loaded rpnet model weights.")

    count, error, correct = 0, 0, 0
    test_loader = DataLoader(
        dataset=TestDataset(split_file=test_dirs, img_size=img_size),
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )

    model.eval()

    start = time()

    for idx, (XI, labels, ims) in enumerate(test_loader):
        count += 1
        YI = [[int(elem) for elem in label.split("_")[:7]] for label in labels]

        if use_gpu:
            x = XI.clone().detach().cuda()
        else:
            x = XI.clone().detach()
        # Forward pass: Compute predicted y by passing x to the model

        _, y_pred = model(x)

        output_y = [elem.data.cpu().numpy().tolist() for elem in y_pred]

        label_pred = [t[0].index(max(t[0])) for t in output_y]

        #   compare YI, outputY

        try:

            if LitRecognitionModule._is_equal(label_pred, YI[0]) == 7:
                correct += 1

            else:
                pass

        except:
            error += 1

        if idx % 500 == 0:
            print(
                "total %s correct %s error %s precision %s avg_time %s"
                % (
                    count,
                    correct,
                    error,
                    float(correct) / count,
                    (time() - start) / count,
                )
            )
    return count, correct, error, float(correct) / count, (time() - start) / count


if __name__ == "__main__":

    num_points = 4
    img_size = (480, 480)

    model = LitRecognitionModule(
        train_set=TrainDataset(split_file=["train.txt"], img_size=img_size),
        val_set=TestDataset(split_file=["val.txt"], img_size=img_size),
        num_points=num_points,
        batch_size=256,
        pretrained_model_path="model.ckpt",
    )

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=100,
        strategy="ddp_find_unused_parameters_false",
        log_every_n_steps=1,
        logger=WandbLogger(
            project="cv-project",
            name="training-pretrained",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 4, 5, 6, 7],
    )

    trainer.fit(model=model)

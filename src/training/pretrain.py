# standard library imports
import os
from typing import Optional
import time

# third party imports
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np


# local imports (i.e. our own code)
import src.data_handlers.data_handlers
from src.data_loaders.data_loaders import DataLoaderPreTrain
from src.modules.detection import DetectionModule


def pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    batch_size: Optional[int] = 16,
    num_epochs: Optional[int] = 300,
    use_gpu: Optional[bool] = False,
    criterion: torch.nn.modules.loss._Loss = nn.L1Loss(),
    lr_scheduler_step_size: Optional[int] = 5,
    lr_scheduler_gamma: Optional[float] = 0.1,
    save_model: Optional[bool] = True,
):
    # initialize the learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
    )

    start = time.time()

    for epoch in range(num_epochs):
        loss_avg = []
        model.train(mode=True)

        for idx, train_data in enumerate(train_loader):
            XI, YI = train_data
            YI = np.array([elem.numpy() for elem in YI]).T
            # TODO: stop using torch.autograd.Variable
            if use_gpu:
                x = XI.clone().detach().cuda(device=torch.device("cuda:6"))
                y = torch.tensor(YI, dtype=torch.float32, requires_grad=False).cuda(
                    device=torch.device("cuda:6")
                )
            else:
                x = XI.clone().detach()
                y = torch.tensor(YI, dtype=torch.float32, requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            y_pred = model(x)

            # Compute and print loss
            running_loss = 0.0

            if len(y_pred) == batch_size:

                loss1 = 0.8 * criterion.cuda(
                    device=torch.device("cuda:6") if torch.device.type == "cuda" else None
                )(y_pred[:, :2], y[:, :2])

                loss2 = 0.2 * criterion.cuda(
                    device=torch.device("cuda:6") if torch.device.type == "cuda" else None
                )(y_pred[:, 2:], y[:, 2:])

                loss = loss1 + loss2
                running_loss += loss1.item() + loss2.item()
                loss_avg.append(running_loss)

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # if not idx % 50:
                #     print(
                #         f"Epoch: {epoch + 1}/{num_epochs}, "
                #         f"Iteration: {idx + 1}/{len(train_loader)}, "
                #         f"Loss: {np.mean(loss_avg):.4f}, "
                #         f"Time: {time.time() - start:.4f}"
                #     )
                #
                #     loss_avg = []

                print(
                    f"Epoch: {epoch + 1}/{num_epochs}, "
                    f"Iteration: {idx + 1}/{len(train_loader)}, "
                    f"Loss: {np.mean(loss_avg):.4f}, "
                    f"Time: {time.time() - start:.4f}"
                )

                loss_avg = []

        print(f"epoch {epoch} average loss: {sum(loss_avg) / len(loss_avg)}")

    if save_model:
        print("Saving model...")
        torch.save(
            obj=model.state_dict(),
            f=f"{os.getenv('MODEL_DIR')}pretrained_detection_module.pt",
        )
        print("Model saved.\n Finished training.")

    return model


if __name__ == "__main__":

    # TODO: add ArgParser
    # TODO: add DataParallel for multi-GPU training (cf. wR2.py)
    split_directories = ["train.txt"]

    batch_size = 256

    pretrain_loader = DataLoader(
        dataset=DataLoaderPreTrain(
            split_file=split_directories, img_size=(480, 480), test_mode=False
        ),
        num_workers=8,
    )

    num_classes = 4
    detection_model = DetectionModule(num_classes)

    if torch.cuda.is_available():
        detection_model = torch.nn.DataParallel(
            detection_model, device_ids=[0, 1, 2, 4, 5, 6, 7]
        )
        detection_model = detection_model.cuda()

    trained_model = pretrain_model(
        model=detection_model,
        train_loader=pretrain_loader,
        use_gpu=torch.cuda.is_available(),
        batch_size=batch_size,
        optimizer=optim.Adam(detection_model.parameters(), lr=0.01),
    )

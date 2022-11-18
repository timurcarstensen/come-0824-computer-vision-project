# Compared to fh0.py
# fh02.py remove the redundant ims in model input
# standard library imports
from __future__ import print_function, division
import os
import argparse
from time import time
from typing import Optional, Tuple, List

# 3rd party imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler

# local imports (i.e. our own code)
import src.data_handlers.data_handlers
from src.modules.recognition import RecognitionModule
from src.data_loaders.data_loaders import DataLoaderTrain, DataLoaderTest


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def is_equal(label_gt, label_pred):
    compare = [1 if int(label_gt[i]) == int(label_pred[i]) else 0 for i in range(7)]
    return sum(compare)


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
        dataset=DataLoaderTest(split_file=test_dirs, img_size=img_size),
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

            if is_equal(label_pred, YI[0]) == 7:
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


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    img_size: Tuple[int, int],
    batch_size: int,
    split_files: List[str],
    use_gpu: Optional[bool] = False,
    character_criterion: torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    num_epochs: Optional[int] = 25,
    store_name: Optional[str] = "rpnet.pt",
):
    # initialize the dataloader
    dataloader = DataLoader(
        dataset=DataLoaderTrain(split_file=split_files, img_size=img_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # initialize the learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        loss_avg = []
        # set the model to training mode
        model.train(mode=True)

        start = time()

        for idx, (XI, Y, labels, ims) in enumerate(dataloader):
            if not len(XI) == batch_size:
                continue

            YI = [[int(elem) for elem in label.split("_")[:7]] for label in labels]
            Y = np.array([elem.numpy() for elem in Y]).T
            if use_gpu:
                x = XI.clone().detach().cuda()
                y = torch.tensor(data=Y, dtype=torch.float32, requires_grad=False).cuda()
            else:
                x = XI.clone().detach()
                y = torch.tensor(data=Y, dtype=torch.float32, requires_grad=False)
            # Forward pass: Compute predicted y by passing x to the model

            fps_pred, y_pred = model(x)

            print(f"fps_pred: {fps_pred}")
            print("y", y)

            print("y_pred", y_pred)
            print("YI", YI)

            # Compute and print loss
            loss = torch.tensor(data=[0.0]).cuda(
                device=torch.device("cuda") if torch.device.type == "cuda" else None
            )
            if use_gpu:
                loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
                loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])

                for j in range(7):
                    l = torch.tensor(
                        data=[elem[j] for elem in YI], dtype=torch.long
                    ).cuda()

                    loss += character_criterion(y_pred[j], l)
            else:
                loss += 0.8 * nn.L1Loss()(fps_pred[:][:2], y[:][:2])
                loss += 0.2 * nn.L1Loss()(fps_pred[:][2:], y[:][2:])

                for j in range(7):
                    l = torch.tensor(data=[elem[j] for elem in YI], dtype=torch.long)
                    loss += character_criterion(y_pred[j], l)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_avg.append(loss.item())

            if not idx % 50:
                print(
                    f"Epoch: {epoch + 1}/{num_epochs}, "
                    f"Iteration: {idx + 1}/{len(dataloader)}, "
                    f"Loss: {np.mean(loss_avg):.4f}, "
                    f"Time: {time() - start:.4f}"
                )

                loss_avg = []

        # TODO: fix evaluation
        # set the model to evaluation mode
        # model.eval()
        #
        # count, correct, error, precision, avg_time = evaluate(
        #     model=model, test_dirs=test_dirs, img_size=img_size, use_gpu=use_gpu
        # )

        # print(
        #     f"*** total {count} error {error} precision {precision} avgTime {avg_time}"
        # )

        path = f"{os.getenv('MODEL_DIR')}_{epoch}_{store_name}"

        print(f"Saving model to {path}")
        # TODO: change saving to model.module.state_dict() as discussed here:
        #  https://discuss.pytorch.org/t/how-to-load-dataparallel-model-which-trained-using-multiple-gpus/146005
        torch.save(obj=model.state_dict(), f=path)
        print("Model saved.")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", required=True, help="path to the input file")
    ap.add_argument("-n", "--epochs", default=100, help="epochs for train")
    ap.add_argument("-b", "--batch_size", default=5, help="batch size for train")
    # ap.add_argument("-se", "--start_epoch", required=True, help="start epoch for train")
    # ap.add_argument("-t", "--test", required=True, help="dirs for test")
    # ap.add_argument("-f", "--folder", required=True, help="folder to store model")
    args = vars(ap.parse_args())

    num_points = 4
    batch_size = 1
    train_dirs = ["train.txt"]
    test_dirs = ["test.txt"]
    img_size = (480, 480)

    epochs = int(args["epochs"])

    # instantiate model
    recognition_model = RecognitionModule(
        num_points=num_points,
        pretrained_model_path="pretrained_detection_module.pt",
        province_num=38,
        alphabet_num=25,
        alphabet_numbers_num=35,
    )
    if torch.cuda.is_available():
        recognition_model = torch.nn.DataParallel(
            recognition_model, device_ids=[0, 4, 5, 6]
        )
        recognition_model = recognition_model.cuda()

    print(get_n_params(recognition_model))

    # optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)

    # train the recognition model end to end
    trained_recognition_model = train_model(
        model=recognition_model,
        img_size=img_size,
        split_files=train_dirs,
        batch_size=batch_size,
        use_gpu=torch.cuda.is_available(),
        character_criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(recognition_model.parameters(), lr=0.001),
        num_epochs=epochs,
    )
    # evaluate(
    #     model=recognition_model,
    #     test_dirs=["test.txt"],
    #     img_size=img_size,
    #     use_gpu=torch.cuda.is_available(),
    #     standalone=False,
    #     model_path="rpnet.pt",
    # )

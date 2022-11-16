# standard library imports

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
    batch_size: int = 16,
    num_epochs: int = 25,
    use_gpu: bool = False,
    criterion=nn.L1Loss(),
    lr_scheduler_step_size: int = 5,
    lr_scheduler_gamma: float = 0.1,
):
    # initialize the learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma
    )
    for epoch in range(num_epochs):
        loss_avg = []
        model.train(True)

        # start = time()

        for i, train_data in enumerate(train_loader):
            XI, YI = train_data
            YI = np.array([elem.numpy() for elem in YI]).T
            if use_gpu:
                x = Variable(XI.cuda(0))
                y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
            else:
                x = Variable(XI)
                print(f"x: {x}")
                y = Variable(torch.FloatTensor(YI), requires_grad=False)
                print(f"y: {y}")
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)

            # Transposing tensors s.t. we can slice along the columns of the predictions and calculate the losses properly
            y_pred = y_pred.T
            y = y.T
            # print(f"Predictions: {y_pred}")
            # print(f"slice: {y_pred[:][:2]}")
            # print(f"slice2: {y_pred[:][2:]}")

            # Compute and print loss
            running_loss = 0.0
            print(f"length y_pred: {len(y_pred)}, batch_size: {batch_size}")

            # loss getting split up, 80% weight on the x, y coordinates, 20% on the width and height of the bounding box
            if len(y_pred[0]) == batch_size:
                if use_gpu:
                    loss1 = 0.8 * criterion.cuda()(y_pred[:][:2], y[:][:2])
                    loss2 = 0.2 * criterion.cuda()(y_pred[:][2:], y[:][2:])
                else:
                    loss1 = 0.8 * criterion(y_pred[:][:2], y[:][:2])
                    loss2 = 0.2 * criterion(y_pred[:][2:], y[:][2:])
                loss = loss1 + loss2
                running_loss += loss1.item() + loss2.item()
                # print(f"loss: {loss}, type: {type(loss)}, value: {loss.item()}, val: {loss.data[0]}")
                loss_avg.append(running_loss)
                print(f"loss: {running_loss}")

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # should be called after optimizer.step()
                # torch.save(model.state_dict(), storeName)
            # if i % 50 == 1:
            #     with open(args["writeFile"], "a") as outF:
            #         outF.write(
            #             "train %s images, use %s seconds, loss %s\n"
            #             % (
            #                 i * batchSize,
            #                 time() - start,
            #                 sum(lossAver[-50:]) / len(lossAver[-50:]),
            #             )
            #         )
        print("%s %s\n" % (epoch, sum(loss_avg) / len(loss_avg)))
        # with open(args["writeFile"], "a") as outF:
        #     outF.write(
        #         "Epoch: %s %s %s\n"
        #         % (epoch, sum(lossAver) / len(lossAver), time() - start)
        #     )
        # torch.save(model.state_dict(), storeName + str(epoch))

    print("Finished training")

    return model


if __name__ == "__main__":
    split_directories = ["train.txt"]

    batch_size = 16

    pretrain_loader = DataLoader(
        DataLoaderPreTrain(
            split_file=split_directories, img_size=(480, 480), test_mode=False
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    num_classes = 4

    detection_model = DetectionModule(num_classes)

    trained_model = pretrain_model(
        model=detection_model,
        train_loader=pretrain_loader,
        use_gpu=torch.cuda.is_available(),
        batch_size=batch_size,
        optimizer=optim.Adam(detection_model.parameters(), lr=0.01),
    )

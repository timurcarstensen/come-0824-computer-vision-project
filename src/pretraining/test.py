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


img_size = (480, 480)
split_directories = ["train.txt"]
data = DataLoaderPreTrain(split_file=split_directories, img_size=img_size)
print(data)
train_loader = DataLoader(data, batch_size=1, shuffle=True)
print(train_loader)


class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2),
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2),
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10,
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x


numClasses = 4
model_conv = wR2(numClasses)

use_gpu = torch.cuda.is_available()
batchSize = 1
criterion = nn.MSELoss()
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)
# optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)
epoch_start = 0


def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    inner_criterion = nn.L1Loss()
    for epoch in range(epoch_start, num_epochs):
        lossAver = []
        model.train(True)

        # start = time()

        for i, train_data in enumerate(train_loader):
            XI, YI = train_data
            # print('%s/%s %s' % (i, times, time()-start))
            YI = np.array([el.numpy() for el in YI]).T
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
            print(f"Predictions: {y_pred}")
            print(f"slice: {y_pred[:][:2]}")
            print(f"slice2: {y_pred[:][2:]}")

            # Compute and print loss
            running_loss = 0.0
            print(f"length y_pred: {len(y_pred)}, batch_size: {batchSize}")

            # loss getting split up, 80% weight on the x, y coordinates, 20% on the width and height of the bounding box
            if len(y_pred[0]) == batchSize:
                if use_gpu:
                    loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                    loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                else:
                    loss1 = 0.8 * inner_criterion(y_pred[:][:2], y[:][:2])
                    print(f"loss1: {loss1}")
                    loss = loss1
                    loss2 = 0.2 * inner_criterion(y_pred[:][2:], y[:][2:])
                    # print(f"loss2: {loss2}")
                    loss = loss1 + loss2
                    running_loss += loss1.item() + loss2.item()
                # print(f"loss: {loss}, type: {type(loss)}, value: {loss.item()}, val: {loss.data[0]}")
                lossAver.append(running_loss)
                print(f"loss: {running_loss}")

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lrScheduler.step()  # should be called after optimizer.step()
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
        print("%s %s\n" % (epoch, sum(lossAver) / len(lossAver)))
        # with open(args["writeFile"], "a") as outF:
        #     outF.write(
        #         "Epoch: %s %s %s\n"
        #         % (epoch, sum(lossAver) / len(lossAver), time() - start)
        #     )
        # torch.save(model.state_dict(), storeName + str(epoch))

    print("Finished training")

    return model


epochs = 2
model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)

# number_train_images = 2

# for i in range(number_train_images):
#     resizedImage, new_labels = data.get_item(i)
#     print(f"resized img: {resizedImage}")
#     print(f"label: {new_labels}")

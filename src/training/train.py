# Compared to fh0.py
# fh02.py remove the redundant ims in model input
# standard library imports
from __future__ import print_function, division
import os
import argparse
from time import time

# 3rd party imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.optim import lr_scheduler

# local imports (i.e. our own code)
from src.modules.recognition import RecognitionModule
from src.data_loaders.data_loaders import DataLoaderTrain, DataLoaderTest


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the input file")
ap.add_argument("-n", "--epochs", default=10000, help="epochs for train")
ap.add_argument("-b", "--batchsize", default=5, help="batch size for train")
ap.add_argument("-se", "--start_epoch", required=True, help="start epoch for train")
ap.add_argument("-t", "--test", required=True, help="dirs for test")
ap.add_argument("-r", "--resume", default="111", help="file for re-train")
ap.add_argument("-f", "--folder", required=True, help="folder to store model")
ap.add_argument("-w", "--writeFile", default="fh02.out", help="file for output")
args = vars(ap.parse_args())

wR2Path = "./wR2/wR2.pth2"
use_gpu = torch.cuda.is_available()
print(use_gpu)

num_classes = 7
num_points = 4
classify_num = 35
img_size = (480, 480)
# lpSize = (128, 64)
prov_num, alpha_num, ad_num = 38, 25, 35
batch_size = int(args["batchsize"]) if use_gpu else 2
train_dirs = args["images"].split(",")
test_dirs = args["test"].split(",")
model_folder = (
    str(args["folder"]) if str(args["folder"])[-1] == "/" else str(args["folder"]) + "/"
)
store_name = model_folder + "fh02.pth"
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

epochs = int(args["epochs"])
#   initialize the output file
if not os.path.isfile(args["writeFile"]):
    with open(args["writeFile"], "wb") as outF:
        pass


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


epoch_start = int(args["start_epoch"])
resume_file = str(args["resume"])
if not resume_file == "111":
    # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
    if not os.path.isfile(resume_file):
        print("fail to load existed model! Existing ...")
        exit(0)
    print("Load existed model! %s" % resume_file)
    model_conv = RecognitionModule(num_points=num_points)
    model_conv = torch.nn.DataParallel(
        model_conv, device_ids=range(torch.cuda.device_count())
    )
    model_conv.load_state_dict(torch.load(resume_file))
    model_conv = model_conv.cuda()
else:
    model_conv = RecognitionModule(num_points=num_points, pretrained_model_path=wR2Path)
    if use_gpu:
        model_conv = torch.nn.DataParallel(
            model_conv, device_ids=range(torch.cuda.device_count())
        )
        model_conv = model_conv.cuda()

print(model_conv)
print(get_n_params(model_conv))

criterion = nn.CrossEntropyLoss()
# optimizer_conv = optim.RMSprop(model_conv.parameters(), lr=0.01, momentum=0.9)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

dst = DataLoaderTrain(train_dirs, img_size)
train_loader = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=8)
scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)


def is_equal(label_gt, label_pred):
    compare = [1 if int(label_gt[i]) == int(label_pred[i]) else 0 for i in range(7)]
    return sum(compare)


def eval(model, test_dirs):
    count, error, correct = 0, 0, 0
    dst = DataLoaderTest(test_dirs, img_size)
    test_loader = DataLoader(dst, batch_size=1, shuffle=True, num_workers=8)
    start = time()
    for i, (XI, labels, ims) in enumerate(test_loader):
        count += 1
        YI = [[int(ee) for ee in el.split("_")[:7]] for el in labels]
        if use_gpu:
            x = Variable(XI.cuda(0))
        else:
            x = Variable(XI)
        # Forward pass: Compute predicted y by passing x to the model

        fps_pred, y_pred = model(x)

        output_y = [el.data.cpu().numpy().tolist() for el in y_pred]
        label_pred = [t[0].index(max(t[0])) for t in output_y]

        #   compare YI, outputY
        try:
            if is_equal(label_pred, YI[0]) == 7:
                correct += 1
            else:
                pass
        except:
            error += 1
    return count, correct, error, float(correct) / count, (time() - start) / count


def train_model(model, criterion, optimizer, num_epochs=25):
    # since = time.time()
    for epoch in range(epoch_start, num_epochs):
        loss_avg = []
        model.train(True)
        scheduler.step()
        start = time()

        for i, (XI, Y, labels, ims) in enumerate(train_loader):
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

            try:
                fps_pred, y_pred = model(x)
            except:
                continue

            # Compute and print loss
            loss = 0.0
            loss += 0.8 * nn.L1Loss().cuda()(fps_pred[:][:2], y[:][:2])
            loss += 0.2 * nn.L1Loss().cuda()(fps_pred[:][2:], y[:][2:])
            for j in range(7):
                l = Variable(torch.LongTensor([el[j] for el in YI]).cuda(0))
                loss += criterion(y_pred[j], l)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            try:
                loss_avg.append(loss.data[0])
            except:
                pass

            if i % 50 == 1:
                with open(args["writeFile"], "a") as outF:
                    outF.write(
                        "train %s images, use %s seconds, loss %s\n"
                        % (
                            i * batch_size,
                            time() - start,
                            sum(loss_avg) / len(loss_avg)
                            if len(loss_avg) > 0
                            else "NoLoss",
                        )
                    )
                torch.save(model.state_dict(), store_name)
        print("%s %s %s\n" % (epoch, sum(loss_avg) / len(loss_avg), time() - start))
        model.eval()
        count, correct, error, precision, avgTime = eval(model, test_dirs)
        with open(args["writeFile"], "a") as outF:
            outF.write(
                "%s %s %s\n" % (epoch, sum(loss_avg) / len(loss_avg), time() - start)
            )
            outF.write(
                "*** total %s error %s precision %s avgTime %s\n"
                % (count, error, precision, avgTime)
            )
        torch.save(model.state_dict(), store_name + str(epoch))
        # save trained model to file
    return model


model_conv = train_model(model_conv, criterion, optimizer_conv, num_epochs=epochs)

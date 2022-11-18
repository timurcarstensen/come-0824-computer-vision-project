# standard library imports
import argparse
import os
from time import time

# 3rd party imports
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

# local imports (i.e. our own code)
from src.data_loaders.data_loaders import DataLoaderTest
from src.modules.roi_pooling import roi_pooling_ims
import src.data_handlers.data_handlers
from src.modules.recognition import RecognitionModule

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input",
    default=["test.txt"],
    required=False,
    help="path to the input folder",
)
# ap.add_argument("-m", "--model", required=True, help="path to the model file")
ap.add_argument("-s", "--store", required=False, help="path to the store folder")
args = vars(ap.parse_args())

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
use_gpu = torch.cuda.is_available()
print(use_gpu)

numClasses = 4
num_points = 4
img_size = (480, 480)
batch_size = 8 if use_gpu else 8
resume_file = f"{os.getenv('MODEL_DIR')}rpnet.pt"

provNum, alphaNum, adNum = 38, 25, 35


def is_equal(label_gt, label_p):
    return sum([1 if int(label_gt[i]) == int(label_p[i]) else 0 for i in range(7)])


model_conv = RecognitionModule(num_points=num_points)
if torch.cuda.is_available():
    model_conv = torch.nn.DataParallel(
        model_conv, device_ids=range(torch.cuda.device_count())
    )

a = torch.load(resume_file, map_location=torch.device("cpu"))

model_conv.load_state_dict(torch.load(resume_file, map_location=torch.device("cpu")))

# model_conv = model_conv.cuda()

model_conv.eval()

# efficiency evaluation
# dst = imgDataLoader([args["input"]], imgSize)
# trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)
#
# start = time()
# for i, (XI) in enumerate(trainloader):
#     x = Variable(XI.cuda(0))
#     y_pred = model_conv(x)
#     outputY = y_pred.data.cpu().numpy()
#     #   assert len(outputY) == batchSize
# print("detect efficiency %s seconds" %(time() - start))


count, correct, error, six_correct = 0, 0, 0, 0

sFolder = str(args["store"])
sFolder = sFolder if sFolder[-1] == "/" else sFolder + "/"
if not path.isdir(sFolder):
    mkdir(sFolder)

dst = DataLoaderTest(split_file=args["input"].split(","), img_size=img_size)
train_loader = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=1)
with open("fh0Eval", "wb") as outF:
    pass

start = time()
for i, (XI, labels, ims) in enumerate(train_loader):
    count += 1
    YI = [[int(ee) for ee in label.split("_")[:7]] for label in labels]
    if use_gpu:
        x = (
            XI.clone()
            .detach()
            .cuda(device=torch.device("cuda") if torch.device.type == "cuda" else None)
        )
    else:
        x = XI.clone().detach()

    # Forward pass: Compute predicted y by passing x to the model

    fps_pred, y_pred = model_conv(x)

    output_y = [elem.data.cpu().numpy().tolist() for elem in y_pred]
    label_pred = [t[0].index(max(t[0])) for t in output_y]

    #   compare YI, outputY
    # try:
    if is_equal(label_pred, YI[0]) == 7:
        correct += 1
        six_correct += 1
    else:
        six_correct += 1 if is_equal(label_pred, YI[0]) == 6 else 0

    if count % 50 == 0:
        print(
            "total %s correct %s error %s precision %s six %s avg_time %s"
            % (
                count,
                correct,
                error,
                float(correct) / count,
                float(six_correct) / count,
                (time() - start) / count,
            )
        )
with open("fh0Eval", "a") as outF:
    outF.write(
        "total %s correct %s error %s precision %s avg_time %s"
        % (count, correct, error, float(correct) / count, (time() - start) / count)
    )

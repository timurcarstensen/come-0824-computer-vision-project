# TODO: Too many bugs here, visualisation is not working on the server

# import cv2
import sys

import cv2
import src.utils.utils
from src.utils.datasets import PretrainDataset

provNum, alphaNum, adNum = 38, 25, 35
provinces = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]
alphabets = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]
ads = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "O",
]

ims = PretrainDataset(split_file=["train.txt"])
imss, (cx, cy, w, h) = ims.__getitem__(0)
# sys.exit()
# img = cv2.imread(imss)
cv2.imshow('image', imss)
sys.exit()
left_up = [(cx - w / 2) * img.shape[1], (cy - h / 2) * img.shape[0]]
right_down = [(cx + w / 2) * img.shape[1], (cy + h / 2) * img.shape[0]]
cv2.rectangle(
    img,
    (int(left_up[0]), int(left_up[1])),
    (int(right_down[0]), int(right_down[1])),
    (0, 0, 255),
    2,
)
#   The first character is Chinese character, can not be printed normally, thus is omitted.
labelPred = [0, 0, 0, 0, 0, 0, 0]
lpn = (
        alphabets[labelPred[1]]
        + ads[labelPred[2]]
        + ads[labelPred[3]]
        + ads[labelPred[4]]
        + ads[labelPred[5]]
        + ads[labelPred[6]]
)
cv2.putText(
    img, lpn, (int(left_up[0]), int(left_up[1]) - 20), cv2.FONT_ITALIC, 2, (0, 0, 255)
)
cv2.imwrite(imss, img)

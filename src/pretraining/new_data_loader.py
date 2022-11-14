from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imutils import paths
import cv2 as cv
import numpy as np
import pathlib
import os

data_dir =f"{pathlib.Path(__file__).parent.parent.parent.resolve()}/resources/data/" 

image_directory = data_dir + "test_images/splits/"

class DataLoaderPreTrain(Dataset):
    """used for pre training"""
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = [image_directory + elem for elem in img_dir]
        print(f"Image dir: {self.img_dir}")
        # print(f"type Image dir: {type(self.img_dir)}")
        self.img_paths = []
        # print(f"Length img dir: {len(img_dir)}")
        for i in range(len(self.img_dir)):
            with open(self.img_dir[i]) as f:
                lines = f.read().splitlines()
            for line in lines:
                self.img_paths.append(f"{data_dir}test_images/{line}")
        print(f"Image paths: {self.img_paths}")
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv.imread(img_name)
        resizedImage = cv.resize(img, self.img_size)
        resizedImage = np.reshape(
            resizedImage,
            (resizedImage.shape[2], resizedImage.shape[0], resizedImage.shape[1]),
        )

        iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
        [leftUp, rightDown] = [
            [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
        ]
        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [
            (leftUp[0] + rightDown[0]) / (2 * ori_w),
            (leftUp[1] + rightDown[1]) / (2 * ori_h),
            (rightDown[0] - leftUp[0]) / ori_w,
            (rightDown[1] - leftUp[1]) / ori_h,
        ]

        resizedImage = resizedImage.astype("float32")
        resizedImage /= 255.0

        return resizedImage, new_labels



class DataLoaderTrain(Dataset):
    """used for training data in rp net"""
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = [image_directory + elem for elem in img_dir]
        self.img_paths = []
        for i in range(len(self.img_dir)):
            with open(self.img_dir[i]) as f:
                lines = f.read().splitlines()
            for line in lines:
                self.img_paths.append(f"{data_dir}test_images/{line}")
        print(f"Image paths: {self.img_paths}")
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv.imread(img_name)
        resizedImage = cv.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype("float32")
        resizedImage /= 255.0
        lbl = img_name.split("/")[-1].rsplit(".", 1)[0].split("-")[-3]

        iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
        [leftUp, rightDown] = [
            [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
        ]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        new_labels = [
            (leftUp[0] + rightDown[0]) / (2 * ori_w),
            (leftUp[1] + rightDown[1]) / (2 * ori_h),
            (rightDown[0] - leftUp[0]) / ori_w,
            (rightDown[1] - leftUp[1]) / ori_h,
        ]

        return resizedImage, new_labels, lbl, 
        


class DataLoaderTest(Dataset):
    """used for testing data in rp net"""
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = [image_directory + elem for elem in img_dir]
        self.img_paths = []
        for i in range(len(self.img_dir)):
            with open(self.img_dir[i]) as f:
                lines = f.read().splitlines()
            for line in lines:
                self.img_paths.append(f"{data_dir}test_images/{line}")
        print(f"Image paths: {self.img_paths}")
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv.imread(img_name)
        resizedImage = cv.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype("float32")
        resizedImage /= 255.0
        lbl = img_name.split("/")[-1].split(".")[0].split("-")[-3]

        return resizedImage, lbl, img_name

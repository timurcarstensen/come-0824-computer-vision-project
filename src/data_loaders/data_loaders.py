# standard library imports
import pathlib
import os
from typing import List, Tuple, Union

# third party imports
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from imutils import paths
import cv2 as cv
import numpy as np

# local imports (i.e. our own code)


class DataLoaderPreTrain(Dataset):
    def __init__(
        self,
        split_file: Union[str, List[str]],
        img_size: Tuple,
        is_transform: bool = None,
        test_mode: bool = False,
    ):
        """
        Initialises the Pretraining Dataloader
        :param split_file: The file containing the list of images to be used for training
        :param img_size: desired size of the images (width, height)
        :param is_transform: @Lukas: was soll hier hin?
        :param test_mode: if true, use the test_images dir; else, use the full dataset
        """

        dataset_dir = "test_images" if test_mode else "CCPD2019"

        self.img_dir = [
            os.getenv("DATA_DIR") + f"{dataset_dir}/splits/" + elem for elem in split_file
        ]
        print(f"Image dir: {self.img_dir}")
        # print(f"type Image dir: {type(self.img_dir)}")
        self.img_paths = []
        # print(f"Length img dir: {len(img_dir)}")
        for i in range(len(self.img_dir)):
            # file = open(img_dir[i])
            # file_list = file.readlines()
            with open(self.img_dir[i]) as f:
                lines = f.read().splitlines()
            for line in lines:
                self.img_paths.append(f"{os.getenv('DATA_DIR')}{dataset_dir}/{line}")
        print(f"Image paths: {self.img_paths}")
        self.img_size = img_size
        # print(f"size: {self.img_size}")
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv.imread(img_name)
        resized_image = cv.resize(img, self.img_size)
        resized_image = np.reshape(
            resized_image,
            (resized_image.shape[2], resized_image.shape[0], resized_image.shape[1]),
        )

        iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
        [left_up, right_down] = [
            [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
        ]

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        new_labels = [
            (left_up[0] + right_down[0]) / (2 * ori_w),
            (left_up[1] + right_down[1]) / (2 * ori_h),
            (right_down[0] - left_up[0]) / ori_w,
            (right_down[1] - left_up[1]) / ori_h,
        ]

        resized_image = resized_image.astype("float32")
        resized_image /= 255.0

        return resized_image, new_labels

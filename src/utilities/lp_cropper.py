import os
from typing import List, Tuple

# third party imports
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from torchvision.transforms import Resize

s = "05-16_53-194&442_482&587-421&521_194&587_255&508_482&442-0_0_4_30_27_25_25-98-156.jpg"


def get_box(img, img_name: str):
    iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
    [left_up, right_down] = [
        [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
    ]
    print(left_up)
    print(right_down)

    box_img = img[left_up[1]:right_down[1], left_up[0]:right_down[0]]
    # resize = Resize((180, 224))
    #cv.imshow("box", box_img)
    #cv.waitKey(0)
    box_img = cv.resize(box_img, (360, 224*2))
    return box_img


def crop_image(image: np.ndarray, box_string: str, crop_size: Tuple[int, int]) -> np.ndarray:
    """Crop image to the specified size.

    Args:
        image (np.ndarray): image to crop
        crop_size (Tuple[int, int]): size to crop to

    Returns:
        np.ndarray: cropped image
    """
    height, width = image.shape[:2]
    new_height, new_width = crop_size
    top = (height - new_height) // 2
    left = (width - new_width) // 2
    bottom = top + new_height
    right = left + new_width
    return image[top:bottom, left:right]


def get_bounds(img_name):
    iname = img_name.rsplit("/", 1)[-1].rsplit(".", 1)[0].split("-")
    [left_up, right_down] = [
        [int(eel) for eel in el.split("&")] for el in iname[2].split("_")
    ]
    return left_up, right_down


def get_average_box_dimensions():
    file_directory = "/mnt/c/Users/tobis/IdeaProjects/come-0824-computer-vision-project/resources/data/CCPD2019/ccpd_tilt"
    # get all txt file names in the directory
    file_names = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if f.endswith(".jpg")]
    height = []
    width = []
    max_height = 0
    max_width = 0
    for file_name in file_names:
        left_up, right_down = get_bounds(file_name)
        height.append(right_down[1] - left_up[1])
        width.append(right_down[0] - left_up[0])
        max_height = max(max_height, right_down[1] - left_up[1])
        max_width = max(max_width, right_down[0] - left_up[0])

    # return average of left_ups and right_downs
    return np.mean(height, axis=0), np.mean(width, axis=0), max_height, max_width


file_directory = "/mnt/c/Users/tobis/IdeaProjects/come-0824-computer-vision-project/resources/data/CCPD2019/ccpd_tilt"
img = cv.imread(
    "/mnt/c/Users/tobis/IdeaProjects/come-0824-computer-vision-project/resources/data/CCPD2019/ccpd_tilt/05-16_53-194&442_482&587-421&521_194&587_255&508_482&442-0_0_4_30_27_25_25-98-156.jpg")
# show image
# cv.imshow("image", img)
# cv.waitKey(0)

a = get_box(img, s)
# print(a)
cv.imshow("a", a)
cv.waitKey(0)

# print(get_average_box_dimensions())

# This file ensures that the dataset is downloaded. Import this file at the top of your code
# to ensure that the dataset is always located in resources/data/

# standard library imports
import os
import pathlib
import tarfile

# 3rd party imports
import gdown

# setting up the paths
# setting the path to the data directory
os.environ[
    "DATA_DIR"
] = f"{pathlib.Path(__file__).parent.parent.parent.resolve()}/resources/data/"

# setting the path to the model weights directory
os.environ["LOG_DIR"] = f"{pathlib.Path(__file__).parent.parent.resolve()}/logs/"

DATASET_GDRIVE_ID = "1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"

dataset_detected = False

if (
    os.path.isfile("/work/ines-tp2022/topics-in-cv/dataset/CCPD2019.tar.xz")
    or os.path.isfile(f"{os.getenv('DATA_DIR')}CCPD2019.tar.xz")
    or os.path.isdir(f"{os.getenv('DATA_DIR')}CCPD2019")
):
    dataset_detected = True

if not dataset_detected:
    print("Dataset not found, downloading...")

    output = (
        "/work/ines-tp2022/topics-in-cv/dataset/CCPD2019.tar.xz"
        if os.path.isdir("/work/ines-tp2022/topics-in-cv/dataset/")
        else f"{os.getenv('DATA_DIR')}/CCPD2019.tar.xz"
    )

    print(f"Downloading to: {output}")

    gdown.download(
        id=DATASET_GDRIVE_ID,
        output=output,
    )

# check if the dataset is already extracted
if not os.path.isdir(f"{os.getenv('DATA_DIR')}/CCPD2019"):
    print("Extracting dataset...")

    tar = tarfile.open(
        "/work/ines-tp2022/topics-in-cv/dataset/CCPD2019.tar.xz"
        if os.path.isfile("/work/ines-tp2022/topics-in-cv/dataset/CCPD2019.tar.xz")
        else f"{os.getenv('DATA_DIR')}/CCPD2019.tar.xz"
    )

    tar.extractall(path=f"{os.getenv('DATA_DIR')}/")
    tar.close()

    print("Done extracting dataset.")

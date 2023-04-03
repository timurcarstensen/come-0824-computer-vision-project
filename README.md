# Topics in CV Project

## License Plate Detection and Recognition

This repository contains the code for the project "License Plate Detection and Recognition" which was part of the class Topics in Computer Vision in the fall term of 2022 at Kyungpook National University.

License plate (LP) detection is one popular research topic in the field of Computer Vision. In this report we make use of the CCPD dataset by [Xu et al.](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf). The dataset contains 250.000 unique images of cars together with annotations for the license plate number and the coordinates of a bounding box of the respective license plate. 

### Abstract

We investigate the use of various neural network architectures for license plate detection and recognition using the CCPD dataset. We first reproduce the results of an existing LP detection model, called RPnet, which is an end-to-end system that detects the bounding box of a LP in an image and recognizes the characters of the LP. We then explore the use of alternative neural network architectures, such as a Vision Transformer and a ResNet50 model, for LP recognition. In addition, we also try replacing the detection module of the RPnet model with a Vision Transformer to predict the LP characters. The results of these different approaches are evaluated and discussed in the [report](https://github.com/timurcarstensen/come-0824-computer-vision-project/blob/master/Report.pdf). Our best performing model uses a detection module with a ResNet50 as a backbone for the LP detection and a scaled bounding box prediction loss in the recognition module to accelerate learning. We match the performance of RPnet using fewer computational resources.

More information regarding implementation and evaluation can be found in the [Report](https://github.com/timurcarstensen/come-0824-computer-vision-project/blob/master/Report.pdf).

### Environment Setup

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   or [Anaconda](https://www.anaconda.com/products/distribution)
2. Navigate to the project directory and run the following command to create a new conda environment

```
conda create -n cv-project python=3.10
```

3. Activate the environment

```
conda activate cv-project
```

4. Install the requirements

```   
pip install -r requirements.txt
```

5. Navigate to src/utils and create a file named **wandb_key_file** and place your wandb API key in it. This will allow
   you to track your experiments on [wandb](https://wandb.ai/)

```
cd src/utils
touch wandb_key_file
nano wandb_key_file
Copy and paste your wandb API key in the file
```

6. Weights & Biases Setup and Usage (**following the example in src/train.py**):
    1. In any training setup, you can specify the WandbLogger as the logger parameter of the Pytorch Lightning Trainer
        1. group_name, project, entity: set to default parameters for now
        4. save_dir: The directory where the model checkpoints will be saved (do not modify)
        5. log_model: logs the final model parameters to wandb (very useful, should also not modify)


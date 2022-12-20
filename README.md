# Topics in CV Project

## License Plate Detection and Recognition

This repository contains the code for the project "License Plate Detection and Recognition" which was part of the class
Topics in Computer Vision in the fall term of 2022 at Kyungpook National University.

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


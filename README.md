# Topics in CV Project
## License Plate Detection and Recognition using Vision Transformers

### Environment Setup
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
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

5. Navigate to src/utils and create a file named **wandb_key_file** and place your wandb API key in it. This will allow you to track your experiments on [wandb](https://wandb.ai/)
```
cd src/utils
touch wandb_key_file
nano wandb_key_file
Copy and paste your wandb API key in the file
```

6. Weights & Biases Setup and Usage (**following the example in src/train.py**):
   1. In any training setup, you can specify the WandbLogger as the logger parameter of the Pytorch Lightning Trainer
      1. entity: The name of the project team (for this purpose we are using "mtp-ai-board-game-engine")
      2. project: The name of the project (for this purpose we are using "cv-project")
      3. group: name of the experiment group (i.e. "testing-iuo-metrics") -> this groups **all** the runs that you start using the group name and aggregates the metrics in wandb
      4. save_dir: The directory where the model checkpoints will be saved (do not modify)
      5. log_model: logs the final model parameters to wandb (very useful, should also not modify)

### Server Setup
1. Connect to the University of Mannheim VPN using Cisco AnyConnect
2. Follow the [following tutorial](https://yangkky.github.io/2019/11/13/pycharm.html) to add a new remote interpreter to your IDE (the tutorial is using PyCharm)
3. Use the following settings: 
   1. *Host*: dws-student-01.informatik.uni-mannheim.de
   2. *Port*: 22
   3. *Username*: ines-tp2022
   4. *Password*: as provided in the group chat
   2. *Interpreter path*: /work/ines-tp2022/miniconda3/envs/cv-project/bin/python3.10
   4. *Sync directory*: /work/ines-tp2022/topics-in-cv/ide_sync_dir/yourfirstname
 


## Vision Transformers
Vision Transformers implementations can be found [here](https://github.com/lucidrains/vit-pytorch).

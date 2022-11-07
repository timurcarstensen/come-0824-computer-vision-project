# Topics in CV Project
## License Plate Detection and Recognition using Vision Transformers

### Environment Setup
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
2. Navigate to the project directory and run the following command to create a new conda environment
```
conda env create -f environment.yaml -n cv-project
```

3. Activate the environment
```
conda activate cv-project
```

4. Install pre-commit hooks
```
pre-commit install
```

### Server Setup

1. Follow the [following tutorial](https://yangkky.github.io/2019/11/13/pycharm.html) to add a new remote interpreter to your IDE (the tutorial is using PyCharm)
2. Use the following settings: 
   1. *Host*: dws-student-01.informatik.uni-mannheim.de
   2. *Port*: 22
   3. *Username*: ines-tp2022
   4. *Password*: as provided in the group chat
   2. *Interpreter path*: /work/ines-tp2022/miniconda3/envs/cv-project/bin/python3.10
   4. *Sync directory*: /work/ines-tp2022/topics-in-cv/ide_sync_dir/yourfirstname
 


## Vision Transformers
Vision Transformers implementations can be found [here](https://github.com/lucidrains/vit-pytorch).

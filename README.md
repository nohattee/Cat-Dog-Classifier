## Install Python dependencies

You can use conda to create environment

```sh
    conda create --name imageclassification python=3.6
```

And you need to activate it

```sh
    conda activate imageclassification
```

After that you can install package below with pip or conda install

1. Python (3.6)
2. Flask
3. Pytorch, torchvision (https://pytorch.org/)
4. imageio


## Checkpoints

-   You can download from my drive (https://bit.ly/34mN2Cx) and put it all to /server/checkpoints/ folder.


## Datasets

-   You can download from here https://www.kaggle.com/tongpython/cat-and-dog


## Getting Started

Download npm and nodejs (https://nodejs.org/)

In the root directory of the project...

1. Install node modules `yarn install` or `npm install`.
2. Start development server `yarn start` or `npm start`.


## Evaluation

If you want to compare the accuracy between models, you can run file `model.py`
```sh
    usage: model.py [-h] [--dataset DATASET] [--checkpoints CHECKPOINTS]

    optional arguments:
    -h, --help            show this help message and exit
    --dataset DATASET     Dataset test path to evaluation
    --checkpoints CHECKPOINTS
                            Path folder contains checkpoint
```
# *CortexNet*

This repo contains the *PyTorch* implementation of *CortexNet*.  
Check the [project website](https://engineering.purdue.edu/elab/CortexNet/) for further information.

## Project structure

The project consists of the following folders and files:

 - [`data/`](data): contains *Bash* scripts and a *Python* class definition inherent video data loading;
 - [`image-pretraining/`](image-pretraining/): hosts the code for pre-training TempoNet's discriminative branch;
 - [`model/`](model): stores several network architectures, including [*PredNet*](https://coxlab.github.io/prednet/), an additive feedback *Model01*, and a modulatory feedback *Model02* ([*CortexNet*](https://engineering.purdue.edu/elab/CortexNet/));
 - [`notebook/`](notebook): collection of *Jupyter Notebook*s for data exploration and results visualisation;
 - [`utils/`](utils): scripts for
   - (current or former) training error plotting,
   - experiments `diff`,
   - multi-node synchronisation,
   - generative predictions visualisation,
   - network architecture graphing;
 - `results@`: link to the location where experimental results will be saved within 3-digit folders;
 - [`new_experiment.sh*`](new_experiment.sh): creates a new experiment folder, updates `last@`, prints a memo about last used settings;
 - `last@`: symbolic link pointing to a new results sub-directory created by `new_experiment.sh`;
 - [`main.py`](main.py): training script for *CortexNet* in *MatchNet* or *TempoNet* configuration;

## Dependencies

 + [*scikit-video*](https://github.com/scikit-video/scikit-video): accessing images / videos

```bash
pip install sk-video
```

 + [*tqdm*](https://github.com/tqdm/tqdm): progress bar

```bash
conda config --add channels conda-forge
conda update --all
conda install tqdm
```

## IDE

This project has been realised with [*PyCharm*](https://www.jetbrains.com/pycharm/) by *JetBrains* and the [*Vim*](http://www.vim.org/) editor.
[*Grip*](https://github.com/joeyespo/grip) has been also fundamental for crafting decent documtation locally.

## Initialise environment

Once you've determined where you'd like to save your experimental results — let's call this directory `<my saving location>` — run the following commands from the project's root directory:

```bash
ln -s <my saving location> results  # replace <my saving location>
mkdir results/000 && touch results/000/train.log  # init. placeholder
ln -s results/000 last  # create pointer to the most recent result
```

## Setup new experiment

Ready to run your first experiment?
Type the following:

```bash
./new_experiment.sh
```

### GPU selection

Let's say your machine has `N` GPUs.
You can choose to use any of these, by specifying the index `n = 0, ..., N-1`.
Therefore, type `CUDA_VISIBLE_DEVICES=n` just before `python ...` in the following sections.

## Train *MatchNet*

 + Download *e-VDS35* (*e.g.* `e-VDS35-May17.tar`) from [here](https://engineering.purdue.edu/elab/eVDS/).
 + Use [`data/resize_and_split.sh`](data/resize_and_split.sh) to prepare your (video) data for training.
   It resizes videos present in folders of folders (*i.e.* directory of classes) and may split them into training and validation set.
   May also skip short videos and trim longer ones.
   Check [`data/README.md`](data/README.md#matchnet-mode) for more details.
 + Run the [`main.py`](main.py) script to start training.
   Use `-h` to print the command line interface (CLI) arguments help.

```bash
python -u main.py --mode MatchNet <CLI arguments> | tee last/train.log
```

## Train *TempoNet*

 + Download *e-VDS35* (*e.g.* `e-VDS35-May17.tar`) from [here](https://engineering.purdue.edu/elab/eVDS/).
 + Pre-train the forward branch (see [`image-pretraining/`](image-pretraining)) on an image data set (*e.g.* `33-image-set.tar` from [here](https://engineering.purdue.edu/elab/eVDS/));
 + Use [`data/resize_and_sample.sh`](data/resize_and_sample.sh) to prepare your (video) data for training.
   It resizes videos present in folders of folders (*i.e.* directory of classes) and samples them.
   Videos are then distributed across training and validation set.
   May also skip short videos and trim longer ones.
   Check [`data/README.md`](data/README.md#temponet-mode) for more details.
 + Run the [`main.py`](main.py) script to start training.
   Use `-h` to print the CLI arguments help.

```bash
python -u main.py --mode TempoNet --pre-trained <path> <CLI args> | tee last/train.log
```

## GPU selection

To run on a specific GPU, say `n`, type `CUDA_VISIBLE_DEVICES=n` just before `python ...`.

# *PyTorch* *MatchNet* implementation

This repo contains the *PyTorch* version of *MatchNet*.

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

## Train *MatchNet*

 + [`utils/resize_and_split.sh`](data/resize_and_split.sh): Prepares your (video) data for training.
   Resizes videos present in folders of folders (*i.e.* directory of classes) and may split them into training and validation set.
   May also skip short videos and trim longer ones.
 + [`main.py`](main.py): Script to start training.
   Use `-h` to print the command line arguments help.

```bash
python -u main.py <CLI arguments> | tee train.log
```

To run on a specific GPU, say `n`, type `CUDA_VISIBLE_DEVICES=n` just before `python ...`.

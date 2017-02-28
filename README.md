# *PyTorch* *MatchNet* implementation

This repo contains the *PyTorch* version of *MatchNet*.

## Dependencies

+ [scikit-video](https://github.com/scikit-video/scikit-video): accessing images/videos
```
pip install sk-video
```
+ [tqdm](https://github.com/tqdm/tqdm): progress bar
```
conda config --add channels conda-forge
conda update --all
conda install tqdm
```

## Train MatchNet

+ [resize_and_split.sh](data/resize_and_split.sh): Prepare your data (video) for training.
Resizes videos present in folders of folder and then splits them into training and validation set.
+ [main.py](main.py): Script to start training.

```
python main.py -data /data/folder/
```

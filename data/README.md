# Video data pre-processing

This folder contains the following scripts:

 - [`add_frame_numbering.sh*`](add_frame_numbering.sh): draws a huge number on each frame on a specific video;
 - [`resize_and_split.sh*`](resize_and_split.sh): see [below](#matchnet-mode);
 - [`resize_and_sample.sh*`](resize_and_sample.sh): see [below](#temponet-mode);
 - [`VideoFolder.py`](VideoFolder.py): *PyTorch* `data.Dataset`'s sub-class for video data loading.

## Remove white spaces from file names
White spaces and scripts are not good friends.
Replace all spaces with `_` in your source data set with the following command, where you have to replace `<root data dir>` with the correct location.

```bash
rename -n "s/ /_/g" <root data dir>/*/*  # for a dry run
rename "s/ /_/g" <root data dir>/*/*     # to rename the files in <root data dir>!!!
```

In *e-VDS35* we have already done this for you.
If you plan to use your own data, this step is fundamental!

## MatchNet mode
### Resize and *split* videos in train-val

In order to speed up data loading, we shall resize the video shortest side to, say, `256`.
To do so run the following script.

```bash
./resize_and_split.sh <source dir> <destination dir>
```

By default, the script will

 - skip videos shorter than `144` frames (`4.8`s)
 - trim videos longer than `654` frames (`21.8`s)
 - use the last `2`s for the validation split
 - resize the shortest side to `256`px

These options can be varied and turned off by changing the header of the script, which now looks something like this.

```bash
################################################################################
# SETTINGS #####################################################################
################################################################################
# comment the next line if you don't want to skip short videos
min_frames=144
# comment the next line if you don't want to limit the max length
max_frames=564
# set split to 0 (seconds) if no splitting is required
split=2
################################################################################
```

## TempoNet mode
### Resize and *sample* videos, then splits in train-val

In order to speed up data loading, we shall resize the video shortest side to, say, `256`.
To do so run the following script.

```bash
./resize_and_sample.sh <source dir> <destination dir>
```

By default, the script will

 - skip videos shorter than `144` frames (`4.8`s)
 - trim videos longer than `654` frames (`21.8`s)
 - perform `5` subsamples, use `4` for training, `1` for validation
 - resize the shortest side to `256`px

These options can be varied and turned off by changing the header of the script, which now looks something like this.

```bash
################################################################################
# SETTINGS #####################################################################
################################################################################
# comment the next line if you don't want to skip short videos
min_frames=144
# comment the next line if you don't want to limit the max length
max_frames=564
# set sampling interval: k - 1 train, 1 val
k=5
################################################################################
```

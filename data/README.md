# Video data pre-processing

This folder contains the following scripts:

 - [`add_frame_numbering.sh*`](add_frame_numbering.sh): draws a huge number on each frame on a specific video;
 - [`dump_data_set.sh*`](dump_data_set.sh): get images from videos for ["traditional training"](https://github.com/pytorch/examples/tree/master/imagenet);
 - [`objectify.sh*`](objectify.sh): convert sampled-data from video-indexed to object-indexed;
 - [`resize_and_split.sh*`](resize_and_split.sh): see [below](#matchnet-mode);
 - [`resize_and_sample.sh*`](resize_and_sample.sh): see [below](#temponet-mode);
 - [`sample_video.sh*`](sample_video.sh): sample the `<src video>` into `k` time-subsampled `<dst prefix>-` videos;
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

### File system

Our input `data_set` looks somehow like this.

```bash
data_set
├── barcode
│   ├── 20160613_140057.mp4
│   ├── 20160613_140115.mp4
│   ├── 20160613_140138.mp4
│   ├── 20160721_023437.mp4
├── bicycle
│   ├── 0707_2_(2).mov
│   ├── 0707_2_(4).mov
```

Running `./resize_and_sample.sh data_set/ processed-data` yields

```bash
processed-data/
├── train
│   ├── barcode
│   │   ├── 20160613_140057.mp4
│   │   ├── 20160613_140115.mp4
│   │   ├── 20160613_140138.mp4
│   │   ├── 20160721_023437.mp4
│   ├── bicycle
│   │   ├── 0707_2_(2).mp4
│   │   ├── 0707_2_(4).mp4
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

The output directory will contain **as many folders as the total number of videos**.
Each folder will contain the individual splits.

### From video-index- to class-major data organisation

If you would like to train against object classes instead of video indices (like explained in the paper), you also need to run

```bash
./objectify.sh <source dir> <destination dir>
```

This script generates a new directory containing **as many folders as classes**, filled with symbolic links from the source directory.
You can use it for both videos and dumped-videos (images) data sets.

### File system

Running `./resize_and_sample.sh data_set/ sampled-data` yields

```bash
sampled-data/
├── train
│   ├── barcode-20160613_140057
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   ├── 3.mp4
│   │   └── 4.mp4
│   ├── barcode-20160613_140115
│   │   ├── 1.mp4
│   │   ├── 2.mp4
```

We can "objectify" the structure with `./objectify.sh sampled-data/ object-sampled-data` and get

```bash
object-sampled-data/
├── train
│   ├── barcode
│   │   ├── 20160613_140057-1.mp4 -> ../../../sampled-data/train/barcode-20160613_140057/1.mp4
│   │   ├── 20160613_140057-2.mp4 -> ../../../sampled-data/train/barcode-20160613_140057/2.mp4
│   │   ├── 20160613_140057-3.mp4 -> ../../../sampled-data/train/barcode-20160613_140057/3.mp4
│   │   ├── 20160613_140057-4.mp4 -> ../../../sampled-data/train/barcode-20160613_140057/4.mp4
│   │   ├── 20160613_140115-1.mp4 -> ../../../sampled-data/train/barcode-20160613_140115/1.mp4
│   │   ├── 20160613_140115-2.mp4 -> ../../../sampled-data/train/barcode-20160613_140115/2.mp4
│   │   ├── 20160613_140115-3.mp4 -> ../../../sampled-data/train/barcode-20160613_140115/3.mp4
```

To train the discriminative feed-forward branch we need to dump our `sampled-data` with `./dump_data_set.sh sampled-data/ dumped-sampled-data`.
The file system will look like this

```bash
dumped-sampled-data/
├── train
│   ├── barcode-20160613_140057
│   │   ├── 1001.png
│   │   ├── 1002.png
│   │   ├── 1003.png
│   │   ├── 1004.png
│   │   ├── 1005.png
│   │   ├── 1006.png
│   │   ├── 1007.png
```

where the "training class" correspond to the **video name**.
If we wish to train against **object classes**, then we can run `./objectify.sh dumped-sampled-data/ dumped-object-sampled-data` and get the following

```bash
dumped-object-sampled-data/
├── train
│   ├── barcode
│   │   ├── 20160613_140057-1001.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1001.png
│   │   ├── 20160613_140057-1002.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1002.png
│   │   ├── 20160613_140057-1003.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1003.png
│   │   ├── 20160613_140057-1004.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1004.png
│   │   ├── 20160613_140057-1005.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1005.png
│   │   ├── 20160613_140057-1006.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1006.png
│   │   ├── 20160613_140057-1007.png -> ../../../dumped-sampled-data/train/barcode-20160613_140057/1007.png
```

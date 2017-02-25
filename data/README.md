# Video data pre-processing

## Remove white spaces from file names
White spaces and scripts are not good friends.
Replace all spaces with `_` in your source data set with the following command, where you have to replace `<root data dir>` with the correct location.

```bash
rename -n "s/ /_/g" <root data dir>/*/*  # for a dry run
rename "s/ /_/g" <root data dir>/*/*     # to rename the files in $src_dir!!!
```

## Resize videos (and split in train-val)
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

# Video data pre-processing

White spaces and scripts are not good friends.
Replace all spaces with `_` in your source data set with the following command, where you have to replace `<root data dir>` with the correct location.

```bash
rename -n "s/ /_/g" <root data dir>/*/*  # for a dry run
rename "s/ /_/g" <root data dir>/*/*     # to rename the files in $src_dir!!!
```

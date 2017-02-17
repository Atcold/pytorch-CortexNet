# Resize video data set minor side to 256

# assert existence of source directory
src_dir='data_set'
if [ ! -d $src_dir ]; then
    echo "Source directory/link "$src_dir" is missing. Exiting."
    exit 1
fi

# replace spaces in file names with underscores on a Linux machine
# rename -n "s/ /_/g" data_set/*/*  # for a dry run
# rename "s/ /_/g" data_set/*/*  # to rename the files in $src_dir!!!

# assert existence of destination directory
dst_dir='256min_data_set'
if [ -d $dst_dir ]; then
    echo 'Destination directory already existent. Exiting.'
    exit 1
fi
# create destination directory
mkdir $dst_dir

for class in $(ls $src_dir); do

    echo "Processing class \"$class\""
    src_class_dir="$src_dir/$class"
    dst_class_dir="$dst_dir/$class"
    mkdir $dst_class_dir

    for video in $(ls $src_class_dir); do
        printf " > rescaling \"$video\""
        src_video_path="$src_class_dir/$video"
        dst_video_path="$dst_class_dir/${video%.*}.mp4"  # replace extension

        # disable audio recording
        # scale the min side to 256, the other to the even number closest
        # to keep the same aspect ratio
        ffmpeg \
        -i "$src_video_path" \
        -an \
        -filter:v "scale=w=2*trunc(128*max(1\, iw/ih)):h=2*trunc(128*max(1\, ih/iw))" \
        -loglevel quiet \
        "$dst_video_path"

        # check the output stream
        printf ' --> '
        ffprobe $dst_video_path 2>&1 | grep Stream | cut -d, -f3 | cut -d' ' -f2
    done

done


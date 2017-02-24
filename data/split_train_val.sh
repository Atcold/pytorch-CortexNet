# Split the whole data set into validation (Ns end cut) and train sets

N=3  # number of seconds to extract for validation

# assert existence of source directory
src_dir=${1%/*}  # remove trailing /, if present
if [ ! -d $src_dir ]; then
    echo "Source directory/link \"$src_dir\" is missing. Exiting."
    exit 1
fi

# assert existence of destination directory
dst_dir="${src_dir}_split"
if [ -d $dst_dir ]; then
    echo "Destination directory \"$dst_dir\" already existent. Exiting."
    exit 1
fi

train_dir="$dst_dir/train"
val_dir="$dst_dir/val"

mkdir $dst_dir
mkdir $train_dir
mkdir $val_dir

for class in $(ls $src_dir); do

    echo "Processing class \"$class\""
    src_class_dir="$src_dir/$class"
    train_class_dir="$dst_dir/train/$class"
    val_class_dir="$dst_dir/val/$class"
    mkdir $train_class_dir
    mkdir $val_class_dir

    for video in $(ls $src_class_dir); do
        echo " > splitting \"$video\""
        src_video_path="$src_class_dir/$video"
        # get src_video duration
        tot_t=$(ffprobe -loglevel error -show_streams $src_video_path | \
        awk '/duration=/{sub(/duration=/,""); print}')
        # compute duration in seconds
        end_t=$(awk "BEGIN{printf (\"%f\",$tot_t-$N)}")
        # compute duration in ffmpeg format
        ffmpeg_end_t=$(awk "BEGIN{printf (\"%02d:%02d:%02.4f\",$end_t/3600,($end_t%3600)/60,($end_t%60))}")

        # get my src_video_path
        # use it until ffmpeg_end_t
        # copy the stream over
        # be quiet (show errors only)
        # save at val_video_path
        train_video_path="$train_class_dir/$video"
        ffmpeg \
            -i $src_video_path \
            -to $ffmpeg_end_t \
            -codec copy \
            -loglevel error \
            $train_video_path

        # start at Ns from the end
        # of my src_video_path
        # copy the stream over (do not re-encode)
        # be quiet (show errors only)
        # save at val_video_path
        val_video_path="$val_class_dir/$video"
        ffmpeg \
            -sseof -$N \
            -i $src_video_path \
            -codec copy \
            -loglevel error \
            $val_video_path

        # check the output stream
        #printf ' --> '
        #ffprobe $dst_video_path 2>&1 | grep Stream | cut -d, -f3 | cut -d' ' -f2
    done
done

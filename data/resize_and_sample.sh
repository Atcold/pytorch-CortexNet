################################################################################
# Pre-process video data
################################################################################
# Alfredo Canziani, Apr 17
################################################################################

# Pre-process video data
#
#  - resize video data minor side to specific size
#  - sample frames and split into train a val sets
#  - skip "too short" videos
#  - limit max length
#
# Run as:
# ./resize_and_split.sh src_path/ dst_path/
#
# It's better to perform the resizing and the sampling together since
# re-encoding is necessary when a temporal sampling is performed.
# Skipping and clipping max length are also easily achievable at this point in
# time.

# current object video data set
# 95% interval: [144, 564] -> [4.8, 21.8] seconds
# mean number of frames: 354 -> 11.8 seconds

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

# some colours
r='\033[0;31m'  # red
g='\033[0;32m'  # green
b='\033[0;34m'  # blue
n='\033[0m'     # none

# check min_frames setting
printf " - "
if [ -n "$min_frames" ]; then
    echo -e "Skipping videos with < $b$min_frames$n frames"
    skip_count=0
else
    echo "No skipping short vidos"
    min_frames=0
fi

# check max_frames setting
printf " - "
if [ -n "$max_frames" ]; then
    echo -e "Trimming videos with > $b$max_frames$n frames"
    trim_count=0
else
    echo "No trimming long vidos"
fi

# check split setting
printf " - "
echo -e "Sampling every $b$k$n frames"
kk=$(awk "BEGIN{print 1/$k}")

# assert existence of source directory
src_dir=${1%/*}  # remove trailing /, if present
if [ ! -d $src_dir ] || [ -z $src_dir ]; then
    echo -e "${r}Source directory/link \"$src_dir\" is missing. Exiting.${n}"
    exit 1
fi
echo -e " - Source directory/link set to \"$b$src_dir$n\""

# assert existence of destination directory
dst_dir="${2%/*}"
if [ -d $dst_dir ]; then
    echo -e "${r}Destination directory \"$dst_dir\" already existent." \
        "Exiting.${n}"
    exit 1
fi
echo -e " - Destination directory set to \"$b$dst_dir$n\""

# check if all is good
printf "Does it look fine? (${g}y${g}${n}/${r}n${n}) "
read ans
if [ $ans == 'n' ]; then
    echo -e "${r}Exiting.${n}"
    exit 0
fi

# for every class
for class in $(ls $src_dir); do

    printf "\nProcessing class \"$class\"\n"

    # define src
    src_class_dir="$src_dir/$class"

    # for each video in the class
    for video in $(ls $src_class_dir); do

        printf " > Loading video \"$video\". "

        # define src video path
        src_video_path="$src_class_dir/$video"

        # count the frames
        frames=$(ffprobe \
            -loglevel quiet \
            -show_streams \
            -select_streams v \
            $src_video_path | awk \
            '/nb_frames=/{sub(/nb_frames=/,""); print}')

        # skip if too short
        if ((frames < min_frames)); then
            printf "Frames: $b$frames$n < $b$min_frames$n min frames. "
            echo -e "${r}Skipping.$n"
            ((skip_count++))
            continue
        fi

        # define and make dst and val dir
        dst_video_path="$dst_dir/train/$class-${video%.*}"
        val_video_path="$dst_dir/val/$class-${video%.*}"
        mkdir -p $dst_video_path
        mkdir -p $val_video_path

        # get src_video frame rate
        fps=$(ffprobe \
            -loglevel error \
            -show_streams \
            -select_streams v \
            $src_video_path | awk \
            '/avg_frame_rate=/{sub(/avg_frame_rate=/,""); print}')

        # get my src_video_path
        # discard audio stram
        # be quiet (show errors only)
        # use complex filter (1 input file, multiple output files)
        # rescale the video stram min side to 256
        # select frames using frame_number % k and send it to streams 1, ..., k
        # as long as frame_number < max_frames
        # use the input average frame rate as output fps
        # send each stream to a separate output file dst_video_path/{1..k-1}.mp4
        # and val_video_path/k.mp4
        printf "Rescaling and sampling"
        ffmpeg \
            -i $src_video_path \
            -an \
            -loglevel error \
            -filter_complex \
                "setpts=$kk*PTS, \
                scale=w=2*trunc(128*max(1\, iw/ih)):h=2*trunc(128*max(1\, ih/iw))[m]; \
                [m]select=n=$k:e=(mod(n\,$k)+1)*lt(n\,$max_frames) \
                $(for ((i=1; i<=$k; i++)); do
                    echo -n "[a$i]"
                done)" \
            $(for ((i=1; i<$k; i++)); do
                echo -n "-r $fps -map [a$i] $dst_video_path/$i.mp4 "
            done
            echo -n "-r $fps -map [a$k] $val_video_path/$k.mp4"
            )

        # check the output stream resolution
        printf ' --> '
        ffprobe \
            -loglevel quiet \
            -show_streams \
            "$dst_video_path/1.mp4" | awk \
            -F= \
            '/^width/{printf $2"x"}; /^height/{printf $2"."}'

        if ((frames > max_frames)); then
            echo -n " Trimming $frames --> $max_frames."
            ((trim_count++))
        fi

        # new line :)
        printf "\n"
    done
done

printf "\n---------------\n"
printf "Skipped $b%d$n videos\n" "$skip_count"
printf "Trimmed $b%d$n videos" "$trim_count"
printf "\n---------------\n\n"
echo -e "${r}Exiting.${n}"
exit 0

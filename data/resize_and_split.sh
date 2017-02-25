# Pre-process video data
#
#  - resize video data minor side to specific size
#  - split into train a val sets
#  - skip "too short" videos
#  - limit max length
#
# Run as:
# ./resize_and_split.sh src_path/ dst_path/
#
# It's better to perform the resizing and the splitting together since
# re-encoding is necessary when a temporal split is performed.
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
# set split to 0 (seconds) if no splitting is required
split=2
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
if [ $split != 0 ]; then
    echo -e "Using last $b$split$n seconds for validation"
    dst="train/"
else
    echo "No train-validation splitting will be performed"
    dst=""
fi

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

    # define src and dst dir, make dst dir
    src_class_dir="$src_dir/$class"
    dst_class_dir="$dst_dir/$dst$class"
    mkdir -p $dst_class_dir

    # if split > 0, deal with validation dir too
    if [ $split != 0 ]; then
        val_class_dir="$dst_dir/val/$class"
        mkdir -p $val_class_dir
    fi

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

        # get src_video duration
        tot_t=$(ffprobe \
            -loglevel quiet \
            -show_streams \
            -select_streams v \
            $src_video_path | awk \
            '/duration=/{sub(/duration=/,""); print}')

        # if there is a max_frames and we are over it, redefine tot_t
        if [ -n "$max_frames" ] && ((frames > max_frames)); then
            printf "Frames: $b$frames$n > $b$max_frames$n max frames. "
            printf "Trimming %.2fs" "$tot_t"
            fps=$(ffprobe \
                -loglevel error \
                -show_streams \
                -select_streams v \
                $src_video_path | awk \
                '/avg_frame_rate=/{sub(/avg_frame_rate=/,""); print}')
            tot_t=$(awk \
                "BEGIN{printf (\"%.4f\",$tot_t-($frames-$max_frames)/($fps))}")
            printf " --> %.2fs. " "$tot_t"
            ((trim_count++))
        fi

        # compute duration in seconds
        end_t=$(awk "BEGIN{printf (\"%f\",$tot_t-$split)}")

        # compute duration in ffmpeg format
        ffmpeg_end_t=$(awk \
            "BEGIN{printf (\"%02d:%02d:%02.4f\",$end_t/3600,($end_t%3600)/60,($end_t%60))}")

        # get my src_video_path
        # discard audio stram
        # use it until ffmpeg_end_t
        # rescale the video stram min side to 256
        # be quiet (show errors only)
        # save at dst_video_path
        printf "Rescaling"
        dst_video_path="$dst_class_dir/$video"
        ffmpeg \
            -i $src_video_path \
            -an \
            -to $ffmpeg_end_t \
            -filter:v "scale=w=2*trunc(128*max(1\, iw/ih)):h=2*trunc(128*max(1\, ih/iw))" \
            -loglevel error \
            $dst_video_path

        # check the output stream resolution
        printf ' --> '
        ffprobe \
            -loglevel quiet \
            -show_streams \
            $dst_video_path | awk \
            -F= \
            '/^width/{printf $2"x"}; /^height/{printf $2"."}'

        if [ $split != 0 ]; then
            # start at Ns from the end
            # of my src_video_path
            # discard audio stram
            # rescale the video stram min side to 256
            # be quiet (show errors only)
            # save at val_video_path
            printf " Splitting"
            val_video_path="$val_class_dir/$video"
            ffmpeg \
                -sseof -$split \
                -i $src_video_path \
                -an \
                -filter:v "scale=w=2*trunc(128*max(1\, iw/ih)):h=2*trunc(128*max(1\, ih/iw))" \
                -loglevel error \
                $val_video_path

            # print temporal split
            duration=$(awk \
            "BEGIN{printf (\"%02d:%02d:%02.4f\",$tot_t/3600,($tot_t%3600)/60,($tot_t%60))}")
            printf " 00:00:00 / $ffmpeg_end_t / $duration."
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

################################################################################
# Dump data set
################################################################################
# Alfredo Canziani, Apr 17
################################################################################
# Run as:
# ./dump_data_set.sh src_path/ dst_path/
################################################################################

# some colours
r='\033[0;31m'  # red
g='\033[0;32m'  # green
b='\033[0;34m'  # blue
n='\033[0m'     # none

# title
echo "Dumping video data set into separate frames"

# assert existence of source directory
src_dir=${1%/*}  # remove trailing /, if present
if [ ! -d $src_dir ] || [ -z $src_dir ]; then
    echo -e "${r}Source directory/link \"$src_dir\" is missing. Exiting.${n}"
    exit 1
fi
echo -e " - Source directory/link set to \"$b$src_dir$n\""

# assert existence of destination directory
dst_dir="${2%/*}"  # remove trailing /, if present
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
for set_ in $(ls $src_dir); do

    printf "\nProcessing $set_ set\n"

    for class in $(ls $src_dir/$set_); do

        printf " > Processing class \"$class\":"

        # define src and dst
        src_class_dir="$src_dir/$set_/$class"
        dst_class_dir="$dst_dir/$set_/$class"
        mkdir -p $dst_class_dir

        # for each video in the class
        for video in $(ls $src_class_dir); do

            printf " \"$video\""

            # define src and dst video paths
            src_video_path="$src_class_dir/$video"
            dst_video_path="$dst_class_dir/${video%.*}%03d.png"

            ffmpeg \
                -loglevel error \
                -i $src_video_path \
                $dst_video_path

        done
        echo ""
    done
done

echo -e "${r}Done. Exiting.${n}"
exit 0

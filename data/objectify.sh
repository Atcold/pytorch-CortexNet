################################################################################
# Link videos into object classes
################################################################################
# Alfredo Canziani, Apr 17
################################################################################
# Run as:
# ./objectify.sh src_path/ dst_path/
################################################################################

# Originally used these. Now they are arguments.
src="sampled-data"         # one folder per video
dst="object-sampled-data"  # one folder per object
src="dumped-sampled-data"         # one folder per video
dst="dumped-object-sampled-data"  # one folder per object

src=${1%/*}  # remove trailing /, if present
dst=${2%/*}  # remove trailing /, if present

classes=$(ls processed-data/train/)
sets="train val"

for c in $classes; do
    echo "Processing class $c"
    for s in $sets; do
        dst_dir="$dst/$s/$c"
        mkdir $dst_dir
        for v in $(ls $src/$s/$c-*/*); do
            video_name=${v#$src/$s/$c-}     # removes leading crap
            video_name=${video_name/'/'/-}  # convert / into -
            ln -s ../../../$v $dst_dir/$video_name
        done
    done
done


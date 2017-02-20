# Add frame number to video, to check correct loading
# ./add_frame_numbering.sh 256min_data_set/barcode/20160613_140057.mp4
# generate labelled small_data_set/barcode/20160613_140057-nb.mp4

src_video=$1
dst_dir="small_data_set"
dst_video="${src_video%.*}-nb.mp4"
dst_video="$dst_dir/${dst_video#*/}"
dst_dir="${dst_video%/*}"

mkdir -p $dst_dir
ffmpeg \
    -i $src_video \
    -filter:v "drawtext=fontsize=200:fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=50:fontcolor=white: box=1: boxcolor=0x00000099" \
    -loglevel quiet \
    $dst_video
printf "$src_video --> $dst_video\n"

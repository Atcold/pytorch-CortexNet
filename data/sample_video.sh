################################################################################
# Video sampler: y_i[n] = x[n_i + N * n], i < N
################################################################################
# Alfredo Canziani, Mar 17
################################################################################
# Run as
# ./sample_video.sh src_video dst_prefix
################################################################################

src="small_data_set/cup/sfsdfs-nb.mp4"
dst="sampled/sfsdfs-nb"
src="data_set/barcode/20160613_140057.mp4"
dst="sampled/20160613_140057"
src="data_set/floor/VID_20160605_094332.mp4"
dst="sampled/VID_20160605_094332"
src="/home/atcold/Videos/20170416_184611.mp4"
dst="bme-car/20170416_184611"
src="/home/atcold/Videos/20170418_113638.mp4"
dst="bme-chair/20170418_113638"
src="/home/atcold/Videos/20160603_133515.mp4"
dst="abhi-car/20160603_133515"
src="/home/atcold/Videos/20170419_125021.mp4"
dst="bme-chair/20170419_125021"

src=$1
dst=$2

k=5
kk=$(awk "BEGIN{print 1/$k}")
ffmpeg \
    -i $src \
    -an \
    -loglevel error \
    -filter_complex \
        "setpts=$kk*PTS, \
        scale=w=2*trunc(128*max(1\, iw/ih)):h=2*trunc(128*max(1\, ih/iw))[m]; \
        [m]select=n=$k:e=(mod(n\, $k)+1)*lt(n\, 564) \
        $(for ((i=1; i<=$k; i++)); do
            echo -n "[a$i]"
        done)" \
    $(for ((i=1; i<=$k; i++)); do
        echo -n "-r 31230000/1042111 -map [a$i] $dst-$i.mp4 "
    done)

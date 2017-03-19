#!/usr/bin/gnuplot -c

# Plot MSE and CE train loss iteratively

# Run as:
# ./show_error_exp.plt EXP -i  # to run it iteratively
# ./show_error_exp.plt EXP     # to run it statically

# Alfredo Canziani, Mar 17

# Get most updated experiments
!./update_experiments.sh -q

# set white on black theme
set terminal wxt background rgb "black" noraise
set xlabel textcolor rgb "white"
set ylabel textcolor rgb "white"
set y2label textcolor rgb "white"
set key textcolor rgb "white"
set border lc rgb 'white'
set grid lc rgb 'white'

set grid
set xlabel "mini batch index / 10"
set ylabel "mMSE"
set y2label "CE"
set y2tics
if (ARG1 + 0 < 20) {  # "+ 0" conversion string to number
    plot \
        "< awk '/data/{print $18,$21,$25}' ../results/".ARG1."/train.log" \
        u 0:1 w lines lw 2 title "MSE", \
        "" \
        u 0:2 w lines lw 2 title "CE" axis x1y2, \
        "" \
        u 0:3 w lines lw 2 title "rpl MSE"
} else {
    plot \
        "< awk '/data/{print $18,$21,$25,$29}' ../results/".ARG1."/train.log" \
        u 0:1 w lines lw 2 title "MSE", \
        "" \
        u 0:3 w lines lw 2 title "rpl MSE", \
        "" \
        u 0:2 w lines lw 2 title "CE" axis x1y2, \
        "" \
        u 0:4 w lines lw 2 title "per CE" axis x1y2
}

if (ARG2 ne '-i') {
    pause -1  # just hang in there
    exit
}

pause 5  # wait 5 seconds
reread   # and start over

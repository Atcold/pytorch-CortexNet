#!/usr/bin/env gnuplot

# Plot MSE and CE train loss iteratively
# Alfredo Canziani, Mar 17


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
set ylabel "MSE"
set y2label "CE"
set y2tics
plot \
    "< awk '/data/{print $18,$21}' ../last/train.log" \
    u 0:1 w lines title "MSE" lw 2, \
    "" \
    u 0:2 w lines title "CE" axis x1y2 lw 2

# pause -1  # just hang in there

pause 5  # wait 5 seconds
reread   # and start over

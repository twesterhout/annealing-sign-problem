#!/usr/bin/gnuplot

set term pngcairo size 640,480 enhanced
set output "correlation.png"

set key bottom right
set xlabel "J / B"
set ylabel "Accuracy"
# set ylabel "overlap"
# set xrange [0:2]
# set yrange [0.8:1]
plot "remote/correlation.dat" u ($3 / $2):1 w p \
         pt 7 ps 0.5 lc rgb "#0060ad" title "Pyrochlore 32"
# pt 7 ps 0.5 lc rgb "#dd181f" title "Kagome 36"

#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 7cm, 5cm \
    transparent enhanced color \
    font "Latin Modern Math,12"

load "../experiments/palettes/dark2.pal"

# set logscale x
# set xrange [80:300000]
# set xtics ("0" 0, "2×10^5" 2e5, "4×10^5" 4e5, "6×10^5" 6e5, "8×10^5" 8e5, "10^6" 1e6)
set xtics 100 scale 1.5 nomirror logscale format "10^{%T}"
set ytics nomirror
# format "10^{%T}"
# set ytics 0.25
# set yrange [0.1:1.05]
set key top left spacing 1.1 width -4.5 font "Latin Modern Math,10"
set border 3

# set xlabel "Index" # font "Latin Modern Math,14"
# set ylabel "" # font "Latin Modern Math,14"

set logscale x

set border lt 1 lw 1 lc "black" back
set grid ls 1 lw 1 lc rgb "#e0000000" dt (6, 8)
set datafile separator ","

set output "frustration_probability.pdf"
plot [1e-11:1.9e-2][0.4:1.0] \
    "../experiments/is_frustrated/heisenberg_kagome_16.csv" u 1:2 w boxes ls 1 lw 2 \
        fillstyle transparent solid 0.5 title "16-site Kagome lattice", \
    "../experiments/is_frustrated/heisenberg_kagome_18.csv" u 1:2 w boxes ls 2 lw 2 \
        fillstyle transparent solid 0.6 title "18-site Kagome lattice", \
    "../experiments/is_frustrated/sk_16_3.csv" u 1:2 w boxes ls 3 lw 2 \
        fillstyle transparent solid 0.7 title "16-site random, realization №3", \

set output
system("convert -density 600 frustration_probability.pdf -quality 00 frustration_probability.png")



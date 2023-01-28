#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 7cm, 5cm \
    transparent enhanced color \
    font "Latin Modern Math,12"

load "../experiments/palettes/dark2.pal"

# set logscale x
# set xrange [80:300000]
set xtics ("0" 0, "2×10^5" 2e5, "4×10^5" 4e5, "6×10^5" 6e5, "8×10^5" 8e5, "10^6" 1e6)
set xtics scale 1.5 nomirror
set ytics 100 scale 1.5 logscale nomirror format "10^{%T}"
# set ytics 0.25
# set yrange [0.1:1.05]
set key spacing 1.1
set border 3

# set xlabel "Index" # font "Latin Modern Math,14"
# set ylabel "" # font "Latin Modern Math,14"

set logscale y

set border lt 1 lw 1 lc "black" back
set grid ls 1 lw 1 lc rgb "#e0000000" dt (6, 8)
set datafile separator ","

set output "coupling_distribution.pdf"
plot [:8.5e5][1e-9:] \
    "../experiments/couplings/heisenberg_kagome_16.csv" u 0:1 w l ls 1 lw 4 title "16-site Kagome lattice", \
    "../experiments/couplings/heisenberg_kagome_18.csv" u 0:1 w l ls 2 lw 4 title "18-site Kagome lattice", \
    "../experiments/couplings/sk_16_3.csv" u 0:1 w l ls 4 lw 4 title "16-site random, realization №3", \

set output
system("convert -density 600 coupling_distribution.pdf -quality 00 coupling_distribution.png")


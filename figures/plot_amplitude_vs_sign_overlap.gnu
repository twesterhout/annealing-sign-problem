#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "../experiments/palettes/dark2.pal"

# set logscale x
# set xrange [80:300000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
# set xtics scale 1.5
# set ytics 0.25
# set yrange [0.1:1.05]
set key top left width -2 font "Latin Modern Math,10"
set grid ls 1 lw 1 lc rgb "#e0000000" dt (6, 8)

set xlabel "Amplitude overlap" # font "Latin Modern Math,14"
set ylabel "Sign overlap" # font "Latin Modern Math,14"

# set border lt 1 lw 1 lc "black" back
# set grid
set datafile separator ","

set output "amplitude_vs_sign_overlap.pdf"
# plot [0.2:][0.2:] \
#     "../experiments/lilo/noise/heisenberg_kagome_16.csv" u 2:3 w p ls 1 pt 7 ps 0.05 title "16-site Kagome lattice", \

points = 'u (column("amplitude_overlap")):(column("median")) with points'
points_border = 'u (column("amplitude_overlap")):(column("median")) with points'
shade = 'u (column("amplitude_overlap")):(column("lower")):(column("upper")) with filledcurves'

plot [0.1:][0.2:] \
    "../experiments/lilo/noise/heisenberg_kagome_16_stats.csv" @shade lc rgb "#b01B9E77" notitle, \
    "" @points ls 1 pt 7 ps 0.5 lw 2 title "16-site Kagome lattice", \
    "" @points_border ls 1 pt 6 lc "black" ps 0.5 lw 1 notitle, \
    \
    "../experiments/lilo/noise/j1j2_square_4x4_stats.csv" @shade lc rgb "#a07570b3" notitle, \
    "" @points ls 3 pt 9 ps 0.6 lw 2 title "16-site J_1-J_2 model", \
    "" @points_border ls 3 pt 8 lc "black" ps 0.6 lw 1 notitle, \
    \
    "../experiments/lilo/noise/heisenberg_kagome_18_stats.csv" @shade lc rgb "#90D95F02" notitle, \
    "" @points ls 2 pt 5 ps 0.5 lw 2 title "18-site Kagome lattice", \
    "" @points_border ls 2 pt 4 lc "black" ps 0.5 lw 1 notitle, \
    \

set output "kagome_overlap_noisy.pdf"
set ytics 0.2
set xtics 0.1
plot [0.5:1][0.2:] \
    "_kagome_noisy.csv" \
        u 3:5:7 w filledcurves lc rgb "#901b9e77" notitle, \
    "" u 3:6 w p ls 1 pt 7 ps 1 lw 2 title "36-site Kagome lattice", \
    "" u 3:6 w p ls 1 pt 6 ps 1 lw 2 lc rgb "black" notitle, \
    "_kagome_noisy_3.csv" \
        u 3:5:7 w filledcurves lc rgb "#901b9e77" notitle, \
    "" u 3:6 w p ls 2 pt 7 ps 1 lw 2 title "36-site Kagome lattice (3rd order)", \
    "" u 3:6 w p ls 2 pt 6 ps 1 lw 2 lc rgb "black" notitle

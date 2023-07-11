#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,14"

load "plot_common.gnu"

set key top left
set xlabel "Amplitude overlap" # font "Latin Modern Math,14"
set ylabel "Sign overlap" # font "Latin Modern Math,14"

points = 'u (column("amplitude_overlap")):(column("median")) with points'
points_border = 'u (column("amplitude_overlap")):(column("median")) with points'
shade = 'u (column("amplitude_overlap")):(column("lower")):(column("upper")) with filledcurves'

set output "amplitude_vs_sign_overlap.pdf"
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
set key bottom right font "Latin Modern Math,12"
set ytics 0.2
set xtics 0.1
plot [0.55:1][0.2:] \
    "_kagome_noisy_2.csv" \
        u 3:5:7 w filledcurves lc rgb "#701b9e77" notitle, \
    "_kagome_noisy_3.csv" \
        u 3:5:7 w filledcurves lc rgb "#70d95f02" notitle, \
    "_kagome_noisy_2.csv" \
        u 3:6 w p ls 1 pt 7 ps 1 lw 2 title "2nd extension", \
    "" u 3:6 w p ls 1 pt 6 ps 1 lw 2 lc rgb "black" notitle, \
    "_kagome_noisy_3.csv" \
        u 3:6 w p ls 2 pt 5 ps 1 lw 2 title "3rd extension", \
    "" u 3:6 w p ls 2 pt 4 ps 1 lw 2 lc rgb "black" notitle

#           u (10**($1)):(0):(prob(2)) w filledcurves lc rgb "#701b9e77" notitle, \
#        "" \
#           u (10**($1)):(prob(2)) w l ls 1 lw 2.5 notitle, \
#        "" \
#           u (10**($1)):(0):(prob(3)) w filledcurves lc rgb "#707570b3" notitle, \
#        "" \
#           u (10**($1)):(prob(3)) w l ls 3 lw 2.5 notitle, \
#        "" \
#           u (10**($1)):(0):(prob(4)) w filledcurves lc rgb "#70d95f02" notitle, \

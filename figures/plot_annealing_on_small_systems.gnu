#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,10"

load "plot_common.gnu"

set logscale x
set xrange [80:300000]
set ytics 0.25
set yrange [0.1:1.05]
set key bottom right

set xlabel "Number of sweeps" font "Latin Modern Math,14"
set ylabel "Probability of convergence" font "Latin Modern Math,14"

points = 'u (column("number_sweeps")):(column("acc_prob_mean")) with linespoints'
points_border = 'u (column("number_sweeps")):(column("acc_prob_mean")) with points'
shade = 'u (column("number_sweeps")):(column("acc_prob_mean") - 2 * column("acc_prob_std")):(column("acc_prob_mean") + 2 * column("acc_prob_std")) with filledcurves'

set output "annealing_on_small_systems.pdf"
plot \
    "../experiments/heisenberg_kagome_16.csv" @shade lc rgb "#901B9E77" notitle, \
    "" @points ls 1 pt 7 ps 0.5 lw 2 title "16-site Kagome lattice", \
    "" @points_border ls 1 pt 6 lc "black" ps 0.5 lw 1 notitle, \
    \
    "../experiments/heisenberg_kagome_18.csv" @shade lc rgb "#90D95F02" notitle, \
    "" @points ls 2 pt 5 ps 0.5 lw 2 title "18-site Kagome lattice", \
    "" @points_border ls 2 pt 4 lc "black" ps 0.5 lw 1 notitle, \
    \
    "../experiments/j1j2_square_4x4.csv" @shade lc rgb "#907570b3" notitle, \
    "" @points ls 3 pt 13 ps 0.6 lw 2 title "16-site J_1-J_2 model", \
    "" @points_border ls 3 pt 12 lc "black" ps 0.6 lw 1 notitle, \
    \
    "../experiments/sk_16_1.csv" @shade lc rgb "#90E7298A" notitle, \
    "" @points ls 4 pt 9 ps 0.5 lw 2 title "16-site random, realization №1", \
    "" @points_border ls 4 pt 8 lc "black" ps 0.5 lw 1 notitle, \
    \
    "../experiments/sk_16_2.csv" @shade lc rgb "#9066A61E" notitle, \
    "" @points ls 5 pt 9 ps 0.6 lw 2 title "16-site random, realization №2", \
    "" @points_border ls 5 pt 8 lc "black" ps 0.6 lw 1 notitle, \
    \
    "../experiments/sk_16_3.csv" @shade lc rgb "#90E6AB02" notitle, \
    "" @points ls 6 pt 9 ps 0.6 lw 2 title "16-site random, realization №3", \
    "" @points_border ls 6 pt 8 lc "black" ps 0.6 lw 1 notitle, \


#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 8cm, 6cm \
    transparent enhanced color \
    font "Latin Modern Math,10"

load "palettes/dark2.pal"

set logscale x
set xrange [80:150000]
# set xtics ("10^2" 100, "10^3" 1000, "10^4" 10000, "10^5" 100000)
set xtics scale 1.5
set ytics 0.1
set yrange [0.1:1.05]
set key bottom right

set xlabel "Number of sweeps"
set ylabel "Probability of convergence"

set border lt 1 lw 1.5 lc "black" back
set grid
set datafile separator ","

set output "small_systems_v2.pdf"
plot \
    "Figure_1/combined_heisenberg_kagome_16.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#901B9E77" notitle, \
    "Figure_1/combined_heisenberg_kagome_16.csv" \
        u 1:2 with linespoints ls 1 pt 7 ps 0.5 lw 2 title "16-site Kagome lattice", \
    "Figure_1/combined_heisenberg_kagome_16.csv" \
        u 1:2 with points ls 1 pt 6 lc "black" ps 0.5 lw 1 notitle, \
    "Figure_1/combined_heisenberg_kagome_18.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#90D95F02" notitle, \
    "Figure_1/combined_heisenberg_kagome_18.csv" \
        u 1:2 with linespoints ls 2 pt 5 ps 0.5 lw 2 title "18-site Kagome lattice", \
    "Figure_1/combined_heisenberg_kagome_18.csv" \
        u 1:2 with points ls 2 lc "black" pt 4 ps 0.5 notitle, \
    "Figure_1/combined_j1j2_square_4x4.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#907570b3" notitle, \
    "Figure_1/combined_j1j2_square_4x4.csv" \
        u 1:2 w linespoints ls 3 pt 13 lw 2 ps 0.6 title "16-site J_1-J_2 model", \
    "Figure_1/combined_j1j2_square_4x4.csv" \
        u 1:2 with points ls 3 pt 12 lc "black" ps 0.6 lw 1 notitle, \
    "Figure_1/combined_sk_16_1.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#90E7298A" notitle, \
    "Figure_1/combined_sk_16_1.csv" \
        u 1:2 with linespoints ls 4 pt 9 lw 2 ps 0.6 title "16-site random, realization №1", \
    "Figure_1/combined_sk_16_1.csv" \
        u 1:2 with points ls 4 pt 8 lc "black" ps 0.6 notitle, \
    "Figure_1/combined_sk_16_2.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#9066A61E" notitle, \
    "Figure_1/combined_sk_16_2.csv" \
        u 1:2 with linespoints ls 5 pt 9 lw 2 ps 0.6 title "16-site random, realization №2", \
    "Figure_1/combined_sk_16_2.csv" \
        u 1:2 with points ls 5 pt 8 lc "black" ps 0.6 notitle, \
    "Figure_1/combined_sk_16_3.csv" \
        u 1:($2 - 2 * $3):($2 + 2 * $3) with filledcurves lc rgb "#90E6AB02" notitle, \
    "Figure_1/combined_sk_16_3.csv" \
        u 1:2 with linespoints ls 6 pt 9 lw 2 ps 0.6 title "16-site random, realization №3", \
    "Figure_1/combined_sk_16_3.csv" \
        u 1:2 with points ls 6 pt 8 lc "black" ps 0.6 notitle

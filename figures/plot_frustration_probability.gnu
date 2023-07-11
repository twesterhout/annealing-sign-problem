#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 7cm, 5cm \
    transparent enhanced color \
    font "Latin Modern Math,12"

load "plot_common.gnu"

set border 3
set key top left spacing 1.1 width -4.5 font "Latin Modern Math,10"
set xtics 100 scale 1.5 nomirror logscale format "10^{%T}"
set ytics nomirror
set logscale x

set output "frustration_probability.pdf"
plot [1e-11:1.9e-2][0.4:1.0] \
    "../experiments/is_frustrated/heisenberg_kagome_16.csv" \
      u 1:2 w boxes ls 1 lw 2 fs solid fc rgb "#8ddac3" notitle, \
    "" \
      u 1:2 w boxes ls 1 lw 2 title "16-site Kagome lattice", \
    "../experiments/is_frustrated/heisenberg_kagome_18.csv" \
      u 1:2 w boxes ls 2 lw 2 fs solid fc rgb "#dfac85" notitle, \
    "" \
      u 1:2 w boxes ls 2 lw 2 title "18-site Kagome lattice", \
    "../experiments/is_frustrated/sk_16_3.csv" \
      u 1:2 w boxes ls 3 lw 2 fs solid fc rgb "#ada9d8" notitle, \
    "" \
      u 1:2 w boxes ls 3 lw 2 title "16-site random, realization №3"

set output
_ = convert_pdf_to_png("frustration_probability.pdf")


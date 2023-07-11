#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 7cm, 5cm \
    transparent enhanced color \
    font "Latin Modern Math,12"

load "plot_common.gnu"

set xtics ("0" 0, "2×10^5" 2e5, "4×10^5" 4e5, "6×10^5" 6e5, "8×10^5" 8e5, "10^6" 1e6)
set xtics scale 1.5 nomirror
set ytics 100 scale 1.5 logscale nomirror format "10^{%T}"

set key spacing 1.1
set border 3
set logscale y

set output "coupling_distribution.pdf"
plot [:8.5e5][1e-9:] \
    "../experiments/couplings/heisenberg_kagome_16.csv" \
      u 0:1 w l ls 1 lw 4 title "16-site Kagome lattice", \
    "../experiments/couplings/heisenberg_kagome_18.csv" \
      u 0:1 w l ls 2 lw 4 title "18-site Kagome lattice", \
    "../experiments/couplings/sk_16_3.csv" \
      u 0:1 w l ls 3 lw 4 title "16-site random, realization №3", \

set output
_ = convert_pdf_to_png("coupling_distribution.pdf")

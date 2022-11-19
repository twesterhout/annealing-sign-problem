#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 2cm \
    transparent enhanced color \
    font "Latin Modern Math,9"

load "palettes/spectral.pal"

set xrange [0.8:1]
set xtics 0.8,0.05,1.0
set yrange [100:5500]
set logscale y
set border lt 1 lw 1.5 lc "black" back
unset key
set grid
set lmargin at screen 0.15
set rmargin at screen 0.95
set tmargin at screen 1
set bmargin at screen 0.15

set output "scatter_pyrochlore.pdf"
plot [:][100:5500] "pyrochlore_sampled_power=0.1_cutoff=0.0002.dat" \
        u 3:1 w p ls 8 pt 7 ps 0.2 title "not extended", \
     "" u 9:1 w p ls 1 pt 7 ps 0.2 title "extended three times"

set output "scatter_kagome.pdf"
plot [:][100:5500] "kagome_sampled_power=0.1_cutoff=0.0002.dat" \
        u 3:1 w p ls 8 pt 7 ps 0.2 title "not extended", \
     "" u 9:1 w p ls 1 pt 7 ps 0.2 title "extended three times"

set output "scatter_sk.pdf"
plot [:][100:5500] "sk_sampled_power=0.1_cutoff=0.0002.dat" \
        u 3:1 w p ls 8 pt 7 ps 0.2 title "not extended", \
     "" u 9:1 w p ls 1 pt 7 ps 0.2 title "extended three times"

set output
system("for m in pyrochlore kagome sk; do \
          convert -density 600 scatter_${m}.pdf -quality 00 scatter_${m}.png; \
        done")

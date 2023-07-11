#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 3cm \
    transparent enhanced color \
    font "Latin Modern Math,8"

load "palettes/spectral.pal"

set xrange [0.8:1]
set xtics ("" 0.8, "" 0.85, "" 0.9, "" 0.95, "" 1)

set border lt 1 lw 1.5 lc "black" back
unset key
set grid
set lmargin at screen 0.15
set rmargin at screen 0.95
set tmargin at screen 0.95
set bmargin at screen 0.05

set output "density_pyrochlore.pdf"
set ytics ("0" 0, "25" 25, "50" 50, "75" 75, "100" 100)
plot [:][:100] "density_of_states.pyrochlore_sampled_power=0.1_cutoff=0.0002.dat" \
        u 1:2 w l ls 8 lw 2.5 title "not extended", \
     "" u 1:3 w l ls 7 lw 2.5 title "extended once", \
     "" u 1:4 w l ls 3 lw 2.5 title "extended twice", \
     "" u 1:5 w l ls 1 lw 2.5 title "extended three times"

set output "density_kagome.pdf"
set ytics 0,4
plot [:][0:16] "density_of_states.kagome_sampled_power=0.1_cutoff=0.0002.dat" \
        u 1:2 w l ls 8 lw 2 title "not extended", \
     "" u 1:3 w l ls 7 lw 2 title "extended once", \
     "" u 1:4 w l ls 3 lw 2 title "extended twice", \
     "" u 1:5 w l ls 1 lw 2 title "extended three times"

set output "density_sk.pdf"
plot "density_of_states.sk_sampled_power=0.1_cutoff=0.0002.dat" \
        u 1:2 w l ls 8 lw 2 title "not extended", \
     "" u 1:3 w l ls 7 lw 2 title "extended once", \
     "" u 1:4 w l ls 3 lw 2 title "extended twice", \
     "" u 1:5 w l ls 1 lw 2 title "extended three times"

set output
system("for m in pyrochlore kagome sk; do \
          convert -density 600 density_${m}.pdf -quality 00 density_${m}.png; \
        done")

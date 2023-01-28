#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 2cm \
    transparent enhanced color \
    font "Latin Modern Math,8"

# load "../experiments/palettes/spectral.pal"
load "../experiments/palettes/dark2.pal"

set datafile separator ","

set border 3 lt 1 lw 1 lc "black" front
set grid ls 1 lw 0.8 lc rgb "#e5000000" dt (6, 8)

set logscale x
set xtics nomirror format "10^%T"
set ytics nomirror 0.4

prob(i) = column(i) > 1e-3 ? column(i) : 1/0

set output "pyrochlore_size_pdf.pdf"
plot [1e1:1.3e5] \
     "_pyrochlore_size_pdf.csv" \
        u (10**($1)):(0):(prob(2)) w filledcurves lc rgb "#701b9e77" notitle, \
     "_pyrochlore_size_pdf.csv" \
        u (10**($1)):(prob(2)) w l ls 1 lw 2.5 notitle, \
     "_pyrochlore_size_pdf.csv" \
        u (10**($1)):(0):(prob(3)) w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u (10**($1)):(prob(3)) w l ls 3 lw 2.5 notitle, \
     "_pyrochlore_size_pdf.csv" \
        u (10**($1)):(0):(prob(4)) w filledcurves lc rgb "#70d95f02" notitle, \
     "" \
        u (10**($1)):(prob(4)) w l ls 2 lw 2.5 notitle, \

set output "kagome_size_pdf.pdf"
plot [1e1:3.9e5] \
     "_kagome_size_pdf.csv" \
        u (10**($1)):(0):(prob(2)) w filledcurves lc rgb "#701b9e77" notitle, \
     "" \
        u (10**($1)):(prob(2)) w l ls 1 lw 2.5 notitle, \
     "" \
        u (10**($1)):(0):(prob(3)) w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u (10**($1)):(prob(3)) w l ls 3 lw 2.5 notitle, \
     "" \
        u (10**($1)):(0):(prob(4)) w filledcurves lc rgb "#70d95f02" notitle, \
     "" \
        u (10**($1)):(prob(4)) w l ls 2 lw 2.5 notitle, \

# set output "density_kagome.pdf"
# set ytics 0,4
# plot [:][0:16] "density_of_states.kagome_sampled_power=0.1_cutoff=0.0002.dat" \
#         u 1:2 w l ls 8 lw 2 title "not extended", \
#      "" u 1:3 w l ls 7 lw 2 title "extended once", \
#      "" u 1:4 w l ls 3 lw 2 title "extended twice", \
#      "" u 1:5 w l ls 1 lw 2 title "extended three times"
# 
# set output "density_sk.pdf"
# plot "density_of_states.sk_sampled_power=0.1_cutoff=0.0002.dat" \
#         u 1:2 w l ls 8 lw 2 title "not extended", \
#      "" u 1:3 w l ls 7 lw 2 title "extended once", \
#      "" u 1:4 w l ls 3 lw 2 title "extended twice", \
#      "" u 1:5 w l ls 1 lw 2 title "extended three times"
# 
set output
system("for m in pyrochlore kagome; do \
          convert -density 600 ${m}_size_pdf.pdf -quality 00 ${m}_size_pdf.png; \
        done")

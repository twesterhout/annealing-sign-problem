#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 3cm \
    transparent enhanced color \
    font "Latin Modern Math,8"

# load "../experiments/palettes/spectral.pal"
load "../experiments/palettes/dark2.pal"

set datafile separator ","

# set xtics ("" 0.8, "" 0.85, "" 0.9, "" 0.95, "" 1)

set border lt 1 lw 1 lc "black" front
# unset key
set grid ls 1 lw 0.8 lc rgb "#e5000000" dt (6, 8)
# set lmargin at screen 0.15
# set rmargin at screen 0.95
# set tmargin at screen 0.95
# set bmargin at screen 0.05

set output "pyrochlore_overlap_pdf.pdf"
set ytics 2
plot [0:1] \
     "_pyrochlore_overlap_pdf.csv" \
        u 1:(0):2 w filledcurves lc rgb "#601b9e77" notitle, \
     "_pyrochlore_overlap_pdf.csv" \
        u 1:2 w l ls 1 lw 2.5 notitle, \
     "" \
        u 1:(0):3 w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u 1:3 w l ls 3 lw 2.5 notitle, \
     "" \
        u 1:(0):4 w filledcurves lc rgb "#80d95f02" notitle, \
     "" \
        u 1:4 w l ls 2 lw 2.5 notitle, \

set output "kagome_overlap_pdf.pdf"
set ytics 2
plot [0:1] \
     "_kagome_overlap_pdf.csv" \
        u 1:(0):2 w filledcurves lc rgb "#601b9e77" notitle, \
     "" \
        u 1:2 w l ls 1 lw 2.5 notitle, \
     "" \
        u 1:(0):3 w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u 1:3 w l ls 3 lw 2.5 notitle, \
     "" \
        u 1:(0):4 w filledcurves lc rgb "#80d95f02" notitle, \
     "" \
        u 1:4 w l ls 2 lw 2.5 notitle, \

set output "kagome_overlap_pdf_noisy.pdf"
set ytics 2
plot [0:1] \
     "_kagome_overlap_pdf.csv" \
        u 1:(0):4 w filledcurves lc rgb "#601b9e77" notitle, \
     "" \
        u 1:4 w l ls 1 lw 2.5 notitle, \
     "_kagome_overlap_pdf_5e-1.csv" \
        u 1:(0):4 w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u 1:4 w l ls 3 lw 2.5 notitle, \
     "_kagome_overlap_pdf_1e0.csv" \
        u 1:(0):4 w filledcurves lc rgb "#80d95f02" notitle, \
     "" \
        u 1:4 w l ls 2 lw 2.5 notitle, \
     "_kagome_overlap_pdf_2e0.csv" \
        u 1:(0):4 w filledcurves lc rgb "#80e7298a" notitle, \
     "" \
        u 1:4 w l ls 4 lw 2.5 notitle, \

prob(i) = column(i) > 1e-2 ? column(i) : 1/0

set output "kagome_amplitude_overlap_pdf_noisy.pdf"
set ytics 20
plot [0:1][0:100] \
     "_kagome_amplitude_overlap_pdf_5e-1.csv" \
        u 1:(0):(prob(4)) w filledcurves lc rgb "#807570b3" notitle, \
     "" \
        u 1:(prob(4)) w l ls 3 lw 2.5 notitle, \
     "_kagome_amplitude_overlap_pdf_1e0.csv" \
        u 1:(0):(prob(4)) w filledcurves lc rgb "#80d95f02" notitle, \
     "" \
        u 1:(prob(4)) w l ls 2 lw 2.5 notitle, \
     "_kagome_amplitude_overlap_pdf_2e0.csv" \
        u 1:(0):(prob(4)) w filledcurves lc rgb "#80e7298a" notitle, \
     "" \
        u 1:(prob(4)) w l ls 4 lw 2.5 notitle, \


set output "sk_overlap_pdf.pdf"
set ytics 2
plot [0:1] \
     "_sk_overlap_pdf.csv" \
        u 1:(0):2 w filledcurves lc rgb "#601b9e77" notitle, \
     "" \
        u 1:2 w l ls 1 lw 2.5 notitle, \
     "" \
        u 1:(0):3 w filledcurves lc rgb "#707570b3" notitle, \
     "" \
        u 1:3 w l ls 3 lw 2.5 notitle, \
     "" \
        u 1:(0):4 w filledcurves lc rgb "#80d95f02" notitle, \
     "" \
        u 1:4 w l ls 2 lw 2.5 notitle, \

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
# set output
# system("for m in pyrochlore kagome sk; do \
#           convert -density 600 density_${m}.pdf -quality 00 density_${m}.png; \
#         done")

set output
system("for m in pyrochlore kagome sk; do \
          convert -density 600 ${m}_overlap_pdf.pdf -quality 00 ${m}_overlap_pdf.png; \
        done")

set terminal pdfcairo size 6cm, 5cm \
    transparent enhanced color \
    font "Latin Modern Math,12"

set output "kagome_overlap_pdf_various_sizes.pdf"
set ytics 2 nomirror format ""
set xtics nomirror
set border 3
set key top left font "Latin Modern Math,10" spacing 1.1
set ylabel "PDF"
set xlabel "Sign overlap"
plot [0:1] \
     "_kagome_overlap_pdf_473_1000.csv" \
        u 1:(6):($4 + 6) w filledcurves lc rgb "#90e7298a" notitle, \
     "_kagome_overlap_pdf_473_1000.csv" \
        u 1:($4 + 6) w l ls 4 lw 2.5 title "[473, 1000]", \
     "_kagome_overlap_pdf_224_473.csv" \
        u 1:(4):($4 + 4) w filledcurves lc rgb "#807570b3" notitle, \
     "_kagome_overlap_pdf_224_473.csv" \
        u 1:($4 + 4) w l ls 3 lw 2.5 title "[224 473]", \
     "_kagome_overlap_pdf_106_224.csv" \
        u 1:(2):($4 + 2) w filledcurves lc rgb "#70d95f02" notitle, \
     "_kagome_overlap_pdf_106_224.csv" \
        u 1:($4 + 2) w l ls 2 lw 2.5 title "[106, 224]", \
     "_kagome_overlap_pdf_50_106.csv" \
        u 1:(0):($4 + 0) w filledcurves lc rgb "#601b9e77" notitle, \
     "_kagome_overlap_pdf_50_106.csv" \
        u 1:($4 + 0) w l ls 1 lw 2.5 title "[50, 106]", \


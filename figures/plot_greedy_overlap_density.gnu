#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 3cm \
    transparent enhanced color \
    font "Latin Modern Math,8"

load "plot_common.gnu"

set ytics 2
set xrange [0:1]
set yrange [0:10]

do for [s in "pyrochlore kagome sk"] {
  set output s."_overlap_pdf.pdf"
  plot \
       "_".s."_overlap_pdf_0.025.csv" \
          u 1:(0):2 w filledcurves lc rgb "#601b9e77" notitle, \
       "" \
          u 1:2 w l ls 1 lw 2.5 notitle, \
       "" \
          u 1:(0):3 w filledcurves lc rgb "#607570b3" notitle, \
       "" \
          u 1:3 w l ls 3 lw 2.5 notitle, \
       "" \
          u 1:(0):4 w filledcurves lc rgb "#70d95f02" notitle, \
       "" \
          u 1:4 w l ls 2 lw 2.5 notitle, \

  set output
  _ = convert_pdf_to_png(s."_overlap_pdf.pdf")
}

# set output "kagome_overlap_pdf.pdf"
# set ytics 2
# plot \
#      "_kagome_overlap_pdf.csv" \
#         u 1:(0):2 w filledcurves lc rgb "#601b9e77" notitle, \
#      "" \
#         u 1:2 w l ls 1 lw 2.5 notitle, \
#      "" \
#         u 1:(0):3 w filledcurves lc rgb "#607570b3" notitle, \
#      "" \
#         u 1:3 w l ls 3 lw 2.5 notitle, \
#      "" \
#         u 1:(0):4 w filledcurves lc rgb "#70d95f02" notitle, \
#      "" \
#         u 1:4 w l ls 2 lw 2.5 notitle, \
# 
# set output "sk_overlap_pdf.pdf"
# set ytics auto
# plot \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:(0):2 w filledcurves lc rgb "#801b9e77" notitle, \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:2 w l ls 1 lw 2.5 notitle, \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:(0):3 w filledcurves lc rgb "#607570b3" notitle, \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:3 w l ls 3 lw 2.5 notitle, \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:(0):4 w filledcurves lc rgb "#50d95f02" notitle, \
#      "_sk_overlap_pdf_0.025.csv" \
#         u 1:4 w l ls 2 lw 2.5 notitle, \
# 
# set output
# system("for m in pyrochlore kagome sk; do \
#           convert -density 600 ${m}_overlap_pdf.pdf -quality 00 ${m}_overlap_pdf.png; \
#         done")

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


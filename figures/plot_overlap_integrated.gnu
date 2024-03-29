#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 2cm, 1.4cm \
    transparent enhanced color \
    font "Latin Modern Math,6"

load "plot_common.gnu"

set tmargin 0.4
set rmargin 0.6

set ylabel "CCDF"

set xtics 0.25
set ytics 0.25

do for [s in "pyrochlore kagome sk"] {
  set output s."_overlap_integrated.pdf"
  plot [0:1][0:1] \
       "_".s."_overlap_integrated.csv" \
          u 1:2 w l ls 1 lw 3 notitle, \
       "" \
          u 1:3 w l ls 3 lw 3 notitle, \
       "" \
          u 1:4 w l ls 2 lw 3 notitle, \

  set output s."_overlap_integrated_per_size.pdf"
  plot [0:1][0:1] \
       "_".s."_overlap_integrated_50_106.csv" \
          u 1:4 w l ls 1 lw 3 title "[50, 106]", \
       "_".s."_overlap_integrated_106_224.csv" \
          u 1:4 w l ls 2 lw 3 title "[106, 224]", \
       "_".s."_overlap_integrated_224_473.csv" \
          u 1:4 w l ls 3 lw 3 title "[224, 473]" , \
       "_".s."_overlap_integrated_473_1000.csv" \
          u 1:4 w l ls 4 lw 3 title "[473, 1000]"

  set output
  _ = convert_pdf_to_png(s."_overlap_integrated.pdf")
  _ = convert_pdf_to_png(s."_overlap_integrated_per_size.pdf")
}

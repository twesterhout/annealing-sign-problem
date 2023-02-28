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

  set output
  _ = convert_pdf_to_png(s."_overlap_integrated.pdf")
}

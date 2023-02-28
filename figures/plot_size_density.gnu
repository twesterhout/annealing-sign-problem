#!/usr/bin/gnuplot --persist

set terminal pdfcairo size 4cm, 1.75cm \
    transparent enhanced color \
    font "Latin Modern Math,8"

load "plot_common.gnu"

set border 3
set logscale x
set xtics nomirror format "10^%T"
set ytics nomirror 0.5
set xrange [2e1:3e6]
set yrange [0:1.8]

do for [s in "pyrochlore kagome sk"] {
  set output s."_size_pdf.pdf"
  plot \
       "_".s."_size_pdf_0.05.csv" \
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
          u (10**($1)):(prob(4)) w l ls 2 lw 2.5 notitle

  set output
  _ = convert_pdf_to_png(s."_size_pdf.pdf")
}

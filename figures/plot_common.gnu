load "../experiments/palettes/dark2.pal"
# Overwrite magenta because it's too bright
set style line 4 lc rgb '#d0408a' 

set datafile separator ","
set border lt 1 lw 1 lc "black" front
set grid ls 1 lw 0.8 lc rgb "#e5000000" dt (6, 8)

strip_extension(s) = system(sprintf("echo '%s' | rev | cut -f 2- -d '.' | rev", s))
convert_pdf_to_png(s) = system(sprintf("convert -density 600 %s.pdf -quality 00 %s.png", strip_extension(s), strip_extension(s)))

prob(i) = column(i) > 1e-3 ? column(i) : 1/0

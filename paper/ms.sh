set local=./

latex sham_colors.tex
bibtex sham_colors
latex sham_colors.tex
latex sham_colors.tex
dvips sham_colors.dvi -Ppdf -G0 -z -t a4
ps2pdf sham_colors.ps

#!/bin/bash
# Build every Paper 2 main figure: export data, render SVG, convert to PDF.
#
# PDF conversion uses inkscape rather than rsvg-convert. rsvg emits uncompressed
# PDF content streams, which turned the contour map into an 8.6 MB file where
# inkscape produces 456 kB from the identical SVG.
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"          # bundle root: holds config.py and output/

# File size in bytes, portable across BSD/macOS (stat -f%z) and GNU/Linux (stat -c%s).
fsize() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }

echo "== exporting data =="
( cd "$ROOT" && python3 figures_d3/export_data.py )
( cd "$ROOT" && python3 figures_d3/export_si.py )

echo "== rendering SVG =="
mkdir -p svg pdf png
for f in fig1 fig2 fig3 fig4 fig5 fig6 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 ga; do
  node "$f.mjs" > /dev/null
  printf "  %-6s svg %6s kB" "$f" "$(( $(fsize svg/$f.svg) / 1024 ))"
  inkscape --export-type=pdf --export-filename="pdf/$f.pdf" "svg/$f.svg" > /dev/null 2>&1
  inkscape --export-type=png --export-dpi=600 --export-filename="png/$f.png" "svg/$f.svg" > /dev/null 2>&1
  printf "   pdf %6s kB   png %6s kB\n" \
    "$(( $(fsize pdf/$f.pdf) / 1024 ))" "$(( $(fsize png/$f.png) / 1024 ))"
done

# The graphical abstract is submitted as a raster: Elsevier wants at least
# 1328 x 531 px for it, so export a high-resolution PNG alongside the vector.
inkscape --export-type=png --export-dpi=600 --export-filename=png/Graphical_Abstract.png svg/ga.svg > /dev/null 2>&1
GA=$(python3 -c "from PIL import Image; im=Image.open('png/Graphical_Abstract.png'); print(f'{im.width}x{im.height}')")
echo "  graphical abstract PNG: $GA (Elsevier minimum 1328x531)"

TOT=$(du -ck pdf/*.pdf | tail -1 | cut -f1)
echo "== done: ${TOT} kB of PDF =="

# /// script
# dependencies = ["fonttools==4.66.0"]
# ///
"""Outline a string set in a font, for the brand assets (scripts/build-brand.ts).

Usage: uv run scripts/brand-glyphs.py FONT TEXT SIZE [TRACKING]
Prints two lines: the advance width in px, then the SVG path. The path is y-down
with the baseline at y=0, so a caller places it with a translate to the baseline.
"""

import sys

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont


def main() -> None:
    font_path, text, size_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    tracking = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    scale = float(size_arg) / upm
    pen = SVGPathPen(glyph_set)
    x = 0.0
    for char in text:
        name = cmap[ord(char)]
        glyph = glyph_set[name]
        glyph.draw(TransformPen(pen, (scale, 0, 0, -scale, x, 0)))
        x += glyph.width * scale + tracking
    print(x - tracking)
    print(pen.getCommands())


if __name__ == "__main__":
    main()

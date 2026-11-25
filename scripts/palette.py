"""The article's colour palette, imported by every figure script.

One colour per kind of text, the same in every figure:

- real texts with the pattern (the Mengzi passage, inner couplets, regular
  eight-legged essays): blue, with a darker and a lighter shade where two real
  sets or two lines must be told apart;
- composed texts (written in the form on other subjects): light blue;
- nonsense texts (random characters): red;
- recombined texts (segments, clauses or character pairs taken from other
  texts): purple, dark for segments and clauses, light for character pairs;
- shuffled or scrambled texts: dark grey;
- real texts without the pattern (outer couplets, non-parallel prose): bluish
  green, with a darker and a lighter shade where two such sets must be told
  apart.

Used by model_comparison.py (Fig. 1), make_control_figures.py (Figs. 2 and 4),
make_trajectory_figure.py (Fig. 3) and baguwen_pseudo_legs.py (the essay figure).
"""

REAL = "#2171b5"
REAL_DARK = "#08306b"
REAL_LIGHT = "#9ecae1"

COMPOSED = "#6baed6"

NONSENSE = "#d62728"
NONSENSE_LIGHT = "#f4a3a3"  # an earlier stage of the same nonsense condition

RECOMBINED = "#5e3c99"
RECOMBINED_LIGHT = "#a58bd0"

SHUFFLED = "#525252"

WITHOUT_PATTERN = "#00b394"  # was #009e73; slightly more cyan and brighter (2026-11-21)
WITHOUT_PATTERN_DARK = "#007a65"  # was #006d4f
WITHOUT_PATTERN_LIGHT = "#80d9ca"  # was #7fcfb1

# Darker variants for characters printed in a condition's colour, where the line colour is too light to read.
TEXT = {
    "real": REAL_DARK,
    "composed": "#3182bd",
    "nonsense": "#a50f15",
    "shuffled": "#333333",
}

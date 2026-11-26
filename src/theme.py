"""Dark green colour theme shared by every plot of the project.

Importing this module switches matplotlib to file output and applies the theme.
The three series greens were checked for colour-blind separation (OKLab distance > 15
between neighbours); the light one has low contrast on white, so bars always get a
dark outline.
"""
import matplotlib

matplotlib.use("Agg")  # the scripts only save figures, they never open a window
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

DARK, MID, LIGHT = "#0A6332", "#2E9A5E", "#78C896"  # series colours, in this fixed order
SERIES = [DARK, MID, LIGHT]
OUTLINE = "#06331C"                                 # bar edges
NEUTRAL = "#8A948F"                                 # reference lines (chance diagonal)
INK, MUTED_INK = "#1F2A24", "#4A5550"               # text stays neutral, never series-coloured
# one-hue ramp for magnitudes (confusion matrices): near-white to very dark green
GREEN_RAMP = LinearSegmentedColormap.from_list("project_greens",
                                               ["#F3F9F5", "#78C896", "#2E9A5E", "#0A6332", "#042414"])
if "project_greens" not in matplotlib.colormaps:
    matplotlib.colormaps.register(GREEN_RAMP)

plt.rcParams.update({
    "axes.prop_cycle": cycler(color=SERIES),
    "image.cmap": "project_greens",
    "patch.edgecolor": OUTLINE,
    "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#E3E8E5",
    "legend.frameon": False,
})

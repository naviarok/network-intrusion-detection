"""Dark green colour theme shared by every plot of the project.

Importing this module switches matplotlib to file output and applies the theme.
The three series greens were checked for colour-blind separation (OKLab distance > 15
between neighbours); the light one has low contrast on white, so bars always get a
dark outline. Text and axes stay neutral grey; confusion matrices use a one-hue green ramp.
"""
import matplotlib

matplotlib.use("Agg")  # the scripts only save figures, they never open a window
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

DARK, MID, LIGHT = "#0A6332", "#2E9A5E", "#78C896"
SERIES = [DARK, MID, LIGHT]
OUTLINE = "#06331C"
NEUTRAL = "#8A948F"
INK, MUTED_INK = "#1F2A24", "#4A5550"
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

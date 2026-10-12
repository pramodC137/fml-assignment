"""Shared plotting style so every figure in the project looks consistent.

Colours come from a colour-vision-deficiency-checked categorical order.
Charts use at most three series; anything more is split into small multiples.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import config

# Categorical slots, used in this fixed order and never cycled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#b9b8b2"

# Sequential (magnitude): one hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
# Diverging (sign, e.g. correlation): blue <-> red with a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2a"]
)


def set_style() -> None:
    """Apply the project style to all subsequent matplotlib figures."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to results/figures/<name>.png for use in the report."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")

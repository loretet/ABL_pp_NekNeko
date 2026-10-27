"""
plotstyle.py -- ONE place for the look of every figure.

Edit the STYLE dictionary below (or call `set_style(...)` at the top of your
script) to change font size, DPI, LaTeX, panel size, colours... for all plots
at once.

    import nekabl as nk
    nk.set_style(font_size=11, usetex=True, save_format="pdf")

Every plot function in the package
    * builds its axes with `subplots(...)` from here,
    * returns (fig, ax) or (fig, axs) so you can keep drawing on top,
    * accepts `ax=` to draw into an existing axis (to combine plots).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Common setup -- change things here
# ----------------------------------------------------------------------------
STYLE = dict(
    font_size=10,          # base font size (labels, titles)
    tick_size=9,           # tick labels
    legend_size=8,         # legend text
    usetex=False,          # True: real LaTeX (needs a LaTeX install)
    font_family="serif",   # 'serif' (Computer-Modern-like) or 'sans-serif'
    dpi=120,               # on-screen DPI
    save_dpi=300,          # DPI of saved raster figures
    save_format="png",     # default format of `save` ('png', 'pdf', ...)
    figure_dir="figures",  # default folder of `save`
    panel_width=2.4,       # inches, width of one profile panel
    panel_height=3.2,      # inches, height of one profile panel
    wide_width=6.0,        # inches, time series / slices
    wide_height=2.6,
    line_width=1.4,
    grid=True,             # light background grid
    cmap="viridis",        # sequential quantities (speed, TKE, nu_t, ...)
    cmap_div="RdBu_r",     # signed quantities (w', theta', fluxes)
    # categorical colours, used in this fixed order (colour-blind checked).
    # Colour identifies a SIMULATION (or a height / budget term when only one
    # simulation is plotted).
    colors=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#6250d6", "#e34948"],
    # Line style identifies a COMPONENT of a quantity (same in every plot)
    linestyles=dict(total="-", res="--", sgs=":", model="-.", ref="--"),
    # profile evolution (thin lines behind the mean profile, see evolution_lines):
    # coloured by time with this colour map (viridis_r: yellow = first, violet = last);
    # the mean profile is then drawn in `evolution_mean_color`
    evolution_cmap="viridis_r",
    evolution_lw=0.7,
    evolution_alpha=0.55,
    evolution_mean_color="k",
    title_size=9,          # averaging-window title of the profile figures
)


def ls(component):
    """Line style of a component: 'total', 'res' (resolved), 'sgs', 'model', 'ref'."""
    return STYLE["linestyles"][component]


# line styles used to tell apart several heights (spectra, PDFs ...) when
# colour is already used for the simulations
HEIGHT_STYLES = ["-", "--", ":", "-."]


def apply():
    """Push STYLE into matplotlib's rcParams."""
    s = STYLE
    mpl.rcParams.update({
        "font.size": s["font_size"],
        "axes.labelsize": s["font_size"],
        "axes.titlesize": s["font_size"],
        "xtick.labelsize": s["tick_size"],
        "ytick.labelsize": s["tick_size"],
        "legend.fontsize": s["legend_size"],
        "legend.frameon": True,          # light box so legends stay readable over lines
        "legend.framealpha": 0.85,
        "legend.edgecolor": "none",
        "font.family": s["font_family"],
        "mathtext.fontset": "cm" if s["font_family"] == "serif" else "dejavusans",
        "text.usetex": s["usetex"],
        "figure.dpi": s["dpi"],
        "savefig.dpi": s["save_dpi"],
        "savefig.bbox": "tight",
        "lines.linewidth": s["line_width"],
        "axes.prop_cycle": mpl.cycler(color=s["colors"]),
        "axes.grid": s["grid"],
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "image.cmap": s["cmap"],
        "axes.formatter.limits": (-3, 4),      # scientific notation for tiny/huge numbers
        "axes.formatter.use_mathtext": True,
    })
    if s["font_family"] == "serif" and not s["usetex"]:
        mpl.rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"]


def set_style(**kwargs):
    """Update STYLE (any key above) and apply it."""
    unknown = set(kwargs) - set(STYLE)
    if unknown:
        raise KeyError(f"unknown style keys {unknown}; valid: {list(STYLE)}")
    STYLE.update(kwargs)
    apply()


def color(i):
    """i-th categorical colour (fixed order)."""
    return STYLE["colors"][i % len(STYLE["colors"])]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def subplots(nrows=1, ncols=1, kind="profile", sharey=None, sharex=None, **kw):
    """plt.subplots with the package sizes.

    kind = 'profile' (tall panels, shared z axis), 'wide' (time series),
           'square' (spectra, scatter), 'slice' (horizontal/vertical slices).
    """
    s = STYLE
    w, h = dict(profile=(s["panel_width"], s["panel_height"]),
                wide=(s["wide_width"], s["wide_height"]),
                square=(3.2, 3.0),
                slice=(3.4, 3.0))[kind]
    if kind == "wide":
        w = w / max(ncols, 1)
    if sharey is None:
        sharey = kind == "profile"
    fig, ax = plt.subplots(nrows, ncols, figsize=kw.pop("figsize", (w * ncols, h * nrows)),
                           sharey=sharey, sharex=sharex if sharex is not None else False,
                           squeeze=False, constrained_layout=True, **kw)
    return fig, (ax[0, 0] if ax.size == 1 else (ax.ravel() if 1 in ax.shape else ax))


def get_ax(ax=None, kind="profile"):
    """Return (fig, ax): the given ax and its figure, or a new single panel."""
    if ax is None:
        return subplots(kind=kind)
    return ax.figure, ax


def run_height_style(i_run, j_height, n_runs):
    """(colour, line style) for curve j_height of simulation i_run: one
    simulation -> colour per height; several -> colour per simulation and
    line style per height."""
    if n_runs == 1:
        return color(j_height), "-"
    return color(i_run), HEIGHT_STYLES[j_height % len(HEIGHT_STYLES)]

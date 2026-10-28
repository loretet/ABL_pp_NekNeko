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


def as_dict(obj, labels=None):
    """Dataset / list / dict of Datasets (one per simulation) -> {label: Dataset}.

    Labels come from `labels`, else ds.attrs['label'], else 'run 1', 'run 2' ..."""
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, (list, tuple)):
        obj = [obj]
    if isinstance(labels, str):
        labels = [labels]
    out = {}
    for i, d in enumerate(obj):
        lab = labels[i] if labels else getattr(d, "attrs", {}).get("label", f"run {i + 1}")
        out[lab] = d
    return out


def zcoord(ds, zi=None):
    """Vertical coordinate for plots: z, z/zi (number), or z/zi with the zi of
    each dataset (zi=True or 'auto' -> ds.attrs['zi'])."""
    if zi is None or zi is False:
        return ds.z
    if zi is True or zi == "auto":
        return ds.z / ds.attrs["zi"]
    return ds.z / zi


def evolution_times(evo):
    """Times of the evolution profiles (end of each statistics window)."""
    return np.asarray(evo.time.values, float)


def evolution_color(evo, t):
    """Colour of the profile at time t (STYLE['evolution_cmap'] over the time range)."""
    tt = evolution_times(evo)
    f = 0.5 if tt[-1] == tt[0] else (t - tt[0]) / (tt[-1] - tt[0])
    return plt.get_cmap(STYLE["evolution_cmap"])(f)


def evolution_lines(ax, evo, var_or_func, zc_func, zorder=1):
    """Draw every profile of `evo` (a (time, z) Dataset) as a thin, transparent
    line behind the mean, coloured by time (STYLE['evolution_cmap']; viridis_r:
    yellow = first window, violet = last). var_or_func: name, or a function
    ds -> DataArray. zc_func: ds -> vertical coordinate. Returns the x values
    drawn (for axis limits)."""
    if evo is None or "time" not in evo.dims:
        return []
    vals = []
    for i in range(evo.time.size):
        d = evo.isel(time=i)
        try:
            da = var_or_func(d) if callable(var_or_func) else d.get(var_or_func)
        except (KeyError, ValueError, StopIteration):
            da = None
        if da is None:
            continue
        ax.plot(da, zc_func(d), color=evolution_color(evo, float(d.time)), lw=STYLE["evolution_lw"],
                alpha=STYLE["evolution_alpha"], zorder=zorder)
        vals.append(np.asarray(da))
    return vals


def mean_color(i, evo=None):
    """Colour of the mean profile of simulation i: the simulation colour, or
    STYLE['evolution_mean_color'] when the evolution lines are drawn."""
    return STYLE["evolution_mean_color"] if evo is not None else color(i)


def mean_lw(evo=None):
    """Line width of the mean profile (thicker on top of the evolution lines)."""
    return STYLE["line_width"] * (1.4 if evo is not None else 1.0)


def evolution_colorbar(fig, axs, evo, **kw):
    """Colour bar of the evolution lines (time of each window). Returns it."""
    if evo is None or "time" not in evo.dims:
        return None
    import matplotlib as mpl
    tt = evolution_times(evo)
    sm = mpl.cm.ScalarMappable(cmap=STYLE["evolution_cmap"],
                               norm=mpl.colors.Normalize(tt[0], tt[-1] if tt[-1] > tt[0] else tt[0] + 1))
    kw = {"shrink": 0.8, "aspect": 30, "pad": 0.01, **kw}
    cb = fig.colorbar(sm, ax=np.ravel(axs).tolist(), **kw)
    cb.set_label(r"profile time $t$ (s, end of window)", fontsize=STYLE["legend_size"])
    cb.ax.tick_params(labelsize=STYLE["legend_size"])
    cb.formatter = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}")
    cb.update_ticks()
    return cb


def _window_text(ds):
    """'t0-t1 s, N windows of D s' from the attrs set by time_average."""
    a = ds.attrs
    if "t0" not in a or "t1" not in a:
        return None
    n = a.get("n_windows")
    if a.get("source") == "field files":      # snapshots: no averaging windows
        t_first = a["t0"] + (a.get("window_min") or 0.0)
        return f"{t_first:.0f}-{a['t1']:.0f} s ({n} snapshot{'s' if n != 1 else ''})"
    txt = f"{a['t0']:.0f}-{a['t1']:.0f} s"
    w0, w1 = a.get("window_min"), a.get("window_max")
    if n and w0 and w1:
        dt = f"{w0:.0f} s" if abs(w1 - w0) <= 0.02 * w1 else f"{w0:.0f}-{w1:.0f} s"
        txt += f" ({n} window{'s' if n > 1 else ''}, stats every {dt})"
    elif n:
        txt += f" ({n} windows)"
    return txt


def average_title(fig, data, evo=None, title=None):
    """Figure title with the averaging window and output interval of the
    statistics of every simulation (and the time range of the evolution lines).
    title: None -> automatic, False -> none, a string -> that string."""
    if title is False:
        return
    if title is None:
        parts = []
        for lab, ds in data.items():
            w = _window_text(ds)
            if w:
                parts.append(f"mean {w}" if len(data) == 1 else f"{lab}: {w}")
        if evo is not None and "time" in evo.dims:
            tt = evolution_times(evo)
            parts.append(f"thin lines: {tt.size} windows ending {tt[0]:.0f}-{tt[-1]:.0f} s")
        title = ";  ".join(parts)
    if title:
        fig.suptitle(title, fontsize=STYLE["title_size"], color="0.25")


def component_legend(ax, components=("total", "res", "sgs"), names=None, **kw):
    """Extra legend explaining the line styles (in neutral grey)."""
    from matplotlib.lines import Line2D
    names = names or dict(total="total", res="resolved", sgs="SGS", model="model", ref="reference")
    h = [Line2D([], [], color="0.3", ls=ls(c), label=names.get(c, c)) for c in components]
    leg = ax.legend(handles=h, **kw)
    ax.add_artist(leg)
    return leg


def run_legend(ax, labels, **kw):
    """Legend of the simulations (colours)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=color(i), label=l) for i, l in enumerate(labels)]
    leg = ax.legend(handles=h, **kw)
    ax.add_artist(leg)
    return leg


def height_legend(ax, heights, title=None, **kw):
    """Legend of the line styles used for heights (when colour = simulation)."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", ls=HEIGHT_STYLES[j % len(HEIGHT_STYLES)], label=f"z = {z:.0f} m")
         for j, z in enumerate(heights)]
    leg = ax.legend(handles=h, title=title, title_fontsize=STYLE["legend_size"], **kw)
    ax.add_artist(leg)
    return leg


def curves_legend(ax, data, loc_runs="upper right", loc_heights="lower left", extra=None):
    """Legends for 'one curve per height' plots: a single legend (colour =
    height) for one simulation, two compact legends (colours = simulations,
    line styles = heights of the first one) for several."""
    if len(data) == 1:
        ax.legend()
        return
    run_legend(ax, list(data), loc=loc_runs)
    zs = [(np.asarray(d.z.values, float), d.attrs.get("zi")) for d in data.values()]
    if all(zi for _, zi in zs):      # same z/zi in every run -> label the heights in z/zi
        rel = [z / zi for z, zi in zs]
        if all(r.shape == rel[0].shape and np.allclose(r, rel[0], atol=0.02) for r in rel):
            from matplotlib.lines import Line2D
            h = [Line2D([], [], color="0.3", ls=HEIGHT_STYLES[j % len(HEIGHT_STYLES)],
                        label=f"$z/z_i$ = {r:.2g}") for j, r in enumerate(rel[0])]
            ax.add_artist(ax.legend(handles=h, loc=loc_heights))
            return
    height_legend(ax, zs[0][0], loc=loc_heights, title=list(data)[0] if len(data) > 1 else None)


def zlabel(ax, zi=None):
    """Vertical-axis label (z or z/zi), only on the first of y-shared panels."""
    shared = len(ax.get_shared_y_axes().get_siblings(ax)) > 1
    ss = ax.get_subplotspec()
    if not shared or ss is None or ss.is_first_col():
        ax.set_ylabel(label("z/zi" if zi else "z"))


def save(fig, name, folder=None, fmt=None, **kw):
    """Save a figure as <figure_dir>/<name>.<save_format>; returns the path."""
    folder = folder or STYLE["figure_dir"]
    os.makedirs(folder, exist_ok=True)
    fmt = fmt or STYLE["save_format"]
    path = os.path.join(folder, f"{name}.{fmt}")
    fig.savefig(path, **kw)
    return path


def symmetric_limits(a, q=99.5):
    """Symmetric colour limits (percentile, robust to spikes) for signed data."""
    v = np.nanpercentile(np.abs(np.asarray(a)), q)
    return -v, v


# ----------------------------------------------------------------------------
# Labels (math text works with and without LaTeX)
# ----------------------------------------------------------------------------
LABELS = {
    "z": r"$z$ (m)",
    "z/zi": r"$z/z_i$",
    "time": r"$t$ (s)",
    "U": r"$\langle u\rangle$ (m s$^{-1}$)",
    "V": r"$\langle v\rangle$ (m s$^{-1}$)",
    "W": r"$\langle w\rangle$ (m s$^{-1}$)",
    "M": r"$M=\sqrt{U^2+V^2}$ (m s$^{-1}$)",
    "wdir": r"wind direction ($^\circ$)",
    "T": r"$\langle\theta\rangle$ (K)",
    "P": r"$\langle p\rangle$ (m$^2$ s$^{-2}$)",
    "uu": r"$\langle u'u'\rangle$ (m$^2$ s$^{-2}$)",
    "vv": r"$\langle v'v'\rangle$ (m$^2$ s$^{-2}$)",
    "ww": r"$\langle w'w'\rangle$ (m$^2$ s$^{-2}$)",
    "uv": r"$\langle u'v'\rangle$ (m$^2$ s$^{-2}$)",
    "uw": r"$\langle u'w'\rangle$ (m$^2$ s$^{-2}$)",
    "vw": r"$\langle v'w'\rangle$ (m$^2$ s$^{-2}$)",
    "tt": r"$\langle\theta'\theta'\rangle$ (K$^2$)",
    "wt": r"$\langle w'\theta'\rangle$ (K m s$^{-1}$)",
    "ut": r"$\langle u'\theta'\rangle$ (K m s$^{-1}$)",
    "vt": r"$\langle v'\theta'\rangle$ (K m s$^{-1}$)",
    "tke": r"$k_{res}$ (m$^2$ s$^{-2}$)",
    "tau": r"$|\tau|$ (m$^2$ s$^{-2}$)",
    "nut": r"$\langle\nu_t\rangle$ (m$^2$ s$^{-1}$)",
    "alpha_t": r"$\langle\alpha_t\rangle$ (m$^2$ s$^{-1}$)",
    "Pr_sgs": r"$\langle\nu_t\rangle/\langle\alpha_t\rangle$",
    "Ri_g": r"$Ri_g$",
    "Ri_f": r"$Ri_f$",
    "N2": r"$N^2$ (s$^{-2}$)",
    "dUdz": r"$\partial U/\partial z$ (s$^{-1}$)",
    "dTdz": r"$\partial\Theta/\partial z$ (K m$^{-1}$)",
    "Sw": r"$S_w$",
    "St": r"$S_\theta$",
    "K_m": r"$K_m$ (m$^2$ s$^{-1}$)",
    "K_h": r"$K_h$ (m$^2$ s$^{-1}$)",
    "Pr_t": r"$Pr_t$",
    "ustar": r"$u_*$ (m s$^{-1}$)",
    "L": r"$L$ (m)",
    "zi": r"$z_i$ (m)",
    "wstar": r"$w_*$ (m s$^{-1}$)",
    "zi_L": r"$-z_i/L$",
    "wt_s": r"$\langle w'\theta'\rangle_s$ (K m s$^{-1}$)",
    "thetastar": r"$\theta_*$ (K)",
    "t_conv": r"$z_i/w_*$ (s)",
    "h_ekman": r"$u_*/|f|$ (m)",
    "tke_sgs": r"$e_{sgs}$ (m$^2$ s$^{-2}$)",
    "tke_res_frac": r"$k_{res}/(k_{res}+e_{sgs})$",
    "e": r"$e_{sgs}$ (m$^2$ s$^{-2}$)",
    "Ri_b": r"$Ri_b$",
    "Q_cross": r"cross-isobaric $Q_\perp$ (m$^2$ s$^{-1}$)",
    "Q_along": r"along-isobaric $Q_\parallel$ (m$^2$ s$^{-1}$)",
    "alpha": r"turning angle ($^\circ$)",
    "alpha_s": r"surface turning angle ($^\circ$)",
    "S": r"$S=|\partial\mathbf{U}/\partial z|$ (s$^{-1}$)",
    "dVdz": r"$\partial V/\partial z$ (s$^{-1}$)",
    "Su": r"$S_u$", "Sv": r"$S_v$", "Kw": r"$K_w$",
    "www": r"$\langle w'w'w'\rangle$ (m$^3$ s$^{-3}$)",
    "wtt": r"$\langle w'\theta'\theta'\rangle$ (K$^2$ m s$^{-1}$)",
    "wwt": r"$\langle w'w'\theta'\rangle$ (K m$^2$ s$^{-2}$)",
    "tke_tot": r"$k_{res}+e_{sgs}$ (m$^2$ s$^{-2}$)",
    "u": r"$u$ (m s$^{-1}$)", "v": r"$v$ (m s$^{-1}$)", "w": r"$w$ (m s$^{-1}$)",
    "t": r"$\theta$ (K)", "p": r"$p$ (m$^2$ s$^{-2}$)",
}


def add_scalar_labels(sym, name=None, units="a.u."):
    """Register axis labels of a passive scalar with one-letter symbol `sym`
    (mean, gradient, variance, fluxes): called by Run for every scalar found."""
    if sym == "t":
        return
    n = name or sym
    S = sym.upper()
    u = f" ({units})" if units else ""
    LABELS.update({
        S: rf"$\langle {n}\rangle${u}",
        f"d{S}dz": rf"$\partial\langle {n}\rangle/\partial z$ ({units} m$^{{-1}}$)",
        f"{sym}{sym}": rf"$\langle {n}'{n}'\rangle$ ({units}$^2$)",
        f"w{sym}": rf"$\langle w'{n}'\rangle$ ({units} m s$^{{-1}}$)",
        f"u{sym}": rf"$\langle u'{n}'\rangle$ ({units} m s$^{{-1}}$)",
        f"v{sym}": rf"$\langle v'{n}'\rangle$ ({units} m s$^{{-1}}$)",
        f"S{sym}": rf"$S_{{{n}}}$",
        f"alpha_{sym}": rf"$\langle\alpha_{{{n}}}\rangle$ (m$^2$ s$^{{-1}}$)",
    })


def label(var):
    """Axis label of a variable name (falls back to the name itself).

    Suffixes are understood: 'uw_sgs', 'uw_tot', 'tke_sgs' ...
    """
    if var in LABELS:
        return LABELS[var]
    for suf, tag in (("_sgs", "sgs"), ("_tot", "tot"), ("_res", "res")):
        if var.endswith(suf) and var[: -len(suf)] in LABELS:
            base = LABELS[var[: -len(suf)]]
            if "\\rangle" in base:
                return base.replace("\\rangle$", f"\\rangle_{{{tag}}}$", 1)
            return f"{base} [{tag}]"
    return var


apply()

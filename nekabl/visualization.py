"""
visualization.py -- Horizontal and vertical slices of field files.

    fld = nk.read_fields("output", rename={"s1": "e", "s3": "nut"})
    nk.plot_hslice(fld, "w", z=100)                 # last snapshot, x-y plane
    nk.plot_vslice(fld, "t", y=2500, fluct=True)    # x-z plane at y = 2500 m
    nk.plot_hslices(fld, "w", heights=[50, 300, 800])

Planes at arbitrary heights / positions are obtained by spectral (GLL)
interpolation, so z = 100 m means exactly 100 m. Signed fields (fluctuations,
w) get a diverging colour map centred at 0, the others a sequential one.
"""

import numpy as np

from . import fields
from . import plotstyle as ps
from . import sem


def _snapshot(fld, time):
    if "time" not in fld.dims:
        return fld
    if time is None:
        return fld.isel(time=-1)
    return fld.sel(time=time, method="nearest")


def _cmap_kw(data, symmetric, cmap, vmin, vmax):
    if symmetric is None:
        symmetric = float(np.nanmin(data)) < 0 < float(np.nanmax(data))
    if symmetric:
        lo, hi = ps.symmetric_limits(data)
        return dict(cmap=cmap or ps.STYLE["cmap_div"], vmin=vmin if vmin is not None else lo,
                    vmax=vmax if vmax is not None else hi)
    return dict(cmap=cmap or ps.STYLE["cmap"], vmin=vmin, vmax=vmax)


def _flabel(var, fluct):
    """Colour-bar label: the variable, or its fluctuation (theta' (K), w' (m/s) ...)."""
    lab = ps.label(var)
    if not fluct:
        return lab
    if lab.startswith("$") and "$" in lab[1:]:
        i = lab.index("$", 1)
        return lab[:i] + "'" + lab[i:]
    return f"{var}'"


def _title(var, fluct, where, t):
    name = f"{var}'" if fluct else var
    return f"{name}, {where}, t = {t:.0f} s"


def hslice(fld, var, z, time=None, fluct=False):
    """x-y plane of `var` at height z (DataArray (y, x))."""
    snap = _snapshot(fld, time)
    p = fields.at_height(snap[var], z).squeeze("z", drop=True)
    return p - fields.hmean(p) if fluct else p


def vslice(fld, var, y=None, x=None, time=None, fluct=False):
    """x-z plane at y (or y-z plane at x) of `var` (DataArray (z, x) or (z, y)).
    With fluct=True the horizontal mean profile is removed."""
    snap = _snapshot(fld, time)
    da = snap[var]
    if fluct:
        da = da - fields.hmean(da)
    if (x is None) == (y is None):
        raise ValueError("give exactly one of x or y")
    dim, val = ("y", y) if y is not None else ("x", x)
    ax = sem.axis_of(da, dim)
    return sem.apply_along(da, ax.interpolation([val]), dim, new_coord=[val]).squeeze(dim, drop=True)


def _panels(data, kind, ax):
    if ax is not None:
        return ax.figure, np.atleast_1d(ax)
    fig, axs = ps.subplots(1, len(data), kind=kind, sharey=True)
    if kind == "wide" and len(data) > 1:
        fig.set_size_inches(ps.STYLE["wide_width"] * min(len(data), 2), ps.STYLE["wide_height"] * 1.1)
    return fig, np.atleast_1d(axs)


def _time_of(fld, time):
    snap = _snapshot(fld, time)
    return float(snap.time) if "time" in snap.coords else snap.attrs.get("time", np.nan)


def plot_hslice(fld, var, z, time=None, fluct=False, labels=None, ax=None, cmap=None, vmin=None,
                vmax=None, symmetric=None, colorbar=True, scale=None):
    """Colour map of `var` on the horizontal plane at height z.

    fld   : field Dataset, or dict / list of them (one panel per simulation,
            shared colour scale).
    scale : divide the axes by this length (e.g. zi) -> axes in x/zi, y/zi.
    """
    data = ps.as_dict(fld, labels)
    fig, axs = _panels(data, "slice", ax)
    planes = {k: hslice(f, var, z, time, fluct) for k, f in data.items()}
    kw = _cmap_kw(np.concatenate([p.values.ravel() for p in planes.values()]), symmetric, cmap, vmin, vmax)
    s = scale or 1.0
    for a, (lab, p) in zip(axs, planes.items()):
        pc = a.pcolormesh(p.x / s, p.y / s, p, shading="gouraud", rasterized=True, **kw)
        a.set_aspect("equal")
        a.set_xlabel(r"$x/z_i$" if scale else r"$x$ (m)")
        a.set_title(_title(var, fluct, f"z = {z:.0f} m", _time_of(data[lab], time))
                    if len(data) == 1 else f"{lab}, z = {z:.0f} m")
        a.grid(False)
    axs[0].set_ylabel(r"$y/z_i$" if scale else r"$y$ (m)")
    if colorbar:
        fig.colorbar(pc, ax=list(axs), label=_flabel(var, fluct))
    return fig, (axs[0] if len(axs) == 1 else axs)


def plot_vslice(fld, var, y=None, x=None, time=None, fluct=False, labels=None, ax=None, cmap=None,
                vmin=None, vmax=None, symmetric=None, zmax=None, colorbar=True, aspect=None):
    """Colour map of `var` on a vertical plane (x-z at y, or y-z at x).
    `fld` can be a dict / list (one panel per simulation, shared colours).

    aspect : 'equal' for true geometry (default: automatic, ABL domains are flat).
    """
    data = ps.as_dict(fld, labels)
    fig, axs = _panels(data, "wide", ax)
    planes = {k: vslice(f, var, y=y, x=x, time=time, fluct=fluct) for k, f in data.items()}
    hdim = "x" if y is not None else "y"
    kw = _cmap_kw(np.concatenate([p.values.ravel() for p in planes.values()]), symmetric, cmap, vmin, vmax)
    where = f"y = {y:.0f} m" if y is not None else f"x = {x:.0f} m"
    for a, (lab, p) in zip(axs, planes.items()):
        pc = a.pcolormesh(p[hdim], p.z, p.transpose("z", hdim), shading="gouraud", rasterized=True, **kw)
        if aspect:
            a.set_aspect(aspect)
        if zmax:
            a.set_ylim(0, zmax)
        a.set_xlabel(f"${hdim}$ (m)")
        a.set_title(_title(var, fluct, where, _time_of(data[lab], time)) if len(data) == 1
                    else f"{lab}, {where}")
        a.grid(False)
    axs[0].set_ylabel(ps.label("z"))
    if colorbar:
        fig.colorbar(pc, ax=list(axs), label=_flabel(var, fluct))
    return fig, (axs[0] if len(axs) == 1 else axs)


def plot_hslices(fld, var, heights, time=None, fluct=False, cmap=None, symmetric=None, scale=None):
    """Horizontal slices at several heights side by side, one shared colour scale."""
    fig, axs = ps.subplots(1, len(heights), kind="slice", sharey=True)
    axs = np.atleast_1d(axs)
    planes = [hslice(fld, var, z, time, fluct) for z in heights]
    kw = _cmap_kw(np.concatenate([p.values.ravel() for p in planes]), symmetric, cmap, None, None)
    s = scale or 1.0
    for ax, p, z in zip(axs, planes, heights):
        pc = ax.pcolormesh(p.x / s, p.y / s, p, shading="gouraud", rasterized=True, **kw)
        ax.set_aspect("equal")
        ax.set_title(f"z = {z:.0f} m")
        ax.set_xlabel(r"$x/z_i$" if scale else r"$x$ (m)")
        ax.grid(False)
    axs[0].set_ylabel(r"$y/z_i$" if scale else r"$y$ (m)")
    fig.colorbar(pc, ax=list(axs), label=_flabel(var, fluct), shrink=0.8)
    return fig, axs

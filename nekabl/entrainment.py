"""
entrainment.py -- The inversion and the entrainment of momentum, heat and
passive scalars.

Profiles (statistics, dims (z) or (time, z), from Run.profile / Run.evolution)
--------------------------------------------------------------------------
    resolve_scalars(prof, scalars)    which scalars: None = all, or names / symbols
    inversion_layer(prof)             zi, entrainment-zone (EZ) bounds and depth
    entrainment_rates(prof, scalars)  per time: jumps, entrainment fluxes, w_e,
                                      entrainment ratio, SGS fraction, Ri_w, E
    entrainment_budget(prof, ent)     w_e from the growth of zi and F_e from the
                                      mixed-layer budget (independent estimates)
    entrainment_table(ent)            printable summary
    plot_inversion, plot_entrainment, plot_entrainment_budget

Fields (3D snapshots from Run.fields)
-------------------------------------
    inversion_height_map(fld, zi)     local inversion height h(x, y)
    inversion_stats(h)                mean, std, skewness of h
    interface_composite(fld, h)       means and fluxes in interface coordinates z - h
    quadrants_at_inversion(fld, zi)   quadrant analysis of w'c' at zi
    plot_inversion_map, plot_interface_composite

All the functions that involve scalars take ``scalars=None`` (all of them,
temperature included) or a name / symbol / list of them, e.g.
``scalars=["co2", "t"]``. The inversion itself is always defined by the
temperature (heat flux and gradient), whatever scalars are selected.

Definitions (see METHODS.md, section Entrainment)
-------------------------------------------------
    F(c)   total (resolved + SGS) vertical flux <w'c'>; momentum: <u'w'>, <v'w'>
    zi     height of the minimum of the heat flux ('flux') or the maximum of
           d<theta>/dz ('gradient')
    EZ     [z_b, z_t]: z_b = highest zero crossing of the heat flux below zi,
           z_t = lowest height above zi where |F_theta| < `frac` |F_theta(zi)|
    Delta  jump  c_ft(zi) - c_ml : free-atmosphere profile (linear fit over
           [z_t, z_t + ft_depth zi]) extrapolated to zi, minus the mixed-layer
           mean over [ml[0], ml[1]] zi
    F_e    total flux at zi (F_peak: extreme value inside the EZ)
    w_e    entrainment velocity  -F_e / Delta
    A      entrainment ratio  -F_e / F_s
    E      w_e / w*   (convective scaling),  Ri_w = (g/theta_0) Delta_theta zi / w*^2
"""

import numpy as np
import xarray as xr

from . import plotstyle as ps
from . import sem
from .diagnostics import _crossing, _extremum, _nodes, _over_time

_FLUID = "uvwp"


# ----------------------------------------------------------------------------
# Scalar selection
# ----------------------------------------------------------------------------
def _attrs(obj):
    return obj if isinstance(obj, dict) else obj.attrs


def resolve_scalars(obj, scalars=None):
    """Symbols of the requested scalars.

    obj     : profile Dataset (attrs 'scalars' = symbols, 'scalar_names' =
              {symbol: name}) or a dict with those two entries.
    scalars : None (all, temperature included), or a name / symbol / list of
              them (case-insensitive). A string made of several symbols
              ("tq") is also accepted.
    """
    a = _attrs(obj)
    avail = a.get("scalars", "t")
    names = {"t": "temperature", **a.get("scalar_names", {})}
    if scalars is None:
        return list(avail)
    if isinstance(scalars, str):
        scalars = [scalars]
    out = []
    for s in scalars:
        hit = [k for k in avail if k == str(s).lower() or names.get(k, "").lower() == str(s).lower()]
        if not hit and len(s) > 1 and all(c in avail for c in s):
            hit = list(s)
        if not hit and s.lower() in ("theta", "temp"):
            hit = ["t"]
        if not hit:
            raise KeyError(f"unknown scalar '{s}'; available: "
                           + ", ".join(f"{names.get(k, k)} ('{k}')" for k in avail))
        out += [h for h in hit if h not in out]
    return out


def _name(prof, sym):
    return {"t": "temperature", **prof.attrs.get("scalar_names", {})}.get(sym, sym)


def _flux_var(prof, c, total=True):
    """Name of the vertical flux of quantity c ('u', 'v' or a scalar symbol)."""
    base = f"{c}w" if c in "uv" else f"w{c}"
    if total and f"{base}_tot" in prof:
        return f"{base}_tot"
    return base if base in prof else None


def _q_label(prof, c):
    return {"u": "u", "v": "v"}.get(c, "\\theta" if c == "t" else _name(prof, c))


# ----------------------------------------------------------------------------
# Inversion layer
# ----------------------------------------------------------------------------
@_over_time
def inversion_layer(prof, method="flux", zmax=None, frac=0.05, sym="t"):
    """Height and depth of the inversion / entrainment zone (dict, or Dataset
    over time).

    method 'flux' (default): zi at the heat-flux minimum, EZ between the zero
           crossing below and the height above where |F| < frac |F(zi)|;
           falls back to 'gradient' if the heat flux is nowhere negative
           (stable / neutral).
    method 'gradient': zi at the maximum of d<theta>/dz, EZ = its full width
           at half maximum.
    Both sets are always returned: zi_flux, zi_grad, ez_flux, ez_grad (depths).
    """
    S = sym.upper()
    out = dict.fromkeys(("zi", "zb", "zt", "ez", "zi_flux", "zb_flux", "zt_flux", "ez_flux",
                         "zi_grad", "zb_grad", "zt_grad", "ez_grad", "Fmin"), np.nan)
    fv = _flux_var(prof, sym)
    if fv is not None:
        z, F = _nodes(prof[fv], zmax)
        if F.min() < 0:
            zi = _extremum(z, F, "min")
            Fm = float(np.interp(zi, z, F))
            i0 = int(np.argmin(abs(z - zi)))
            zb = _crossing(z[: i0 + 1][::-1], F[: i0 + 1][::-1], 0.0)
            zt = _crossing(z[i0:], F[i0:], frac * Fm)
            out.update(zi_flux=zi, zb_flux=zb, zt_flux=zt, ez_flux=zt - zb, Fmin=Fm)
    if f"d{S}dz" in prof:
        z, G = _nodes(prof[f"d{S}dz"], zmax)
        zg = _extremum(z, G, "max")
        i0 = int(np.argmin(abs(z - zg)))
        half = 0.5 * float(np.interp(zg, z, G))
        zb = _crossing(z[: i0 + 1][::-1], G[: i0 + 1][::-1], half)
        zt = _crossing(z[i0:], G[i0:], half)
        out.update(zi_grad=zg, zb_grad=zb, zt_grad=zt, ez_grad=zt - zb)
    use = "flux" if (method == "flux" and np.isfinite(out["zi_flux"])) else "grad"
    out.update(zi=out[f"zi_{use}"], zb=out[f"zb_{use}"], zt=out[f"zt_{use}"], ez=out[f"ez_{use}"])
    return out

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


# ----------------------------------------------------------------------------
# Entrainment
# ----------------------------------------------------------------------------
def _jump(z, C, zi, zt, ml, ft_depth):
    """(c_ml, c_ft(zi), gamma, c(z_t)) of a mean profile."""
    zz = np.linspace(ml[0] * zi, ml[1] * zi, 40)
    c_ml = float(np.mean(np.interp(zz, z, C)))
    z0 = zt if np.isfinite(zt) else 1.1 * zi
    sel = (z >= z0) & (z <= z0 + ft_depth * zi)
    if sel.sum() < 3:                      # near the domain top: use the highest nodes above z0
        sel = np.zeros(z.size, bool)
        above = np.where(z >= z0)[0]
        sel[above[-3:] if above.size >= 3 else above] = True
    if sel.sum() < 2:
        return c_ml, np.nan, np.nan, np.nan
    gamma, c0 = np.polyfit(z[sel], C[sel], 1)
    return c_ml, float(c0 + gamma * zi), float(gamma), float(np.interp(z0, z, C))


@_over_time
def _entrain(prof, quantities, method, zmax, frac, ml, ft_depth, min_jump):
    inv = inversion_layer(prof, method, zmax, frac)
    zi, zt = inv["zi"], inv["zt"]
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    out = dict(inv)
    if not np.isfinite(zi):
        return out
    z = prof.z.values
    keep = z <= zmax if zmax is not None else np.ones(z.size, bool)
    z = z[keep]
    for c in quantities:
        C = prof[c.upper()].values[keep]
        ftot, fres = _flux_var(prof, c), _flux_var(prof, c, total=False)
        c_ml, c_ft, gam, c_top = _jump(z, C, zi, zt, ml, ft_depth)
        D = c_ft - c_ml
        if not abs(D) > min_jump * (np.ptp(C) or 1.0):      # no jump (or no free-atmosphere data)
            D = np.nan
        out.update({f"Cml_{c}": c_ml, f"Cft_{c}": c_ft, f"gamma_{c}": gam,
                    f"Delta_{c}": D, f"Delta_top_{c}": c_top - c_ml})
        if ftot is None:
            continue
        F = prof[ftot].values[keep]
        Fe, Fs = float(np.interp(zi, z, F)), float(F[0])
        inside = (z >= inv["zb"]) & (z <= inv["zt"]) if np.isfinite(inv["zb"] + inv["zt"]) \
            else np.zeros(z.size, bool)
        Fpk = float(F[inside][np.argmax(abs(F[inside]))]) if inside.any() else np.nan
        out.update({f"Fs_{c}": Fs, f"Fe_{c}": Fe, f"Fpeak_{c}": Fpk,
                    f"we_{c}": -Fe / D if np.isfinite(D) else np.nan,
                    f"A_{c}": -Fe / Fs if abs(Fs) > 1e-12 else np.nan})
        if fres is not None and fres != ftot:
            Fr = float(np.interp(zi, z, prof[fres].values[keep]))
            out[f"sgs_{c}"] = 1.0 - Fr / Fe if abs(Fe) > 1e-30 else np.nan
    # scalings (convective and shear)
    ustar = float(prof["tau"].values[0]) ** 0.5 if "tau" in prof else np.nan
    Fs_t, Dt = out.get("Fs_t", np.nan), out.get("Delta_t", np.nan)
    ws = float(np.cbrt(g / T0 * Fs_t * zi)) if Fs_t > 0 else np.nan
    out.update(ustar=ustar, wstar=ws, zi_L=np.nan)
    if np.isfinite(Dt) and Dt > 0:
        out["Ri_w"] = g / T0 * Dt * zi / ws**2 if np.isfinite(ws) else np.nan
        out["Ri_u"] = g / T0 * Dt * zi / ustar**2 if ustar > 0 else np.nan
    for c in quantities:
        w_e = out.get(f"we_{c}", np.nan)
        out[f"E_{c}"] = w_e / ws if np.isfinite(ws) else np.nan
        if c != "t":
            out[f"we_ratio_{c}"] = w_e / out["we_t"] if np.isfinite(out.get("we_t", np.nan)) else np.nan
    out.pop("zi_L")
    return out


def entrainment_rates(prof, scalars=None, momentum=True, method="flux", zmax=None, frac=0.05,
                ml=(0.2, 0.7), ft_depth=0.3, min_jump=1e-3):
    """Entrainment diagnostics for each time of `prof` (dict if `prof` has no
    time dimension).

    scalars  : None (all scalars incl. temperature), or names / symbols.
    momentum : also the horizontal momentum components u and v.
    method   : 'flux' or 'gradient' (definition of zi and of the EZ).
    zmax     : ignore everything above (sponge layer).
    ml, ft_depth : mixed-layer averaging range (in zi) and depth (in zi) of the
               free-atmosphere fit above the EZ.
    min_jump : jumps smaller than this fraction of the profile range are 'no jump' (NaN).

    Variables (suffix _c: c = 'u', 'v', 't' or the scalar symbol):
      zi, zb, zt, ez          inversion height, EZ bounds / depth (m)
      zi_flux/grad, ez_flux/grad   both definitions
      Cml_c, Cft_c, Delta_c, gamma_c, Delta_top_c   mixed-layer value, free-atmosphere value
                              at zi, jump, free-atmosphere gradient, jump to z_t
      Fs_c, Fe_c, Fpeak_c     surface flux, flux at zi, extreme flux in the EZ
      we_c, A_c               entrainment velocity (m/s), -Fe/Fs
      sgs_c                   SGS fraction of Fe_c (resolution check)
      wstar, ustar, Ri_w, Ri_u, E_c (= we_c/w*), we_ratio_c (= we_c/we_t)
    """
    syms = resolve_scalars(prof, scalars)
    q = (["u", "v"] if momentum and "U" in prof and "V" in prof else []) + syms
    if "t" not in q:
        q.append("t")           # always needed for the inversion scalings
    res = _entrain(prof, tuple(q), method, zmax, frac, tuple(ml), ft_depth, min_jump)
    quantities = [c for c in q if c in ("u", "v") or c in syms or c == "t"]
    if isinstance(res, xr.Dataset):
        res.attrs.update(prof.attrs)
        res.attrs.update(quantities=quantities, selected=[s for s in syms], method=method,
                         scalars=prof.attrs.get("scalars", "t"))
    else:
        res = dict(res)
        res["quantities"] = quantities
    return res


def _window_tavg(ent):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN variables (e.g. no jump in U)
        return {k: float(np.nanmean(v)) for k, v in ent.data_vars.items()}


def entrainment_table(ent, momentum=True):
    """Text table of the time-averaged entrainment numbers (Dataset from `entrainment_rates`)."""
    if not isinstance(ent, xr.Dataset):
        ent = xr.Dataset({k: ("time", [v]) for k, v in ent.items() if np.isscalar(v)},
                         coords={"time": [0.0]})
    m = _window_tavg(ent)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = {k: float(np.nanstd(v)) for k, v in ent.data_vars.items()}
    names = {"t": "temperature", **ent.attrs.get("scalar_names", {})}
    n = ent.time.size
    L = [f"entrainment ({ent.attrs.get('method', 'flux')} method, mean +- std of {n} windows)",
         f"  zi = {m['zi']:.1f} m   EZ = [{m['zb']:.1f}, {m['zt']:.1f}] m  (depth {m['ez']:.1f} m = "
         f"{m['ez'] / m['zi'] if m['zi'] else np.nan:.2f} zi);  zi(flux) {m['zi_flux']:.1f}, zi(gradient) {m['zi_grad']:.1f} m, "
         f"EZ depth (gradient FWHM) {m['ez_grad']:.1f} m"]
    if np.isfinite(m.get("wstar", np.nan)):
        L.append(f"  w* = {m['wstar']:.3g} m/s  Ri_w = {m.get('Ri_w', np.nan):.3g}  "
                 f"Ri_u = {m.get('Ri_u', np.nan):.3g}")
    L.append(f"  {'quantity':14s} {'Delta':>10s} {'gamma':>10s} {'F_s':>10s} {'F_e':>10s} "
             f"{'w_e (m/s)':>12s} {'A=-Fe/Fs':>9s} {'E=we/w*':>9s} {'SGS %':>6s}")
    for c in ent.attrs.get("quantities", []):
        if c in "uv" and not momentum:
            continue
        g = lambda k: m.get(f"{k}_{c}", np.nan)  # noqa: E731
        L.append(f"  {names.get(c, c) if c not in 'uv' else 'U' if c == 'u' else 'V':14s} "
                 f"{g('Delta'):10.3g} {g('gamma'):10.3g} {g('Fs'):10.3g} {g('Fe'):10.3g} "
                 f"{g('we'):12.3g} {g('A'):9.3g} {g('E'):9.3g} {100 * g('sgs'):6.1f}"
                 + (f"   (+-{sd.get(f'we_{c}', np.nan):.2g})" if n > 1 else ""))
    return "\n".join(L)

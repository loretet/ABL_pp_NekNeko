"""
profiles.py -- From raw statistics to physical mean profiles.

Typical use
-----------
    raw  = nk.load_stats("output", case=case)          # raw moments (time, z)
    prof = nk.derive(nk.time_average(raw, 2000, 3000), case=case)   # (z)
    hist = nk.derive(raw, case=case)                   # (time, z): time-height

`time_average` does the window-weighted time average of the RAW moments
(correct way: central moments are built afterwards from the averaged raw ones,
so the variance includes the slow variability between windows).

`derive` builds everything that the available statistics allow:
    means          U V W P T, wind speed M, wind direction wdir
    gradients      dUdz dVdz dTdz S (shear) N2, Ri_g
    central moments (same letters as raw, now fluctuations):
                   uu vv ww uv uw vw  tke   ut vt wt tt   (+ uuw www wtt ... if 'full')
    SGS            nut alpha_t  uw_sgs vw_sgs wt_sgs ...  Pr_sgs
    totals         uw_tot vw_tot wt_tot  tau (total stress) Ri_f Ri_b
    transport      Q_along, Q_cross: cumulative mass transport int_0^z (u - u_g) dz
                   along / across the isobars (needs u_g, else the top wind is used)
    higher order   skewness Su Sv Sw St, kurtosis Kw (full sets)
Missing statistics are simply skipped.

`from_fields` builds the same raw moments from field snapshots (few samples,
but any quantity, e.g. the SGS TKE of the Deardorff model).

Plots (all return fig, ax(s)): plot_profiles, plot_fluxes, plot_time_height,
plot_hodograph, and plot_overview (all profiles of a group -- 'momentum',
'heat', 'scalar', 'stability' -- in one big figure).
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import plotstyle as ps
from . import sem

_FLUID_LETTERS = "uvwp"
_NOT_SCALARS = "uvwpek"          # single letters that are not transported scalars


def geostrophic_wind(prof, case=None):
    """(Ug, Vg, source): from case.u_geo (number = along x, or [ug, vg, ...]),
    else the mean wind at the top of the domain (free atmosphere)."""
    ug = getattr(case, "u_geo", None) if case is not None else None
    if ug is not None:
        if np.ndim(ug) == 0:
            return float(ug), 0.0, "case"
        return float(ug[0]), float(ug[1]) if len(ug) > 1 else 0.0, "case"
    top = prof.isel(z=-1)
    return top.U, top.V, "top of the domain"


def extra_scalars(raw, sym="t"):
    """Symbols of the scalars other than the buoyancy one found in raw stats."""
    found = [k for k in raw.data_vars if len(k) == 1 and k not in _NOT_SCALARS and k != sym]
    if "s" in found:     # 's' -> 'S' would clash with the shear S
        print("[nekabl] scalar symbol 's' clashes with the shear 'S': use another letter (e.g. 'c')")
        found.remove("s")
    return "".join(found)


# ----------------------------------------------------------------------------
# Time averaging
# ----------------------------------------------------------------------------
def time_average(raw, t0=None, t1=None, moments="window"):
    """Window-weighted time average of raw statistics over [t0, t1].

    Only windows lying entirely inside [t0, t1] are used. Works also on raw
    moments from field snapshots (equal weights).

    moments : 'window' (default) -> fluctuations are taken about the mean of
              EACH statistics window (horizontal + window average), and the
              resulting central moments are averaged. The returned raw
              moments are rebuilt so that derive() gives exactly these
              averaged central moments. A slow drift of the mean during the
              averaging time (e.g. the inertial oscillation of the wind)
              is then NOT counted as turbulence.
              'global' -> raw moments averaged first, fluctuations about the
              mean of the whole averaging time (stationary flow; the drift of
              the mean adds to variances and higher moments).
              Fluxes such as <u'w'> are hardly affected (<w> ~ 0).
    """
    if "time" not in raw.dims:
        return raw
    t = raw.time.values
    win = raw["window"].values if "window" in raw.coords else np.ones_like(t)
    start = t - win
    tol = 1e-6 * max(abs(t).max(), 1.0)
    sel = np.ones(t.size, bool)
    if t0 is not None:
        sel &= start >= t0 - tol
    if t1 is not None:
        sel &= t <= t1 + tol
    if not sel.any():
        raise ValueError(f"no statistics window inside [{t0}, {t1}]; windows end at {t}")
    sub = raw.isel(time=np.where(sel)[0])
    w = xr.DataArray(win[sel] / win[sel].sum(), dims="time")
    out = (sub * w).sum("time", keep_attrs=True)
    if moments == "window" and sub.time.size > 1:
        out = _window_moments(sub, w, out)
    elif moments not in ("window", "global"):
        raise ValueError("moments must be 'window' or 'global'")
    out = out.drop_vars([c for c in ("window",) if c in out.coords])
    out.attrs.update(t0=float(start[sel][0]), t1=float(t[sel][-1]), n_windows=int(sel.sum()),
                     averaging_time=float(win[sel].sum()), moments=moments,
                     window_min=float(win[sel].min()), window_max=float(win[sel].max()))
    out.z.attrs.update(raw.z.attrs)
    return out


def _window_moments(sub, w, out):
    """Rebuild the averaged raw moments in `out` so that central moments taken
    about the mean of the whole period equal the window-average of the
    central moments of each window (see time_average, moments='window').

    Pure products <abc..>:  raw = sum over subsets A of the factors of
        avg_w(C_w(A)) * prod(means of the other factors)   (C(single) = 0)
    Gradient terms <a db/dz>:  avg_w(<a db/dz>_w - A_w dB_w/dz) + A dB/dz
    e_ab = <da/dx_k db/dx_k>:   avg_w(e_w - dA_w dB_w)   + dA dB."""
    letters = "".join(k for k in sub.data_vars if len(k) == 1)
    tavg = lambda da: (da * w).sum("time")  # noqa: E731
    cbar = {}

    def cavg(key):           # window-averaged central moment of sorted letters `key`
        if key not in cbar:
            cm = central_moment(sub, key, letters) if len(key) > 1 else None
            cbar[key] = None if cm is None else tavg(cm)
        return cbar[key]

    res = out.copy()
    for name in sub.data_vars:
        if len(name) < 2 or not set(name) <= set(letters) or "d" in name:
            continue
        n, total, ok = len(name), 0.0, True
        for r in range(n + 1):
            for idx in itertools.combinations(range(n), r):
                if r == 1:
                    continue                       # <a'> = 0
                term = 1.0
                if r >= 2:
                    c = cavg("".join(sorted(name[i] for i in idx)))
                    if c is None:
                        ok = False
                        break
                    term = c
                for i in range(n):
                    if i not in idx:
                        term = term * out[name[i]]
                total = total + term
            if not ok:
                break
        if ok:
            res[name] = total
    # terms with mean gradients (only d/dz of the means is non-zero)
    try:
        D = sem.axis_of(sub, "z").derivative()
    except Exception:  # noqa: BLE001 - no GLL axis (e.g. uniform field grid): leave as is
        return res
    ddz = lambda da: sem.apply_along(da, D, "z")  # noqa: E731
    grad_w = {c: ddz(sub[c]) for c in letters}
    grad = {c: ddz(out[c]) for c in letters}
    edigit = {"1": "u", "2": "v", "3": "w"}
    for name in sub.data_vars:
        if len(name) == 5 and name[1] == "d" and name[3] == "d" and name[4] == "z":
            a, b = name[0], name[2]
            if a in letters and b in letters:
                res[name] = tavg(sub[name] - sub[a] * grad_w[b]) + out[a] * grad[b]
        elif name.startswith("e") and len(name) == 3:
            a, b = edigit.get(name[1], name[1]), edigit.get(name[2], name[2])
            if a in letters and b in letters:
                res[name] = tavg(sub[name] - grad_w[a] * grad_w[b]) + grad[a] * grad[b]
    return res


# ----------------------------------------------------------------------------
# Central moments
# ----------------------------------------------------------------------------
def _lookup(raw, letters):
    """Dict: sorted letters -> raw variable name, for pure-product moments."""
    table = {}
    for name in raw.data_vars:
        if set(name) <= set(letters) and "d" not in name:
            table["".join(sorted(name))] = name
    return table


def central_moment(raw, name, letters):
    """<a'b'c'...> from the raw moments (inclusion-exclusion over subsets).

    Returns None when a needed raw moment is missing.
    """
    table = _lookup(raw, letters)
    n = len(name)
    out = 0.0
    for r in range(n + 1):
        for sub in itertools.combinations(range(n), r):
            rest = [name[i] for i in range(n) if i not in sub]
            key = "".join(sorted(name[i] for i in sub))
            term = 1.0 if not sub else raw[table[key]] if key in table else None
            if term is None:
                return None
            for c in rest:
                if c not in table:
                    return None
                term = term * raw[table[c]]
            out = out + (-1) ** (n - r) * term
    return out


# ----------------------------------------------------------------------------
# Derived profiles
# ----------------------------------------------------------------------------
def _params(case, g, T0):
    if case is not None:
        g = g if g is not None else case.g
        T0 = T0 if T0 is not None else case.T0
    return (9.81 if g is None else g), (300.0 if T0 is None else T0)


def derive(raw, case=None, g=None, T0=None, sym="t", nu=None):
    """Physical profiles from raw statistics (see module docstring).

    raw  : raw Dataset, dims (z) or (time, z).
    case : CaseInfo (read_case) providing g and theta_0; or give g, T0.
    sym  : symbol of the buoyancy scalar (potential temperature).
    nu   : molecular viscosity for the resolved dissipation (optional).
    """
    g, T0 = _params(case, g, T0)
    if case is not None and nu is None:
        nu = case.nu
    ax = sem.axis_of(raw, "z")
    D = ax.derivative()

    def ddz(da):
        return sem.apply_along(da, D, "z")

    out = xr.Dataset(coords=raw.coords)
    out.attrs.update(raw.attrs)
    out.attrs.update(g=g, T0=T0, sym=sym)
    if nu is not None:
        out.attrs["nu"] = nu
    has = lambda *names: all(n in raw for n in names)  # noqa: E731
    S = sym.upper()
    letters = _FLUID_LETTERS + sym + extra_scalars(raw, sym)
    out.attrs["scalars"] = sym + extra_scalars(raw, sym)

    # --- means and gradients -------------------------------------------------
    for c in letters:
        if c in raw:
            out[c.upper()] = raw[c]
            out[f"d{c.upper()}dz"] = ddz(raw[c])
    if has("u", "v"):
        out["M"] = np.hypot(raw.u, raw.v)
        out["wdir"] = (270.0 - np.rad2deg(np.arctan2(raw.v, raw.u))) % 360.0
        out["S"] = np.hypot(out.dUdz, out.dVdz)
    if S in out:
        out["N2"] = g / T0 * out[f"d{S}dz"]
        if "S" in out:
            out["Ri_g"] = out.N2 / (out.S**2).where(out.S > 0.01 * out.S.max("z"))
        if "M" in out:   # bulk Ri between the lowest grid point and z
            dz = out.z - out.z.isel(z=0)
            out["Ri_b"] = (g / T0 * (out[S] - out[S].isel(z=0)) * dz
                           / (out.M**2).where(out.M > 1e-3)).where(dz > 0, 0.0)
    if has("u", "v"):    # cumulative mass transport along / across the isobars
        ug, vg, src = geostrophic_wind(out, case)
        G = np.hypot(ug, vg)
        ca, sa = ug / G, vg / G                       # unit vector along u_g
        du, dv = out.U - ug, out.V - vg
        out["Q_along"] = (du * ca + dv * sa).cumulative_integrate("z")
        out["Q_cross"] = (-du * sa + dv * ca).cumulative_integrate("z")   # 90 deg left of u_g
        out.attrs["geostrophic"] = src

    # --- central moments of products (uu, uw, wt, uuw, wtt, tttt, ...) -----
    for name in raw.data_vars:
        if len(name) >= 2 and set(name) <= set(letters) and "d" not in name:
            cm = central_moment(raw, name, letters)
            if cm is not None:
                out[name] = cm

    # --- terms with derivatives: <a db/dx_k>, e_ab --------------------------
    mean_grad = {c: out[f"d{c.upper()}dz"] for c in letters if f"d{c.upper()}dz" in out}
    for name in raw.data_vars:
        if len(name) == 5 and name[1] == "d" and name[3] == "d":     # e.g. pdudz, wdtdz
            a, b, k = name[0], name[2], name[4]
            val = raw[name]
            if k == "z" and a in raw and b in mean_grad:
                val = val - raw[a] * mean_grad[b]
            out[name] = val
    edigit = {"1": "u", "2": "v", "3": "w"}
    for name in raw.data_vars:
        if name.startswith("e") and len(name) == 3:
            a, b = (edigit.get(name[1], name[1]), edigit.get(name[2], name[2]))
            if a in mean_grad and b in mean_grad:
                out[name] = raw[name] - mean_grad[a] * mean_grad[b]
    if nu is not None and has("e11", "e22", "e33"):
        out["eps_res"] = nu * (out.e11 + out.e22 + out.e33)

    # --- TKE, stresses, skewness --------------------------------------------
    if all(k in out for k in ("uu", "vv", "ww")):
        out["tke"] = 0.5 * (out.uu + out.vv + out.ww)
    for c in "uvw" + sym:
        if c * 3 in out and c * 2 in out:
            out[f"S{c}"] = out[c * 3] / out[c * 2] ** 1.5
        if c * 4 in out and c * 2 in out:
            out[f"K{c}"] = out[c * 4] / out[c * 2] ** 2

    # --- SGS and totals --------------------------------------------------------
    for k in raw.data_vars:
        if k in ("taux", "tauy", "tauz"):           # Nek5000 KTH wall stress
            out[k] = raw[k]
        if k.endswith("_sgs") or k == "nut" or k.startswith("alpha_"):
            out[k] = raw[k]
            base = k[:-4]
            if k.endswith("_sgs") and base in out:
                out[base + "_tot"] = out[base] + raw[k]
    if "nut" in out and f"alpha_{sym}" in out:
        a = out[f"alpha_{sym}"]      # masked where alpha_t < 1 % of its maximum (above the BL)
        out["Pr_sgs"] = out.nut / a.where(a > 0.01 * a.max("z"))
    if "uw_tot" in out and "vw_tot" in out:
        out["tau"] = np.hypot(out.uw_tot, out.vw_tot)
    if "uw" in out and "vw" in out:
        out["tau_res"] = np.hypot(out.uw, out.vw)
    if "uw_sgs" in out and "vw_sgs" in out:
        out["tau_sgs"] = np.hypot(out.uw_sgs, out.vw_sgs)
    wt = f"w{sym}_tot" if f"w{sym}_tot" in out else (f"w{sym}" if f"w{sym}" in out else None)
    uw = "uw_tot" if "uw_tot" in out else ("uw" if "uw" in out else None)
    if wt and uw and "dUdz" in out:
        vw = uw.replace("u", "v", 1)
        prod = -(out[uw] * out.dUdz + out[vw] * out.dVdz)
        # masked where the shear production is < 1 % of its maximum (above the BL)
        big = abs(prod) > 0.01 * abs(prod).max("z")
        out["Ri_f"] = (g / T0 * out[wt]) / (-prod).where(big)

    # --- SGS TKE (Deardorff 'e' from fields or a TKE scalar) -----------------
    for e in ("e", "k"):
        if e in raw.data_vars and "tke" in out:
            out["tke_sgs"] = raw[e]
            out["tke_res_frac"] = out.tke / (out.tke + raw[e])
            break
    out.z.attrs.update(raw.z.attrs)
    return out

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

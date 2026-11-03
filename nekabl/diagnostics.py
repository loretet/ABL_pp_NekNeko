"""
diagnostics.py -- Bulk ABL diagnostics from derived profiles (see profiles.derive).

All functions accept a derived Dataset with dims (z) or (time, z): with a time
dimension they return time series (one value per statistics window).

Scales and heights
    abl_height(prof, method)      'flux' | 'gradient' | 'stress' | 'tke' | 'richardson'
    scales(prof, case)            u*, <w'theta'>_s, theta*, L, zi, w*, -zi/L, ...
    timeseries(raw, case)         scales for every statistics window
Surface layer / wall model
    similarity(prof, sc, case)    phi_m, phi_h from the LES vs MOST
    wall_model_check(prof, case)  sampling height, u* LES vs MOST, ...
SGS / resolution
    sgs_fraction(prof)            SGS part of stress and heat flux, Pr_sgs
    resolution(prof, zi)          grid points in the BL, first-point height ...
Time step
    cfl_dt(fld)                   largest stable dt for a snapshot
    plot_timestep(steps)          dt, CFL and solver iterations from the log

References
    Sullivan et al. (1998, JAS) -- zi from max potential temperature gradient
    Kosovic & Curry (2000, JAS) -- zi from the 5 % surface stress height
    Vogelezang & Holtslag (1996, BLM) -- bulk Richardson number height
    Brasseur & Wei (2010, PoF) -- log-layer mismatch and phi_m overshoot
    Kawai & Larsson (2012, PoF) -- wall-model sampling height
    Pope (2000) -- resolved TKE fraction >~ 80 %
"""

import functools
import re

import numpy as np
import xarray as xr

from . import most
from . import plotstyle as ps
from . import sem


def _over_time(func):
    """Apply a single-profile function to every time of a (time, z) Dataset."""

    @functools.wraps(func)
    def wrapper(prof, *args, **kw):
        if "time" not in prof.dims:
            return func(prof, *args, **kw)
        res = [func(prof.isel(time=i), *args, **kw) for i in range(prof.time.size)]
        if isinstance(res[0], dict):
            return xr.Dataset({k: ("time", [r[k] for r in res]) for k in res[0]},
                              coords={"time": prof.time})
        return xr.DataArray(res, dims="time", coords={"time": prof.time})

    return wrapper


def _nodes(da, zmax=None):
    """Profile values on the grid points (optionally only z <= zmax)."""
    z, f = da.z.values, da.values
    keep = z <= zmax if zmax is not None else np.ones(z.size, bool)
    return z[keep], f[keep]


def _extremum(z, f, kind="min"):
    """Location of the min/max of f(z): grid point refined by a parabola through
    its two neighbours (robust: no interpolation overshoots on coarse grids)."""
    i = int(np.argmin(f) if kind == "min" else np.argmax(f))
    if 0 < i < z.size - 1:
        z0, z1, z2 = z[i - 1:i + 2]
        f0, f1, f2 = f[i - 1:i + 2]
        den = (z0 - z1) * (z0 - z2) * (z1 - z2)
        A = (z2 * (f1 - f0) + z1 * (f0 - f2) + z0 * (f2 - f1)) / den
        B = (z2**2 * (f0 - f1) + z1**2 * (f2 - f0) + z0**2 * (f1 - f2)) / den
        if A != 0:
            return float(np.clip(-B / (2 * A), z0, z2))
    return float(z[i])


def _crossing(z, f, level, start=0):
    """First height above index `start` where f crosses `level` (linear interp)."""
    d = f[start:] - level
    idx = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    if not idx.size:
        return np.nan
    k = start + idx[0]
    return float(z[k] + (level - f[k]) * (z[k + 1] - z[k]) / (f[k + 1] - f[k]))


def _surface(prof, var):
    return float(prof[var].isel(z=0))


# ----------------------------------------------------------------------------
# ABL height
# ----------------------------------------------------------------------------
@_over_time
def abl_height(prof, method="flux", zmax=None, frac=0.05, Ri_c=0.25, sym="t"):
    """Boundary-layer height (m).

    method
      'flux'       height of the minimum of the total heat flux (convective BL)
      'gradient'   height of the maximum of dTheta/dz (Sullivan et al. 1998)
      'stress'     height where the total stress is `frac` of its surface value,
                   divided by (1 - frac) (Kosovic & Curry 2000; neutral/stable BL)
      'tke'        height where the TKE falls to `frac` of its maximum
      'richardson' first height where the bulk Richardson number exceeds Ri_c
    zmax : ignore everything above (e.g. the sponge layer).
    """
    S = sym.upper()
    if method == "flux":
        var = f"w{sym}_tot" if f"w{sym}_tot" in prof else f"w{sym}"
        return _extremum(*_nodes(prof[var], zmax), "min")
    if method == "gradient":
        return _extremum(*_nodes(prof[f"d{S}dz"], zmax), "max")
    if method == "stress":
        z, f = _nodes(prof["tau" if "tau" in prof else "tau_res"], zmax)
        return _crossing(z, f, frac * f[0]) / (1 - frac)
    if method == "tke":
        k = prof.tke + prof.tke_sgs if "tke_sgs" in prof else prof.tke
        z, f = _nodes(k, zmax)
        imax = int(np.argmax(f))
        return _crossing(z, f, frac * f[imax], start=imax)
    if method == "richardson":
        z, T = _nodes(prof[S], zmax)
        U, V = prof.U.values[: z.size], prof.V.values[: z.size]
        g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
        Rib = g / T0 * (T - T[0]) * (z - z[0]) / np.maximum(U**2 + V**2, 1e-6)
        return _crossing(z, Rib, Ri_c, start=1)
    raise ValueError(f"unknown method {method}")


def abl_heights(prof, zmax=None, sym="t"):
    """All the zi definitions at once (dict, or Dataset over time)."""
    out = {}
    for m in ("flux", "gradient", "stress", "tke", "richardson"):
        try:
            out[m] = abl_height(prof, m, zmax=zmax, sym=sym)
        except KeyError:
            pass
    return xr.Dataset(out) if "time" in prof.dims else out


# ----------------------------------------------------------------------------
# Scales
# ----------------------------------------------------------------------------
@_over_time
def scales(prof, case=None, zi_method="auto", zmax=None, kappa=None, sym="t", zi=None):
    """Surface-layer and mixed-layer scales (dict, or Dataset over time).

    ustar      total stress at the surface, (uw_tot^2 + vw_tot^2)^(1/4)
    wt_s       total heat flux at the surface
    thetastar  -wt_s / u*
    L          Obukhov length
    zi         ABL height (method `zi_method`, or the value `zi`); 'auto' uses
               'flux' (heat-flux minimum) if the surface heat flux is positive
               and 'stress' (5 % of the surface stress) otherwise
    wstar      Deardorff velocity (g/theta0 wt_s zi)^(1/3) (0 if stable)
    zi_L       -zi/L  (> ~10: convective, ~0 neutral, < 0 stable)
    t_conv     large-eddy turnover time zi / w*
    h_ekman    u* / |f| (neutral BL height scale), if f is known
    """
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    kappa = kappa or (case.kappa if case is not None and case.kappa else 0.4)
    wt = f"w{sym}_tot" if f"w{sym}_tot" in prof else f"w{sym}"
    us = float(_surface(prof, "tau") ** 0.5) if "tau" in prof else np.nan
    wts = _surface(prof, wt) if wt in prof else (case.surface_flux if case else np.nan)
    L = float(most.obukhov_length(us, wts, T0, g, kappa))
    if zi_method == "auto":
        zi_method = "flux" if wts > 0 else "stress"
    h = float(zi) if zi is not None else abl_height(prof, zi_method, zmax=zmax, sym=sym)
    ws = float(np.cbrt(g / T0 * wts * h)) if wts > 0 else 0.0
    out = dict(ustar=us, wt_s=wts, thetastar=-wts / us if us > 0 else np.nan, L=L, zi=h,
               wstar=ws, zi_L=-h / L if np.isfinite(L) and L != 0 else 0.0,
               t_conv=h / ws if ws > 0 else np.nan)
    if "taux" in prof and "tauy" in prof:      # wall stress written by Nek5000 (KTH)
        out["ustar_wall"] = float(np.hypot(_surface(prof, "taux"), _surface(prof, "tauy")) ** 0.5)
    if case is not None and case.f:
        out["h_ekman"] = us / abs(case.f)
    return out

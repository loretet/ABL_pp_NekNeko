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


def timeseries(raw, case=None, zi_method="auto", zmax=None, sym="t"):
    """`scales` for every statistics window (raw = load_stats output)."""
    from .profiles import derive

    return scales(derive(raw, case=case, sym=sym), case, zi_method=zi_method, zmax=zmax, sym=sym)


def plot_timeseries(ts, variables=("ustar", "wt_s", "zi", "zi_L"), labels=None, axs=None, **kw):
    """Stacked time series of scalar diagnostics (one panel per variable);
    `ts` can be a dict / list of time series (one colour per simulation)."""
    variables = [variables] if isinstance(variables, str) else list(variables)
    data = ps.as_dict(ts, labels)
    if axs is None:
        fig, axs = ps.subplots(len(variables), 1, kind="wide", sharex=True, sharey=False,
                               figsize=(ps.STYLE["wide_width"], 1.5 * len(variables) + 0.5))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    for i, (lab, d) in enumerate(data.items()):
        for ax, v in zip(axs, variables):
            if v in d:
                ax.plot(d.time, d[v], marker="o", ms=3, color=ps.color(i), label=lab, **kw)
    for ax, v in zip(axs, variables):
        ax.set_ylabel(ps.label(v))
        if v in ("L", "zi_L"):     # |L| blows up near neutral: robust range, outliers off-axis
            vals = np.concatenate([np.asarray(d[v], float) for d in data.values() if v in d])
            vals = vals[np.isfinite(vals)]
            if vals.size > 4:
                lo, hi = np.percentile(vals, [10, 95])
                pad = 0.25 * (hi - lo) if hi > lo else 1.0
                if vals.min() < lo - pad or vals.max() > hi + pad:
                    ax.set_ylim(lo - pad, hi + pad)
    axs[-1].set_xlabel(ps.label("time"))
    if len(data) > 1:
        axs[0].legend()
    return fig, (axs[0] if len(axs) == 1 else axs)


# ----------------------------------------------------------------------------
# Surface layer similarity and the wall model
# ----------------------------------------------------------------------------
def similarity(prof, sc, case=None, sym="t"):
    """Monin-Obukhov check: phi_m = kappa z/u* dM/dz, phi_h = kappa z/theta* dTheta/dz
    from the LES, and the MOST values at the same zeta = z/L.

    sc : output of `scales` (single profile).
    """
    kappa = case.kappa if case is not None and case.kappa else 0.4
    Pr = case.Pr if case is not None and case.Pr else 1.0
    p = prof.isel(z=slice(1, None))
    z = p.z
    zeta = z / sc["L"] if np.isfinite(sc["L"]) else 0 * z
    out = xr.Dataset(coords={"z": z})
    out["zeta"] = zeta
    out["phi_m"] = kappa * z / sc["ustar"] * p.S
    out["phi_m_most"] = ("z", most.phi_m(zeta.values))
    S = sym.upper()
    if f"d{S}dz" in p and np.isfinite(sc["thetastar"]):
        out["phi_h"] = kappa * z / sc["thetastar"] * p[f"d{S}dz"]
        out["phi_h_most"] = ("z", most.phi_h(zeta.values, Pr))
    out.attrs.update({k: prof.attrs[k] for k in ("t0", "t1", "n_windows", "window_min", "window_max")
                      if k in prof.attrs})
    out.attrs.update(zi=sc["zi"], L=sc["L"], ustar=sc["ustar"], Pr=Pr)
    return out


def similarity_evolution(evo, case=None, sym="t", **scale_kw):
    """similarity() of every profile of a (time, z) Dataset, each with its own
    u*, theta*, L, zi (for plot_similarity(..., evolution=))."""
    sc = scales(evo, case, **scale_kw)
    out = [similarity(evo.isel(time=i), {k: float(sc[k][i]) for k in sc.data_vars}, case, sym)
           for i in range(evo.time.size)]
    res = xr.concat(out, dim="time", coords="minimal", compat="override").assign_coords(time=evo.time)
    res["zi"] = ("time", [o.attrs["zi"] for o in out])
    return res


def plot_similarity(sim, zmax_zi=0.3, h=None, labels=None, axs=None, evolution=None, title=None):
    """phi_m and phi_h from the LES (solid + markers) against MOST at the same
    z/L (dashed), versus z/zi in the surface layer. `sim` can be a dict / list
    (one colour per simulation). `h` = wall-model sampling height (m) of the
    first simulation (dotted line). evolution: similarity_evolution(...) of the
    windows, thin lines coloured by time (each with its own zi); title: see
    profiles.plot_profiles (uses the attrs of the mean profile if present)."""
    data = ps.as_dict(sim, labels)
    from .profiles import _evolution
    evo = _evolution(data, evolution)
    created = axs is None
    first = next(iter(data.values()))
    keys = [k for k in ("phi_m", "phi_h") if k in first]
    if axs is None:
        fig, axs = ps.subplots(1, len(keys))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    for i, (lab, sm) in enumerate(data.items()):
        zi, L = sm.attrs["zi"], sm.attrs["L"]
        s = sm.where(sm.z <= zmax_zi * zi, drop=True)
        zf = np.linspace(float(sm.z[0]) * 0.5, zmax_zi * zi, 200)
        zeta = zf / L if np.isfinite(L) else 0 * zf
        curves = dict(phi_m=most.phi_m(zeta), phi_h=most.phi_h(zeta, sm.attrs.get("Pr", 1.0)))
        for ax, k in zip(axs, keys):
            if k in s:
                if evo is not None and k in evo:
                    ps.evolution_lines(ax, evo.where(evo.z <= zmax_zi * evo.zi, drop=True),
                                       lambda d, k=k: d[k], lambda d: d.z / d.zi)
                c = ps.mean_color(i, evo)
                ax.plot(s[k], s.z / zi, "o", ls=ps.ls("total"), ms=3, color=c, label=lab, zorder=3)
                ax.plot(curves[k], zf / zi, ls=ps.ls("ref"), color=c, lw=1, zorder=3)
    for ax, k in zip(axs, keys):
        ax.axhspan(0, 0.1, color="0.93", zorder=0)
        if h is not None:
            ax.axhline(h / first.attrs["zi"], color="0.3", ls=":", lw=1)
        ax.set_xlabel(r"$\phi_m$" if k == "phi_m" else r"$\phi_h$")
        ps.zlabel(ax, True)
    axs[0].set_ylim(0, zmax_zi)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="0.3", marker="o", ms=3, ls=ps.ls("total"), label="LES"),
          Line2D([], [], color="0.3", ls=ps.ls("ref"), label="MOST")]
    if h is not None:
        hs.append(Line2D([], [], color="0.3", ls=":", label=r"$h_{wm}$"))
    leg = axs[0].legend(handles=hs, loc="upper left")
    axs[0].add_artist(leg)
    if len(data) > 1:
        ps.run_legend(axs[-1], list(data), loc="upper right")
    ps.evolution_colorbar(fig, axs, evo)
    if title is not False and (title or created):
        ps.average_title(fig, data, evo, title)
    return fig, (axs[0] if len(axs) == 1 else axs)


def wall_model_check(prof, case, sc=None, verbose=True):
    """Wall-model diagnostics (single, time-averaged profile).

    * sampling height h (GLL point `h_index` above the wall) relative to z0,
      zi and L -- h should be in the surface layer (h < ~0.1 zi) and a few
      points off the wall (Kawai & Larsson 2012)
    * u* from the LES total surface stress vs u* from MOST applied to the
      *mean* wind at h (the wall model works on instantaneous local values,
      so a few % difference is expected; large ones point at problems)
    * the LES wind vs the MOST log/stability profile at the first points
      (log-layer mismatch, Brasseur & Wei 2010)
    """
    sc = sc or scales(prof, case)
    hi = case.h_index if case is not None and case.h_index is not None else 1
    h_dist = getattr(case, "h", None)      # sampling distance (Nek5000 WMLES), else the GLL point
    h = float(h_dist) if h_dist else float(prof.z[hi])
    z0, kappa = case.z0, case.kappa or 0.4
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    if h_dist:
        Mh = float(sem.interp(prof.M, np.array([h]))[0])
        hi = int((prof.z.values < h).sum())
    else:
        Mh = float(prof.M[hi])
    us_most, L_most = most.solve_ustar(Mh, h, z0, sc["wt_s"], T0, g, kappa)
    zz = prof.z.values[1:5]
    M_most = most.wind_profile(zz, sc["ustar"], sc["L"], z0, kappa)
    res = dict(h=h, h_index=hi, h_over_z0=h / z0, h_over_zi=h / sc["zi"],
               h_over_L=h / sc["L"] if np.isfinite(sc["L"]) else 0.0,
               M_h=Mh, ustar_les=sc["ustar"], ustar_most_mean=us_most,
               ustar_ratio=sc["ustar"] / us_most, L_les=sc["L"], L_most_mean=L_most,
               points_below_h=hi, points_in_surface_layer=int(((prof.z > 0) & (prof.z <= 0.1 * sc["zi"])).sum()),
               M_les_first=prof.M.values[1:5], M_most_first=M_most, z_first=zz)
    if verbose:
        print("---- wall model check ------------------------------------")
        print(f" model {case.wall_model}, z0 = {z0} m, kappa = {kappa}")
        print(f" sampling height h = z[{hi}] = {h:.2f} m  (h/z0 = {h / z0:.0f}, "
              f"h/zi = {res['h_over_zi']:.3f}, h/L = {res['h_over_L']:.3f})")
        print(f" grid points in the surface layer (z < 0.1 zi): {res['points_in_surface_layer']}")
        print(f" u* LES (total surface stress) = {sc['ustar']:.4f} m/s")
        print(f" u* MOST from mean M(h)={Mh:.3f} m/s = {us_most:.4f} m/s  "
              f"(ratio LES/MOST = {res['ustar_ratio']:.3f})")
        print(f" L  LES = {sc['L']:.1f} m,  L MOST(mean) = {L_most:.1f} m")
        for z_, a, b in zip(zz, res["M_les_first"], M_most):
            print(f"   z = {z_:8.2f} m   M_LES = {a:6.3f}   M_MOST = {b:6.3f}   "
                  f"diff = {100 * (a - b) / b:+5.1f} %")
        if res["h_over_zi"] > 0.15:
            print(" ! h is above ~0.1-0.15 zi: the sampling point is not in the surface layer")
        print("-----------------------------------------------------------")
    return res

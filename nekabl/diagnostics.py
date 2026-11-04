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


# ----------------------------------------------------------------------------
# SGS and resolution
# ----------------------------------------------------------------------------
def sgs_fraction(prof, sym="t"):
    """SGS fractions of the stress and heat flux, and the SGS Prandtl number.
    Fractions are masked where the total flux is below 2 % (stress) or 5 %
    (heat flux) of its maximum."""
    out = xr.Dataset(coords=prof.coords)
    if "tau" in prof and "tau_sgs" in prof:
        out["stress"] = prof.tau_sgs / prof.tau.where(prof.tau > 0.02 * prof.tau.max())
    wt, wts = f"w{sym}_tot", f"w{sym}_sgs"
    if wt in prof and wts in prof:
        tot = prof[wt].where(abs(prof[wt]) > 0.05 * abs(prof[wt]).max())
        out["heat_flux"] = prof[wts] / tot
    if "tke_res_frac" in prof:
        out["tke_sgs"] = 1 - prof.tke_res_frac
    for k in ("nut", f"alpha_{sym}", "Pr_sgs"):
        if k in prof:
            out[k] = prof[k]
    return out


def plot_sgs(prof, zi=None, zmax=None, sym="t", labels=None, axs=None, evolution=None, title=None):
    """Three panels (one colour per simulation):
    <nu_t> (solid) and <alpha_t> (dashed);  SGS fraction of the stress (solid),
    heat flux (dashed) and TKE (dotted);  SGS Prandtl number <nu_t>/<alpha_t>.
    evolution: (time, z) derived profiles, drawn as thin lines coloured by time
    (one simulation only; see profiles.plot_profiles). title: see plot_profiles."""
    from .profiles import _evo_z, _evolution
    data = ps.as_dict(prof, labels)
    evo = _evolution(data, evolution)
    created = axs is None
    if axs is None:
        fig, axs = ps.subplots(1, 3)
    else:
        fig = axs[0].figure
    for i, (lab, p) in enumerate(data.items()):
        fr = sgs_fraction(p, sym)
        z = ps.zcoord(p, zi)
        c, lw = ps.mean_color(i, evo), ps.mean_lw(evo)
        if evo is not None:
            ez = _evo_z(p, zi)
            for ax, k in ((axs[0], "nut"), (axs[0], f"alpha_{sym}"), (axs[1], "stress"),
                          (axs[1], "heat_flux"), (axs[2], "Pr_sgs")):
                if k in fr:
                    ps.evolution_lines(ax, evo, lambda d, k=k: sgs_fraction(d, sym).get(k), ez)
        for k, style in (("nut", "-"), (f"alpha_{sym}", "--")):
            if k in fr:
                axs[0].plot(fr[k], z, color=c, ls=style, lw=lw)
        for k, style in (("stress", "-"), ("heat_flux", "--"), ("tke_sgs", ":")):
            if k in fr:
                axs[1].plot(fr[k], z, color=c, ls=style, lw=lw)
        if "Pr_sgs" in fr:
            axs[2].plot(fr.Pr_sgs, z, color=c, label=lab, lw=lw)
    from matplotlib.lines import Line2D
    frs = [sgs_fraction(p, sym) for p in data.values()]
    has = lambda k: any(k in f for f in frs)  # noqa: E731
    axs[0].legend(handles=[Line2D([], [], color="0.3", ls=st, label=lb) for k, st, lb in
                           (("nut", "-", r"$\nu_t$"), (f"alpha_{sym}", "--", r"$\alpha_t$")) if has(k)])
    axs[0].set_xlabel(r"(m$^2$ s$^{-1}$)")
    axs[1].legend(handles=[Line2D([], [], color="0.3", ls=st, label=lb) for k, st, lb in
                           (("stress", "-", "stress"), ("heat_flux", "--", "heat flux"),
                            ("tke_sgs", ":", "TKE")) if has(k)])
    axs[1].set_xlim(-0.1, 1.1)
    axs[1].set_xlabel("SGS fraction")
    axs[2].set_xlabel(ps.label("Pr_sgs"))
    if len(data) > 1:
        axs[2].legend()
    for ax in axs:
        ps.zlabel(ax, zi)
    if zmax:
        axs[0].set_ylim(0, zmax)
    ps.evolution_colorbar(fig, axs, evo)
    if title is not False and (title or created):
        ps.average_title(fig, data, evo, title)
    return fig, axs


def resolution(prof, zi, fld=None, verbose=True):
    """Simple resolution metrics: vertical points in the BL, first-point height,
    horizontal spacing (if a field Dataset is given), resolved TKE fraction."""
    z = prof.z.values
    res = dict(N=prof.z.attrs.get("N"), nz=z.size, dz_first=z[1] - z[0],
               dz_mean_bl=float(np.mean(np.diff(z[z <= zi]))) if (z <= zi).sum() > 1 else np.nan,
               points_in_bl=int(((z > 0) & (z <= zi)).sum()),
               points_in_surface_layer=int(((z > 0) & (z <= 0.1 * zi)).sum()))
    if fld is not None:
        Lx, Ly = float(np.ptp(fld.x.values)), float(np.ptp(fld.y.values))
        res.update(Lx=Lx, Ly=Ly, dx_mean=Lx / (fld.x.size - 1), dy_mean=Ly / (fld.y.size - 1),
                   Lx_over_zi=Lx / zi)
    if "tke_res_frac" in prof:
        res["min_resolved_tke_frac_bl"] = float(prof.tke_res_frac.where(
            (prof.z > 0) & (prof.z < zi)).min())
    if verbose:
        print("---- resolution -------------------------------------------")
        for k, v in res.items():
            print(f" {k:<26}: {v:.4g}" if isinstance(v, float) else f" {k:<26}: {v}")
        if "min_resolved_tke_frac_bl" in res and res["min_resolved_tke_frac_bl"] < 0.8:
            print(" ! resolved TKE < 80 % somewhere in the BL (Pope's criterion)")
        print("-----------------------------------------------------------")
    return res


# ----------------------------------------------------------------------------
# Time step
# ----------------------------------------------------------------------------
def _inv_spacing(c, N):
    """1/local spacing as in Neko's CFL (centred inside an element, one-sided
    at the element ends; the largest value is kept at shared nodes)."""
    inv = np.zeros(c.size)
    if N is None:
        return 1.0 / np.gradient(c)
    for e in range((c.size - 1) // N):
        x = c[e * N:(e + 1) * N + 1]
        d = np.empty(N + 1)
        d[0], d[-1] = x[1] - x[0], x[-1] - x[-2]
        d[1:-1] = 0.5 * (x[2:] - x[:-2])
        sl = slice(e * N, (e + 1) * N + 1)
        inv[sl] = np.maximum(inv[sl], 1.0 / d)
    return inv


def cfl_dt(fld, cfl=0.5):
    """Largest time step giving `cfl` for a field snapshot, with Neko's CFL
    definition: dt * max(|u|/dx + |v|/dy + |w|/dz), dx = local GLL spacing."""
    snap = fld.isel(time=-1) if "time" in fld.dims else fld
    N = snap.x.attrs.get("N")
    ix, iy, iz = (xr.DataArray(_inv_spacing(snap[d].values, N), dims=d) for d in ("x", "y", "z"))
    c = abs(snap.u) * ix + abs(snap.v) * iy + abs(snap.w) * iz
    return float(cfl / c.max())


def plot_timestep(steps, iterations=True, labels=None, axs=None):
    """dt, CFL (and Krylov iterations) versus simulated time, from read_log.
    `steps` can be a dict / list of logs (one colour per simulation; the
    iterations panel then shows the pressure solver only)."""
    data = ps.as_dict(steps, labels)
    first = next(iter(data.values()))
    its = [c for c in first.columns if c.startswith("it_")] if iterations else []
    if len(data) > 1:
        its = [c for c in its if "Pres" in c][:1]
    keys = [k for k in ("dt", "cfl") if k in first] + (["its"] if its else [])
    if axs is None:
        fig, axs = ps.subplots(len(keys), 1, kind="wide", sharex=True, sharey=False,
                               figsize=(ps.STYLE["wide_width"], 1.5 * len(keys) + 0.5))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    for i, (lab, st) in enumerate(data.items()):
        for ax, k in zip(axs, keys):
            if k == "its":
                for j, c in enumerate(its):
                    if c in st:
                        col = ps.color(j) if len(data) == 1 else ps.color(i)
                        ax.plot(st.t, st[c], lw=0.8, color=col, label=c[3:] if len(data) == 1 else lab)
                ax.set_ylabel("iterations")
            elif k in st:
                ax.plot(st.t, st[k], lw=0.8, color="k" if len(data) == 1 else ps.color(i), label=lab)
                ax.set_ylabel(r"$\Delta t$ (s)" if k == "dt" else "CFL")
    if its:
        axs[-1].legend(ncol=min(4, max(len(its), len(data))))
    if len(data) > 1:
        axs[0].legend()
    axs[-1].set_xlabel(ps.label("time"))
    return fig, (axs[0] if len(axs) == 1 else axs)

def plot_step_series(series, window=600.0, labels=None, axs=None):
    """Per-step series written by the case (Run.step_series: u* from the wall torque,
    dt, CFL, wall-model heat flux and Obukhov length) versus time, as running
    means over `window` seconds (thin line = raw, thick = mean)."""
    data = ps.as_dict({k: v for k, v in (series.items() if isinstance(series, dict) else
                                         enumerate(series)) if v is not None}, labels)
    keys = [k for k in ("ustar", "wm_q", "wm_L", "dt", "cfl") if any(k in s for s in data.values())]
    if axs is None:
        fig, axs = ps.subplots(len(keys), 1, kind="wide", sharex=True, sharey=False,
                               figsize=(ps.STYLE["wide_width"], 1.4 * len(keys) + 0.6))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    ylab = {"ustar": r"$u_*$ torque (m s$^{-1}$)", "wm_q": r"wall model $q$ (K m s$^{-1}$)",
            "wm_L": r"wall model $L$ (m)", "dt": r"$\Delta t$ (s)", "cfl": "CFL"}
    for i, (lab, s) in enumerate(data.items()):
        t = s.time.values
        dt_mean = max(np.median(np.diff(t)), 1e-9)
        n = int(max(1, min(window / dt_mean, t.size // 4)))
        for ax, k in zip(axs, keys):
            if k not in s:
                continue
            y = s[k].values.astype(float)
            if k == "wm_L":
                y = np.where(np.abs(y) > 1e5, np.nan, y)
            c = ps.color(i)
            ax.plot(t, y, color=c, lw=0.3, alpha=0.35)
            ym = np.convolve(np.nan_to_num(y), np.ones(n) / n, mode="same") if n > 1 else y
            ax.plot(t[n:-n], ym[n:-n], color=c, lw=1.4, label=lab)
            ax.set_ylabel(ylab[k])
            if k == "wm_L":      # the neutral start (L -> infinity) would hide everything else
                hi = np.nanpercentile(y[t > 0.1 * t[-1]], 99) if np.isfinite(y).any() else 1.0
                ax.set_ylim(0, 1.5 * hi)
            elif k == "dt":
                ax.set_ylim(0, None)
    if len(data) > 1:
        axs[0].legend()
    axs[-1].set_xlabel(ps.label("time"))
    return fig, (axs[0] if len(axs) == 1 else axs)

# ----------------------------------------------------------------------------
# Ekman layer: cross-isobaric mass transport and wind turning
# ----------------------------------------------------------------------------
def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


@_over_time
def ekman_transport(prof, case=None, zmax=None, rho=1.0):
    """Integrated mass transport of the boundary layer and wind turning (dict,
    or Dataset over time for a (time, z) history).

    With G = (Ug, Vg) the geostrophic wind (case.u_geo, else the top wind),
    a = G/|G| (along the isobars) and c = k x a (across, towards low pressure
    for f > 0):

    Q_along   rho int_0^zmax (u - u_g) . a dz   (< 0: wind deficit)       kg m-1 s-1
    Q_cross   rho int_0^zmax (u - u_g) . c dz   (cross-isobaric transport)
    Q_cross_ekman, Q_along_ekman
              the same from the surface stress alone (steady, horizontally
              homogeneous, no stress at zmax):  f k x Q = -tau_s  ->
              Q = rho (<v'w'>_s, -<u'w'>_s) / f      (total = resolved + SGS stress)
    alpha_s   turning angle between the wind at the lowest point above the surface
              and G, degrees, counter-clockwise positive (> 0 in the NH)
    alpha_zi  same at zi (if prof.attrs['zi'] is known)
    wdir_s, wdir_g  meteorological directions (deg, where the wind comes from)
    zmax default: the top of the domain (a sponge layer, if any, is included).
    """
    from .profiles import geostrophic_wind

    ug, vg, _ = geostrophic_wind(prof, case)
    ug, vg = float(ug), float(vg)
    G = np.hypot(ug, vg)
    ca, sa = ug / G, vg / G
    p = prof if zmax is None else prof.where(prof.z <= zmax, 0.0)
    du, dv = p.U - ug, p.V - vg
    out = dict(Q_along=rho * float(sem.integrate(du * ca + dv * sa, "z")),
               Q_cross=rho * float(sem.integrate(-du * sa + dv * ca, "z")))
    f = getattr(case, "f", None) if case is not None else None
    uw = "uw_tot" if "uw_tot" in prof else "uw"
    vw = "vw_tot" if "vw_tot" in prof else "vw"
    if f and uw in prof and vw in prof:
        qx, qy = rho * _surface(prof, vw) / f, -rho * _surface(prof, uw) / f
        out["Q_along_ekman"] = qx * ca + qy * sa
        out["Q_cross_ekman"] = -qx * sa + qy * ca
    ang_g = np.degrees(np.arctan2(vg, ug))
    k = 1 if float(prof.z[0]) <= 0 else 0          # first point above the wall
    out["alpha_s"] = float(_wrap(np.degrees(np.arctan2(prof.V[k], prof.U[k])) - ang_g))
    zi = prof.attrs.get("zi")
    if zi and np.isfinite(zi):
        Ui, Vi = (float(sem.interp(prof[c], np.array([zi]))[0]) for c in ("U", "V"))
        out["alpha_zi"] = float(_wrap(np.degrees(np.arctan2(Vi, Ui)) - ang_g))
    out["wdir_s"] = float((270.0 - np.degrees(np.arctan2(prof.V[k], prof.U[k]))) % 360.0)
    out["wdir_g"] = float((270.0 - ang_g) % 360.0)
    return out


def plot_ekman(ts, labels=None, axs=None):
    """Time series of the cross-isobaric transport (LES: solid, from the surface
    stress via the Ekman balance: dashed) and of the surface turning angle.
    `ts` from ekman_transport(history) or Run.timeseries(); dict / list for
    several runs."""
    data = ps.as_dict(ts, labels)
    if axs is None:
        fig, axs = ps.subplots(3, 1, kind="wide", sharex=True, sharey=False,
                               figsize=(ps.STYLE["wide_width"], 5.0))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    for i, (lab, d) in enumerate(data.items()):
        c = ps.color(i)
        for ax, v in zip(axs[:2], ("Q_cross", "Q_along")):
            if v in d:
                ax.plot(d.time, d[v], marker="o", ms=3, color=c, ls=ps.ls("total"), label=lab)
            if v + "_ekman" in d:
                ax.plot(d.time, d[v + "_ekman"], color=c, ls=ps.ls("ref"))
        for v, st in (("alpha_s", "total"), ("alpha_zi", "res")):
            if v in d:
                axs[2].plot(d.time, d[v], marker="o", ms=3, color=c, ls=ps.ls(st))
    short = {"Q_cross": r"$Q_\perp$ (m$^2$ s$^{-1}$)", "Q_along": r"$Q_\parallel$ (m$^2$ s$^{-1}$)",
             "alpha": r"turning ($^\circ$)"}
    for ax, v in zip(axs, ("Q_cross", "Q_along", "alpha")):
        ax.set_ylabel(short[v])
        ax.axhline(0, color="0.6", lw=0.6)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([], [], color="0.3", ls=ps.ls("total"), label="LES, cross-isobaric"),
                           Line2D([], [], color="0.3", ls=ps.ls("ref"), label=r"$-\tau_s/f$ (Ekman)")],
                  loc="best")
    axs[1].set_title("along-isobaric (deficit)", fontsize=ps.STYLE["legend_size"], loc="left")
    axs[2].legend(handles=[Line2D([], [], color="0.3", ls=ps.ls("total"), label="surface"),
                           Line2D([], [], color="0.3", ls=ps.ls("res"), label=r"$z_i$")], loc="best")
    if len(data) > 1:
        ps.run_legend(axs[1], list(data), loc="best")
    axs[-1].set_xlabel(ps.label("time"))
    return fig, axs

# ----------------------------------------------------------------------------
# Surface heat flux and temperature: LES vs wall model
# ----------------------------------------------------------------------------
def surface_bc(case, t):
    """Prescribed surface value(s) at time(s) t: ('neumann', q) or ('dirichlet', theta_s);
    a time-dependent Nek5000 surface temperature (case.surface_temperature_expr) is
    evaluated. (None, None) if unknown."""
    t = np.asarray(t, float)
    expr = getattr(case, "surface_temperature_expr", None)
    if expr and re.fullmatch(r"[-+*/(). 0-9eE]*time[-+*/(). 0-9eE]*", expr.replace("time", " time ")):
        return "dirichlet", eval(expr, {"__builtins__": {}}, {"time": t})  # noqa: S307 (checked)
    bc = getattr(case, "temp_bc_type", None)
    if bc == "neumann" and case.surface_flux is not None:
        return bc, np.full(t.shape, float(case.surface_flux))
    if bc == "dirichlet" and case.surface_temperature is not None:
        return bc, np.full(t.shape, float(case.surface_temperature))
    return bc, None


def _wm_name(case):
    wm = str(getattr(case, "wall_model", "") or "").lower()
    return "richardson" if "richardson" in wm else "most"


@_over_time
def _surface_one(prof, case, model, sym, t):
    S = sym.upper()
    hi = case.h_index if case.h_index is not None else 1
    h = float(case.h) if getattr(case, "h", None) else float(prof.z[hi])
    Mh = float(sem.interp(prof.M, np.array([h]))[0])
    Th = float(sem.interp(prof[S], np.array([h]))[0])
    wt = f"w{sym}_tot" if f"w{sym}_tot" in prof else f"w{sym}"
    if "time" in prof.coords:
        tt = float(prof.time)
    elif "t0" in prof.attrs and "t1" in prof.attrs:      # time-averaged profile: mid-window
        tt = 0.5 * (prof.attrs["t0"] + prof.attrs["t1"])
    else:
        tt = t if t is not None else np.nan
    bc, val = surface_bc(case, tt)
    out = dict(h=h, M_h=Mh, T_h=Th, ts_les=_surface(prof, S),
               q_les=_surface(prof, wt) if wt in prof else np.nan,
               ustar_les=float(_surface(prof, "tau") ** 0.5) if "tau" in prof else np.nan,
               bc_value=float(val) if val is not None else np.nan,
               bc_dirichlet=float(bc == "dirichlet"))
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    out["L_les"] = float(most.obukhov_length(out["ustar_les"], out["q_les"], T0, g, case.kappa or 0.4))
    if val is not None and np.isfinite(val) and case.z0:
        g = prof.attrs.get("g", 9.81)
        wm = most.neko_wall_model(Mh, Th, h, case.z0, case.z0h, case.kappa or 0.4,
                                  case.Pr or 1.0, g, bc, float(val), model)
        out.update({f"{k}_wm": float(v) for k, v in wm.items()})
    return out


def surface_check(prof, case, model=None, sym="t"):
    """Surface heat flux, surface temperature and u*: what the LES has versus what
    the wall model gives for the MEAN wind and temperature at the sampling height h.

    prof  : derived profile (z) or history (time, z) (Run.profile / Run.history).
    model : 'most' or 'richardson' (default from case.wall_model).
    Returns a dict / Dataset(time) with
      q_les      total (resolved + SGS) heat flux at the wall node
      ts_les     temperature at the wall node (what the simulation has at the surface)
      ustar_les  sqrt of the total stress at the wall node
      bc_value   prescribed surface flux (Neumann) or temperature (Dirichlet)
      q_wm, ts_wm, ustar_wm, L_wm, Ri_b_wm   most.neko_wall_model on the mean M(h), T(h):
                 Dirichlet -> q_wm is the flux the model imposes for the mean state;
                 Neumann   -> ts_wm is the surface temperature implied by its heat law.
      M_h, T_h, h
    The model works node by node on instantaneous values, so the mean-state values
    differ from the averaged model output (Jensen); the log (Neko) or wmles.dat
    (Nek5000) give the latter, see Run.wall_model_series and plot_surface."""
    model = model or _wm_name(case)
    return _surface_one(prof, case, model, sym, None)


def plot_surface(chk, series=None, labels=None, window=600.0, axs=None):
    """Surface heat flux, surface temperature, u* and L versus time.
    chk    : surface_check of a history (dict / list for several runs).
    series : optional per-step wall-model output (Run.wall_model_series), same keys.
    Styles: LES statistics = solid with markers, wall model on the mean state = dashed,
    wall-model output every step = thin dash-dot (running mean over `window` s),
    prescribed value = grey."""
    data = ps.as_dict(chk, labels)
    ser = series if isinstance(series, dict) else ({k: series for k in data} if series is not None else {})
    panels = [("q", r"surface heat flux (K m s$^{-1}$)"), ("ts", r"surface $\theta_s$ (K)"),
              ("ustar", ps.label("ustar")), ("L", ps.label("L"))]
    if axs is None:
        fig, axs = ps.subplots(len(panels), 1, kind="wide", sharex=True, sharey=False,
                               figsize=(ps.STYLE["wide_width"], 1.5 * len(panels) + 0.7))
    else:
        fig = np.ravel(axs)[0].figure
    axs = np.atleast_1d(axs)
    for i, (lab, d) in enumerate(data.items()):
        c = ps.color(i)
        t = d.time
        dirichlet = bool(np.nanmax(np.atleast_1d(d.get("bc_dirichlet", 0.0))))
        fixed = "ts" if dirichlet else "q"          # the prescribed quantity
        for ax, (k, _) in zip(axs, panels):
            if f"{k}_les" in d:
                ax.plot(t, d[f"{k}_les"], marker="o", ms=3, color=c, ls=ps.ls("total"))
            if f"{k}_wm" in d and k != fixed:
                ax.plot(t, d[f"{k}_wm"], marker="s", ms=3, color=c, ls=ps.ls("ref"), mfc="none")
        if "bc_value" in d and np.isfinite(d.bc_value).any():
            axs[1 if dirichlet else 0].plot(t, d.bc_value, color="0.5", lw=2, alpha=0.6, zorder=1)
        if "T_h" in d:
            axs[1].plot(t, d.T_h, color=c, ls=ps.ls("sgs"), lw=1)
        s = ser.get(lab)
        if s is not None and len(s):
            tt = np.asarray(s["t"], float)
            n = int(max(1, min(window / max(np.median(np.diff(tt)), 1e-9), tt.size // 4)))
            for ax, (k, _) in zip(axs, panels):
                if k in s:
                    y = np.asarray(s[k], float)
                    if k == "L":
                        y = np.where(np.abs(y) > 1e5, np.nan, y)
                    ym = (np.convolve(np.nan_to_num(y), np.ones(n) / n, mode="same") if n > 1 else y)
                    sl = slice(n, -n if n > 1 else None)
                    ax.plot(tt[sl], ym[sl], color=c, ls=ps.ls("model"), lw=1.0)
    for ax, (k, yl) in zip(axs, panels):
        ax.set_ylabel(yl)
    Ls = np.concatenate([np.asarray(dd.L_les.where(dd.time > dd.time[0] + 0.2 * (dd.time[-1] - dd.time[0])))
                         for dd in data.values() if "L_les" in dd and "time" in dd.dims])
    Ls = Ls[np.isfinite(Ls)]
    Lall = np.concatenate([np.asarray(dd.L_les) for dd in data.values() if "L_les" in dd])
    Lall = Lall[np.isfinite(Lall)]
    if Ls.size:       # the (near-)neutral start would hide the rest: clip only if it is far out
        lo, hi = np.percentile(Ls, [2, 98])
        pad = 0.5 * (hi - lo) + 1.0
        if Lall.min() < lo - 2 * pad or Lall.max() > hi + 2 * pad:
            axs[3].set_ylim(min(lo - pad, 0) if lo < 0 else max(lo - pad, 0), hi + pad if hi > 0 else 0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", marker="o", ms=3, ls=ps.ls("total"), label="LES (statistics)"),
         Line2D([], [], color="0.3", marker="s", ms=3, mfc="none", ls=ps.ls("ref"),
                label="wall model, mean state"),
         Line2D([], [], color="0.3", ls=ps.ls("model"), label="wall model output"),
         Line2D([], [], color="0.5", lw=2, alpha=0.6, label="prescribed"),
         Line2D([], [], color="0.3", ls=ps.ls("sgs"), label=r"$\theta(h)$")]
    axs[0].legend(handles=h, loc="best", ncol=2)
    if len(data) > 1:
        ps.run_legend(axs[-1], list(data), loc="best")
    axs[-1].set_xlabel(ps.label("time"))
    return fig, axs

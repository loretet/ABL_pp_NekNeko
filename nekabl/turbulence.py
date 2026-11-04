"""
turbulence.py -- Turbulence processing.

From statistics (derived profiles, cheap):
    tke_budget(prof, eps=None)  TKE budget: shear (resolved + SGS), buoyancy (resolved
                                + SGS), turbulent and pressure transport, dissipation
    heat_flux_budget(prof)      budget of <w'theta'> (transport/pressure need 'full' scalar stats)
    theta_variance_budget(prof) budget of <theta'^2>/2 (transport needs 'full' scalar stats)
    eddy_viscosity(prof)        diagnosed K_m, K_h, Pr_t and mixing length
From field files:
    dissipation_from_fields(fld)  SGS dissipation profile (Deardorff model, or
                                  local equilibrium for other eddy-viscosity models)
    spectrum(fld, var, z)       1D (x or y) or radial 2D (co)spectra
    autocorrelation(fld, var, z), integral_length(R)
    quadrant_analysis(fld, 'w', 't', z), pdf(fld, var, z)

Budget of the TOTAL (resolved + SGS) TKE, horizontally homogeneous flow
---------------------------------------------------------------------
 d(k + <e>)/dt =
     - <u'w'> dU/dz - <v'w'> dV/dz                 shear_res        (basic)
     - <tau_13> dU/dz - <tau_23> dV/dz             shear_sgs        (basic + fluid SGS stats)
     + g/theta0 <w'theta'>                         buoyancy_res     (basic)
     + g/theta0 <w'theta'>_sgs                     buoyancy_sgs     (basic + scalar SGS stats)
     - d/dz <w' u_i'u_i'>/2                        turb_transport   (FULL fluid set)
     - d/dz <p'w'>                                 pres_transport   (FULL fluid set)
     - eps                                         dissipation      (from fields, see below)
     + nu d2k/dz2 - nu <du_i'/dx_k du_i'/dx_k>     viscous          (FULL set + nu; ~0 in LES)
     + d/dz <(nu + 2 K_m) de/dz>                   sgs_transport    (SGS-TKE model with its flux in the stats: Nek5000 KTH)
     + resolved/SGS cross transport + storage      -> residual
The exchange term between resolved and SGS TKE, <tau_ij' S_ij'>, cancels in
the total budget, which is why the total is the natural budget for LES.
Coriolis does no work on TKE; the sponge acts only near the top.
The residual is -(sum of the available terms): for a stationary flow it is
what is missing to close the budget (SGS transport, plus eps and the
transports when they are not available).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import fields
from . import plotstyle as ps
from . import sem


def _ddz(da):
    return sem.ddz(da)


def _residual(b):
    tot = sum(b[k] for k in b.data_vars)
    b["residual"] = -tot
    return b


def _on(prof, da):
    """Profile `da` on the heights of `prof` (spectral interpolation if needed)."""
    if da.z.size == prof.z.size and np.allclose(da.z.values, prof.z.values):
        return da.assign_coords(z=prof.z)
    out = sem.interp(da, prof.z.values)
    return out.assign_coords(z=prof.z)


def _missing_note(b, missing):
    b.attrs["missing"] = "; ".join(missing)
    if missing:
        print(f"[nekabl] {b.attrs['name']} budget: not available -> " + "; ".join(missing))
    return b


# ----------------------------------------------------------------------------
# Budgets (from statistics)
# ----------------------------------------------------------------------------
def tke_budget(prof, eps=None, sgs=True, nu=None):
    """Budget of the total (resolved + SGS) TKE, m2 s-3 (see module docstring).

    prof : derived profiles (profiles.derive).
    eps  : optional SGS dissipation profile (positive), e.g. from
           dissipation_from_fields; it is interpolated on prof.z.
    sgs  : include the SGS shear and buoyancy production (total-TKE budget);
           False gives the classic resolved-TKE budget (then the residual
           also contains the resolved-to-SGS transfer).
    nu   : molecular viscosity for the viscous terms (default prof.attrs['nu']).
    """
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    sym = prof.attrs.get("sym", "t")
    nu = nu if nu is not None else prof.attrs.get("nu")
    b = xr.Dataset(coords=prof.coords, attrs=dict(prof.attrs))
    b.attrs["name"] = "TKE"
    missing = []
    b["shear_res"] = -(prof.uw * prof.dUdz + prof.vw * prof.dVdz)
    if sgs:
        if "uw_sgs" in prof and "vw_sgs" in prof:
            b["shear_sgs"] = -(prof.uw_sgs * prof.dUdz + prof.vw_sgs * prof.dVdz)
        else:
            missing.append("shear_sgs (no fluid SGS stats)")
    if f"w{sym}" in prof:
        b["buoyancy_res"] = g / T0 * prof[f"w{sym}"]
    if sgs:
        if f"w{sym}_sgs" in prof:
            b["buoyancy_sgs"] = g / T0 * prof[f"w{sym}_sgs"]
        else:
            missing.append("buoyancy_sgs (no scalar SGS stats)")
    if all(k in prof for k in ("uuw", "vvw", "www")):
        b["turb_transport"] = -_ddz(0.5 * (prof.uuw + prof.vvw + prof.www))
    else:
        missing.append("turb_transport (needs the 'full' fluid stats)")
    if "pw" in prof:
        b["pres_transport"] = -_ddz(prof.pw)
    else:
        missing.append("pres_transport (needs the 'full' fluid stats)")
    if "dkdz_diff" in prof:
        b["sgs_transport"] = _ddz(prof.dkdz_diff)
    if eps is not None:
        b["dissipation"] = -_on(prof, eps)
    else:
        missing.append("dissipation (pass eps=, e.g. from dissipation_from_fields)")
    if nu and all(k in prof for k in ("e11", "e22", "e33")):
        b["viscous"] = nu * _ddz(_ddz(prof.tke)) - nu * (prof.e11 + prof.e22 + prof.e33)
    b = _residual(b)
    return _missing_note(b, missing)


def heat_flux_budget(prof):
    """Budget of the vertical heat flux <w'theta'> (K m s-2):
    gradient production -<w'w'> dTheta/dz, buoyancy g/theta0 <theta'^2>,
    turbulent transport -d<w'w'theta'>/dz and pressure scrambling
    -<theta' dp'/dz> = -d<p'theta'>/dz + <p' dtheta'/dz> (both 'full' scalar)."""
    g, T0 = prof.attrs.get("g", 9.81), prof.attrs.get("T0", 300.0)
    s = prof.attrs.get("sym", "t")
    S = s.upper()
    b = xr.Dataset(coords=prof.coords, attrs=dict(prof.attrs))
    b.attrs["name"] = r"$\langle w'\theta'\rangle$"
    missing = []
    b["gradient_prod"] = -prof.ww * prof[f"d{S}dz"]
    b["buoyancy"] = g / T0 * prof[s + s]
    if f"ww{s}" in prof:
        b["turb_transport"] = -_ddz(prof[f"ww{s}"])
    else:
        missing.append("turb_transport (needs the 'full' scalar stats)")
    if f"p{s}" in prof and f"pd{s}dz" in prof:
        b["pressure"] = -(_ddz(prof[f"p{s}"]) - prof[f"pd{s}dz"])
    else:
        missing.append("pressure term (needs the 'full' scalar stats)")
    b = _residual(b)
    return _missing_note(b, missing)


def theta_variance_budget(prof, sgs=True):
    """Budget of half the temperature variance <theta'^2>/2 (K2 s-1):
    gradient production by the resolved (and SGS) heat flux, turbulent
    transport -d<w'theta'theta'>/dz/2 ('full' scalar), residual = -chi + ..."""
    s = prof.attrs.get("sym", "t")
    S = s.upper()
    b = xr.Dataset(coords=prof.coords, attrs=dict(prof.attrs))
    b.attrs["name"] = r"$\langle\theta'^2\rangle/2$"
    missing = []
    b["gradient_prod_res"] = -prof[f"w{s}"] * prof[f"d{S}dz"]
    if sgs and f"w{s}_sgs" in prof:
        b["gradient_prod_sgs"] = -prof[f"w{s}_sgs"] * prof[f"d{S}dz"]
    if f"w{s}{s}" in prof:
        b["turb_transport"] = -_ddz(0.5 * prof[f"w{s}{s}"])
    else:
        missing.append("turb_transport (needs the 'full' scalar stats)")
    b = _residual(b)
    return _missing_note(b, missing)


_TERM_COLOR = dict(shear=0, buoyancy=1, turb_transport=2, pres_transport=3, gradient_prod=0,
                   pressure=3, dissipation=6, viscous=5)


def _term_style(k):
    base, comp = k, "total"
    for suf, c in (("_res", "res"), ("_sgs", "sgs")):
        if k.endswith(suf):
            base, comp = k[: -len(suf)], c
    return ps.color(_TERM_COLOR.get(base, 4)), ps.ls(comp)


def plot_budget(bud, zi=None, scale=1.0, zmax=None, labels=None, ax=None, xlabel=None, legend=True,
                focus=True, title=None):
    """All terms of a budget versus height. Colour = term, line style =
    resolved (dashed) / SGS (dotted) / not split (solid), residual = grey.
    `bud` can be a dict / list (one panel per simulation, shared axes).
    `scale` multiplies every term (e.g. zi / wstar**3 for convective scaling);
    it can also be a dict {label: scale}.
    focus : x range set by the terms above the lowest 10 % of the plotted
            height (the large near-wall values go off-axis, marked by a
            note); False -> full range.
    title : see profiles.plot_profiles (averaging window of each budget).
    The time evolution is not drawn here: colour already identifies the term."""
    data = ps.as_dict(bud, labels)
    if ax is None:
        fig, axs = ps.subplots(1, len(data), sharex=False,
                               figsize=(3.3 * len(data) + (1.9 if legend else 0), ps.STYLE["panel_height"]))
    else:
        fig, axs = ax.figure, ax
    axs = np.atleast_1d(axs)
    for a, (lab, bd) in zip(axs, data.items()):
        sc = scale.get(lab, 1.0) if isinstance(scale, dict) else scale
        z = ps.zcoord(bd, zi)
        for k in bd.data_vars:
            if k == "residual":
                a.plot(bd[k] * sc, z, color="0.55", lw=2.2, alpha=0.7, label="residual", zorder=1)
            else:
                c, style = _term_style(k)
                a.plot(bd[k] * sc, z, color=c, ls=style, label=k.replace("_", " "))
        a.axvline(0, color="0.6", lw=0.6)
        ps.zlabel(a, zi)
        unscaled = (scale == 1.0) if not isinstance(scale, dict) else False
        a.set_xlabel(xlabel or f"{bd.attrs.get('name', '')} budget" + ("" if unscaled else " (scaled)"))
        if len(data) > 1:
            a.set_title(lab)
        if zmax:
            a.set_ylim(0, zmax)
    if focus:       # per panel (runs may use different scalings)
        for a, (lab, bd) in zip(axs, data.items()):
            sc = scale.get(lab, 1.0) if isinstance(scale, dict) else scale
            z = np.asarray(ps.zcoord(bd, zi))
            top = zmax or float(z.max())
            inner, full = 0.0, 0.0
            for k in bd.data_vars:
                v = np.abs(np.asarray(bd[k] * sc))
                ok = np.isfinite(v) & (z <= top)
                full = max(full, np.nanmax(v[ok], initial=0.0))
                ok &= z >= 0.1 * top
                inner = max(inner, np.nanmax(v[ok], initial=0.0))
            if 0 < 2.5 * inner < full:
                a.set_xlim(-1.3 * inner, 1.3 * inner)
                a.text(0.02, 0.01, "near-wall values off-axis", transform=a.transAxes,
                       fontsize=ps.STYLE["legend_size"] - 1, color="0.4", va="bottom")
    if legend:
        axs[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    if title is not False and (title or ax is None):
        ps.average_title(fig, data, None, title)
    return fig, (axs[0] if len(axs) == 1 else axs)


def dissipation_from_fields(fld, case=None, model=None, g=None, T0=None, e="e", delta="delta",
                            nut="nut", t="t", c_eps=(0.19, 0.74)):
    """Horizontally (and time) averaged SGS dissipation profile from field files.

    model='deardorff' (default if `e` and `delta` are in fld), as in Neko:
        N2 = g/theta0 dtheta/dz,  l = min(0.76 sqrt(e/N2), Delta) if N2 > 0 else Delta
        eps = (0.19 + 0.74 l/Delta) e^(3/2) / l
    model='equilibrium' (any eddy-viscosity model, needs u, v, w, nu_t):
        local equilibrium of the SGS TKE: eps = 2 nu_t S_ij S_ij
        (buoyancy production of SGS TKE neglected).
    Gradients use the spectral-element derivatives.
    """
    if case is not None:       # the SGS model may use its own constants (Nek5000 tke.f)
        g = g if g is not None else (getattr(case, "sgs_g", None) or case.g)
        T0 = T0 if T0 is not None else (getattr(case, "sgs_T0", None) or case.T0)
    g, T0 = g or 9.81, T0 or 300.0
    if model is None:
        model = "deardorff" if (e in fld and delta in fld) else "equilibrium"
    if model == "deardorff":
        E = fld[e].clip(min=1e-12)
        D = fld[delta]
        N2 = g / T0 * sem.ddz(fld[t], "z")
        lstab = 0.76 * np.sqrt(E / N2.where(N2 > 0))
        l = xr.where(N2 > 0, np.minimum(lstab.fillna(np.inf), D), D)
        eps = (c_eps[0] + c_eps[1] * l / D) * E**1.5 / l
    elif model == "equilibrium":
        grads = {}
        for i, c in enumerate("uvw"):
            for j, d in enumerate("xyz"):
                grads[i, j] = sem.ddz(fld[c], d)
        SS = 0.0
        for i in range(3):
            for j in range(3):
                Sij = 0.5 * (grads[i, j] + grads[j, i])
                SS = SS + Sij * Sij
        eps = 2 * fld[nut] * SS
    else:
        raise ValueError("model must be 'deardorff' or 'equilibrium'")
    prof = fields.hmean(eps)
    if "time" in prof.dims:
        prof = prof.mean("time")
    prof = prof.drop_vars([c for c in ("wz",) if c in prof.coords])
    prof.name = "eps_sgs"
    prof.attrs["model"] = model
    prof.z.attrs.update(fld.z.attrs)
    return prof


def eddy_viscosity(prof, min_grad=1e-4, min_shear=0.02, sym="t"):
    """Diagnosed (total = resolved + SGS) eddy viscosity and diffusivity.

    K_m  = -(<u'w'> dU/dz + <v'w'> dV/dz) / S^2       (projection on the shear)
    K_h  = -<w'theta'> / dTheta/dz   (masked where |dTheta/dz| < min_grad K/m:
                                       counter-gradient / well mixed layer)
    Pr_t = K_m / K_h (only where both are positive),  l_m = sqrt(K_m / S)
    K_m is masked where the shear S is below `min_shear` times its maximum
    (above the BL the wind is uniform and the ratio is meaningless).
    Also K_m_res, K_h_res from the resolved fluxes only, and <nu_t>, <alpha_t>.
    """
    S = sym.upper()
    out = xr.Dataset(coords=prof.coords, attrs=prof.attrs)
    sh2 = (prof.S**2).where(prof.S > min_shear * prof.S.max())
    for tag, suf in (("", "_tot"), ("_res", "")):
        uw, vw = prof.get("uw" + suf), prof.get("vw" + suf)
        if uw is not None and vw is not None:
            out["K_m" + tag] = -(uw * prof.dUdz + vw * prof.dVdz) / sh2
        wt = prof.get(f"w{sym}" + suf)
        if wt is not None and f"d{S}dz" in prof:
            grad = prof[f"d{S}dz"].where(abs(prof[f"d{S}dz"]) > min_grad)
            out["K_h" + tag] = -wt / grad
    if "K_m" in out and "K_h" in out:
        out["Pr_t"] = (out.K_m / out.K_h).where((out.K_m > 0) & (out.K_h > 0))
    if "K_m" in out:
        out["l_m"] = np.sqrt(out.K_m.where(out.K_m > 0) / prof.S)
    for k in ("nut", f"alpha_{sym}"):
        if k in prof:
            out[k] = prof[k]
    return out


def plot_eddy_viscosity(ev, zi=None, zmax=None, labels=None, axs=None, sym="t", evolution=None,
                        title=None):
    """K_m with <nu_t>, K_h with <alpha_t>, and Pr_t. Line style: total
    (solid), resolved (dashed), SGS model (dotted); colour = simulation.
    evolution: (time, z) DERIVED profiles (e.g. run.evolution()); the total
    K_m, K_h, Pr_t of every window are drawn as thin lines coloured by time.
    title: see profiles.plot_profiles."""
    from .profiles import _evo_z, _evolution
    data = ps.as_dict(ev, labels)
    evo = _evolution(data, evolution)
    created = axs is None
    if axs is None:
        fig, axs = ps.subplots(1, 3)
    else:
        fig = axs[0].figure
    for i, (lab, e) in enumerate(data.items()):
        z = ps.zcoord(e, zi)
        c, lw = ps.mean_color(i, evo), ps.mean_lw(evo)
        if evo is not None:
            for ax, k in zip(axs, ("K_m", "K_h", "Pr_t")):
                ps.evolution_lines(ax, evo, lambda d, k=k: eddy_viscosity(d, sym=sym).get(k), _evo_z(e, zi))
        for ax, k, sgs in ((axs[0], "K_m", "nut"), (axs[1], "K_h", f"alpha_{sym}")):
            for name, comp in ((k, "total"), (k + "_res", "res"), (sgs, "sgs")):
                if name in e:
                    ax.plot(e[name], z, color=c, ls=ps.ls(comp), lw=lw)
        if "Pr_t" in e:
            axs[2].plot(e.Pr_t, z, "o", ls="-", ms=2.5, color=c, label=lab, lw=lw)
    for ax, k in zip(axs, ("K_m", "K_h", "Pr_t")):
        ax.set_xlabel(ps.label(k))
        ps.zlabel(ax, zi)
    axs[0].axvline(0, color="0.6", lw=0.6)
    axs[1].axvline(0, color="0.6", lw=0.6)
    axs[2].set_xlim(0, 2)
    ps.component_legend(axs[0], names=dict(total="total", res="resolved", sgs="SGS model"))
    if len(data) > 1:
        axs[2].legend()
    if zmax:
        axs[0].set_ylim(0, zmax)
    ps.evolution_colorbar(fig, axs, evo)
    if title is not False and (title or created):
        ps.average_title(fig, data, evo, title)
    return fig, axs

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

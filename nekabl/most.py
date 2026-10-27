"""
most.py -- Monin-Obukhov similarity theory (MOST).

The functions are the same as in Neko's MOST wall model (src/wall_models/
bcknd/cpu/most_cpu.f90) so that LES results can be compared with what the
wall model assumes:
    unstable (zeta < 0): Businger-Dyer (Paulson 1970 integrated forms), gamma=16
    stable   (zeta > 0): Beljaars & Holtslag (1991), a=1, b=2/3, c=5, d=0.35

zeta = z / L,  L = -u*^3 theta_0 / (kappa g <w'theta'>_s)
"""

import numpy as np


def phi_m(zeta):
    """Dimensionless wind shear  kappa z / u* dU/dz."""
    zeta = np.asarray(zeta, float)
    a, b, c, d = 1.0, 2.0 / 3.0, 5.0, 0.35
    stable = 1 + zeta * (a + b * np.exp(-d * zeta) * (1 + c - d * zeta))
    unstable = (1 - 16 * np.minimum(zeta, 0)) ** -0.25
    return np.where(zeta >= 0, stable, unstable)


def phi_h(zeta, Pr=1.0):
    """Dimensionless temperature gradient  kappa z / theta* dTheta/dz.

    Pr is the neutral value phi_h(0) (the 'Pr' of the Neko MOST wall model,
    whose heat log law is  q = kappa/Pr u* (theta_s - theta) / [ln(z/z0h) - psi_h]).
    """
    zeta = np.asarray(zeta, float)
    a, b, c, d = 1.0, 2.0 / 3.0, 5.0, 0.35
    stable = 1 + zeta * (a * np.sqrt(1 + 2.0 / 3.0 * a * np.maximum(zeta, 0))
                         + b * np.exp(-d * zeta) * (1 + c - d * zeta))
    unstable = (1 - 16 * np.minimum(zeta, 0)) ** -0.5
    return Pr * np.where(zeta >= 0, stable, unstable)


def psi_m(zeta):
    """Integrated stability correction for momentum."""
    zeta = np.asarray(zeta, float)
    a, b, c, d = 1.0, 2.0 / 3.0, 5.0, 0.35
    stable = -a * zeta - b * (zeta - c / d) * np.exp(-d * zeta) - b * c / d
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    unstable = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
    return np.where(zeta >= 0, stable, unstable)


def psi_h(zeta):
    """Integrated stability correction for heat."""
    zeta = np.asarray(zeta, float)
    a, b, c, d = 1.0, 2.0 / 3.0, 5.0, 0.35
    stable = (-b * (zeta - c / d) * np.exp(-d * zeta)
              - (1 + 2.0 / 3.0 * a * np.maximum(zeta, 0)) ** 1.5 - b * c / d + 1)
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    unstable = 2 * np.log((1 + x**2) / 2)
    return np.where(zeta >= 0, stable, unstable)


def obukhov_length(ustar, wt_s, T0=300.0, g=9.81, kappa=0.4):
    """L = -u*^3 theta_0 / (kappa g <w'theta'>_s)  (inf for neutral)."""
    with np.errstate(divide="ignore"):
        return -np.asarray(ustar) ** 3 * T0 / (kappa * g * np.asarray(wt_s))


def solve_ustar(M, h, z0, wt_s, T0=300.0, g=9.81, kappa=0.4, niter=100, tol=1e-6):
    """u* and L from MOST given the wind speed M at height h and the surface
    heat flux (Neumann case, as the Neko wall model with a flux BC).

    Fixed-point iteration: u* = kappa M / [ln(h/z0) - psi_m(h/L) + psi_m(z0/L)].
    """
    us = kappa * M / np.log(h / z0)
    L = np.inf
    for _ in range(niter):
        L = obukhov_length(us, wt_s, T0, g, kappa) if wt_s != 0 else np.inf
        corr = 0.0 if not np.isfinite(L) else (-psi_m(h / L) + psi_m(z0 / L))
        new = kappa * M / (np.log(h / z0) + corr)
        if abs(new - us) < tol * max(us, 1e-12):
            us = new
            break
        us = 0.5 * (us + new)
    return float(us), float(L)


def wind_profile(z, ustar, L, z0, kappa=0.4):
    """MOST mean wind speed M(z)."""
    z = np.asarray(z, float)
    corr = 0.0 if not np.isfinite(L) else (-psi_m(z / L) + psi_m(z0 / L))
    return ustar / kappa * (np.log(z / z0) + corr)


# ----------------------------------------------------------------------------
# Offline replica of the Neko wall models (src/wall_models/bcknd/cpu)
# ----------------------------------------------------------------------------
def slaw_m(z, L, z0):
    """ln(z/z0) - psi_m(z/L) + psi_m(z0/L)  (neutral if L is inf/None/0)."""
    if L is None or not np.isfinite(L) or L == 0:
        return float(np.log(z / z0))
    return float(np.log(z / z0) - psi_m(z / L) + psi_m(z0 / L))


def slaw_h(z, L, z0h):
    """ln(z/z0h) - psi_h(z/L) + psi_h(z0h/L)."""
    if L is None or not np.isfinite(L) or L == 0:
        return float(np.log(z / z0h))
    return float(np.log(z / z0h) - psi_h(z / L) + psi_h(z0h / L))


def _newton_L(Ri_b, h, z0, z0h, Pr, dirichlet, tol=1e-3, max_count=50, step=1e-3):
    """Obukhov length from the bulk Richardson number, as in most_cpu.f90."""
    def f(L):
        if dirichlet:
            return Ri_b - Pr * h / L * slaw_h(h, L, z0h) / slaw_m(h, L, z0) ** 2
        return Ri_b - Pr * h / L / slaw_m(h, L, z0) ** 3

    sign = 1.0 if Ri_b > 0 else -1.0
    L = h / (max(Ri_b, 1e-4) if Ri_b > 0 else min(Ri_b, -1e-4))
    L_old, count = 1e10, 0
    while abs(L_old - L) / abs(L) > tol and count < max_count:
        L_old, count = L, count + 1
        d = step * L
        dfdl = (f(L + d) - f(L - d)) / (2 * d)
        if abs(dfdl) < 1e-12:
            break
        new = L - f(L) / dfdl
        if new * sign <= 0:
            new = 0.5 * L
        L = sign * min(max(abs(new), 1e-8), 1e8)
    return L


def neko_wall_model(M, ti, h, z0, z0h=None, kappa=0.4, Pr=1.0, g=9.81, bc_type="neumann",
                    bc_value=0.0, model="most"):
    """What a Neko wall model ('most' or 'richardson') returns for a wind speed M and a
    temperature ti sampled at height h, with a 'neumann' (bc_value = surface heat flux,
    K m/s, > 0 upward) or 'dirichlet' (bc_value = surface temperature) temperature BC.

    Returns dict(ustar, q, ts, L, Ri_b). For the Neumann BC `ts` is the surface
    temperature implied by the model's heat law (Neko does not compute it in MOST);
    for Dirichlet `q` is the heat flux the model imposes. Same formulas as
    src/wall_models/bcknd/cpu/{most,richardson}_cpu.f90, with two sign conventions
    made consistent (see METHODS.md): the Richardson neutral heat flux is
    kappa u* (ts - ti) / (Pr ln(h/z0h)) and the implied ts satisfies q = C (ts - ti).
    """
    z0h = z0h or z0
    M = max(float(M), 1e-6)
    dirichlet = bc_type == "dirichlet"
    ts = bc_value if dirichlet else None
    q = None if dirichlet else bc_value
    thr = 1e-4
    if model == "most":
        Ri_b = (g * h / ti * (ti - ts) / M**2 if dirichlet
                else -g * h / ti * q * Pr / (M**3 * kappa**2))
        L = np.inf if abs(Ri_b) <= thr else _newton_L(Ri_b, h, z0, z0h, Pr, dirichlet)
        us = kappa * M / slaw_m(h, L, z0)
        if dirichlet:
            q = kappa / Pr * us * (ts - ti) / slaw_h(h, L, z0h)
        else:
            ts = ti + q * Pr * slaw_h(h, L, z0h) / (kappa * us)
        if np.isfinite(L):     # Neko's L, reference temperature ti
            L = -ti * us**3 / (kappa * g * q) if q else np.inf
        return dict(ustar=us, q=q, ts=ts, L=L, Ri_b=Ri_b)
    if model == "richardson":
        Ri_b = (g * h / ti * (ti - ts) / M**2 if dirichlet
                else -g * h / ti * q / (M**3 * kappa**2))
        lnm, lnh, l = np.log(h / z0), np.log(h / z0h), kappa * h
        if Ri_b > thr:
            tau = M**2 / lnm**2 * (0.25 + 0.75 / (1 + 4 * Ri_b)) / 1.0 * (l / h) ** 2
            us = np.sqrt(tau)
            C = (0.145 / (1 + 4 * Ri_b)) / 0.145 * (l / h) * us / Pr / lnh       # q = C (ts - ti)
        elif Ri_b < -thr:
            a, a_h = kappa / lnm, kappa / lnh
            c, c_h = 7.4 * a**2 * 2 * (h / z0) ** 0.5, 5.3 * a_h**2 * 2 * (h / z0h) ** 0.5
            us = np.sqrt(a**2 * M**2 * (1 - 2 * Ri_b / (1 + c * abs(Ri_b) ** 0.5)))
            C = a_h**2 / 0.74 * M * (1 - 2 * Ri_b / (1 + c_h * abs(Ri_b) ** 0.5))
        else:
            us = kappa * M / lnm
            C = kappa * us / (Pr * lnh)       # Neko: no Pr and opposite sign (see METHODS.md)
        if dirichlet:
            q = C * (ts - ti)
        else:
            ts = ti + q / C
        L = -ts * us**3 / (kappa * g * q) if q and abs(Ri_b) > thr else np.inf
        return dict(ustar=float(us), q=float(q), ts=float(ts), L=float(L), Ri_b=float(Ri_b))
    raise ValueError("model must be 'most' or 'richardson'")

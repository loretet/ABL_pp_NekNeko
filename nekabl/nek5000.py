"""
nek5000.py -- Nek5000 interface: everything is converted to the package
convention (z vertical, Neko statistics names), so the rest of the package
does not know which code produced the data.

Nek5000 ABL cases usually have y vertical. The conversion is the proper
rotation (no mirror, Coriolis veering keeps its sign)

        x -> x,   y_nek -> z,   z_nek -> -y        u -> u,  v_nek -> w,  w_nek -> -v

applied to the field files (fields.read_field(..., vertical='y')) and, by
renaming + sign changes, to the statistics.

Statistics: KTH-toolbox 'sts' files
-----------------------------------
`load_stats` reads the statistics files of the KTH Nek5000 toolbox
(``sts<case>0.f00001`` ...; 2D, averaged in time and in the spanwise direction,
or 3D), averages them over the remaining homogeneous direction(s) with the GLL
weights, and returns the same raw Dataset (time, z) as stats.load_stats for
Neko. The order of the 84 variables is KTH_FIELDS, taken from the ``stat.f`` of
the stable_16_TKE case (the toolbox stat.f is edited per project, so if yours
differs pass ``fields_map=`` or edit the table).

With the one-equation SGS TKE model (Moeng & Sullivan 2015, as in tke.f) the
second scalar is the SGS TKE e. It is stored as 'k' in the raw statistics and
used as 'tke_sgs' by profiles.derive; its SGS-transport term <(nu+2Km) de/dz>
is kept as 'dkdz_diff'.

Physical parameters (g, theta0, z0, f, ...) are read from the .par (WMLES
section) and the .usr (``parameter(name = value)`` lines) by `read_par`; pass
anything else with Run(..., case=dict(...)).
"""

import configparser
import glob
import os
import re

import numpy as np
import xarray as xr

from . import case as _case
from . import fields as _fields
from . import stats as _stats

# KTH statistics toolbox as in the case's stat.f (Nek5000 axes: v = vertical velocity,
# d/dy = vertical; 'k' = SGS TKE = scalar 2; 'nu_*' = <(nu+nu_t) d_u/d_x>, 'xi_*' = <(alpha+K_h) dT/dx>,
# 'ks_*' = <(nu+2Km) de/dx>)
KTH_FIELDS = {
    1: "u", 2: "v", 3: "w", 4: "p", 5: "uu", 6: "vv", 7: "ww", 8: "pp", 9: "uv", 10: "vw",
    11: "uw", 12: "pu", 13: "pv", 14: "pw",
    15: "pdudx", 16: "pdudy", 17: "pdudz", 18: "pdvdx", 19: "pdvdy", 20: "pdvdz",
    21: "pdwdx", 22: "pdwdy", 23: "pdwdz",
    24: "uuu", 25: "vvv", 26: "www", 27: "uuv", 28: "uuw", 29: "uvv", 30: "vvw",
    31: "uww", 32: "vww", 33: "ppp", 34: "pppp", 35: "uvw", 36: "uuuu", 37: "vvvv", 38: "wwww",
    39: "e11", 40: "e22", 41: "e33", 42: "e12", 43: "e13", 44: "e23",
    45: "nu_dudx", 46: "nu_dudy", 47: "nu_dudz", 48: "nu_dvdx", 49: "nu_dvdy", 50: "nu_dvdz",
    51: "nu_dwdx", 52: "nu_dwdy", 53: "nu_dwdz", 54: "nutot",
    55: "t", 56: "tt", 57: "ttt", 58: "tttt", 59: "ut", 60: "vt", 61: "wt",
    62: "dtdx", 63: "dtdy", 64: "dtdz", 65: "xi_dtdx", 66: "xi_dtdy", 67: "xi_dtdz", 68: "xitot",
    69: "taux", 70: "tauy", 71: "tauz",
    72: "k", 73: "kk", 74: "kkk", 75: "kkkk", 76: "uk", 77: "vk", 78: "wk",
    79: "dkdx", 80: "dkdy", 81: "dkdz", 82: "ks_dkdx", 83: "ks_dkdy", 84: "ks_dkdz",
}

# rotation y-up -> z-up: component and direction maps with signs
_COMP = {"u": ("u", 1), "v": ("w", 1), "w": ("v", -1), "p": ("p", 1), "t": ("t", 1), "k": ("k", 1)}
_DIR = {"x": ("x", 1), "y": ("z", 1), "z": ("y", -1)}
_IDX = {"1": "u", "2": "v", "3": "w"}
_RIDX = {"u": "1", "v": "2", "w": "3"}
_ORDER = "uvwptk"


def rotate_name(name):
    """Nek5000 (y-up) statistics name -> (package name, sign)."""
    m = re.fullmatch(r"(p|nu_|xi_|ks_|)d([uvwtk])d([xyz])", name)
    if m:
        c, sc = _COMP[m.group(2)]
        d, sd = _DIR[m.group(3)]
        return f"{m.group(1)}d{c}d{d}", sc * sd
    m = re.fullmatch(r"e([123])([123])", name)
    if m:
        (a, sa), (b, sb) = _COMP[_IDX[m.group(1)]], _COMP[_IDX[m.group(2)]]
        i, j = sorted((_RIDX[a], _RIDX[b]))
        return f"e{i}{j}", sa * sb
    m = re.fullmatch(r"tau([xyz])", name)
    if m:
        d, sd = _DIR[m.group(1)]
        return f"tau{d}", sd
    if name in ("nutot", "xitot"):
        return name, 1
    if set(name) <= set(_ORDER):
        new, sign = [], 1
        for c in name:
            cc, sc = _COMP[c]
            new.append(cc)
            sign *= sc
        return _canonical("".join(new)), sign
    return name, 1


def _canonical(name):
    """Order letters as Neko does (u v w p t) so that names match: 'wu' -> 'uw'."""
    order = {c: i for i, c in enumerate(_ORDER)}
    s = "".join(sorted(name, key=lambda c: order[c]))
    return {"up": "pu", "vp": "pv", "wp": "pw"}.get(s, s)


# ----------------------------------------------------------------------------
# Case parameters
# ----------------------------------------------------------------------------
def _fnum(s):
    try:
        return float(str(s).lower().replace("d", "e"))
    except ValueError:
        return None


def read_usr(path):
    """Constants of the .usr: every ``parameter(name = value)`` with a numeric
    value, plus the WMLES sampling height set in user_set_sampling_height.
    Returns a dict {name_lowercase: float}."""
    found = {}
    files = sorted(glob.glob(os.path.join(path, "*.usr")) + glob.glob(os.path.join(path, "compile", "*.usr")))
    if not files:
        return found
    txt = open(files[0], errors="ignore").read()
    for m in re.finditer(r"parameter\s*\(\s*(\w+)\s*=\s*([-+0-9.eEdD]+)\s*\)", txt, re.I):
        v = _fnum(m.group(2))
        if v is not None:
            found[m.group(1).lower()] = v
    m = re.search(r"wmles_surface_temp\s*=\s*([-+*/(). 0-9eEdD]*time[-+*/(). 0-9eEdD]*)\s*$",
                  txt, re.I | re.M)
    if m:
        found["surface_temperature_expr"] = m.group(1).strip()
    m = re.search(r"wmles_sampling_h\s*=\s*([-+0-9.eEdD]+)", txt, re.I)
    if m:
        found["sampling_h"] = _fnum(m.group(1))
    return found


def casename(path):
    """Case name from SESSION.NAME / the .par file ('n_16'), or None."""
    for d in (path, os.path.dirname(os.path.abspath(path))):
        s = os.path.join(d, "SESSION.NAME")
        if os.path.isfile(s):
            first = open(s).readline().strip()
            if first:
                return first
        pars = sorted(glob.glob(os.path.join(d, "*.par")))
        if pars:
            return os.path.splitext(os.path.basename(pars[0]))[0]
    return None


def read_par(path):
    """CaseInfo from a Nek5000 case folder: .par (general, viscosity, WMLES
    section), SIZE (polynomial order) and .usr (f, u_geo, theta0, g, sampling
    height). Anything missing stays None: complete it with Run(case=dict(...))."""
    raise NotImplementedError

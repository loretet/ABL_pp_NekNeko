"""
stats.py -- Read the statistics written by Neko (and Nek5000) into xarray.

Which file for what?
--------------------
* ``fluid_stats*.csv``            (avg_direction "xy")  mean velocity/pressure moments
* ``scalar_stats_<name>*.csv``    (avg_direction "xy")  scalar moments (needs fluid_stats)
* ``fluid_sgs_stats*.csv``        <nu_t> and SGS stresses
* ``scalar_sgs_stats*.csv``       <alpha_t> and SGS scalar fluxes
These are cheap and should be your first choice for *anything averaged*
(profiles, fluxes, budgets, ABL height, u*, L, ...).

For 3D statistics (no avg_direction) and for Nek5000 (avg/rms/rm2 files of
avg_all, or KTH-toolbox-like "sts" files) the fld files are read and averaged
horizontally with the exact GLL quadrature: the output is identical in shape
to the CSV route, so everything downstream works the same way.

What you get
------------
A *raw* Dataset with dims (time, z): the variables are exactly the Neko
"raw statistics" <u>, <uu>, <uuw>, <p dudx>, ... (NOT central moments yet).
Turn them into physical profiles with `profiles.derive` (after an optional
weighted time average with `profiles.time_average`).

Naming
------
fluid basic : p u v w pp uu vv ww uv uw vw
fluid full  : + uuu vvv www uuv uuw uvv uvw vvw uww vww uuuu vvvv wwww ppp pppp
                pu pv pw pdudx ... pdwdz e11 e22 e33 e12 e13 e23
scalar      : the Neko letter 's' is replaced by the symbol you choose
              (default 't' for temperature): t ut vt wt tt ttt tttt utt ...
fluid SGS   : nut, uu_sgs, vv_sgs, ww_sgs, uv_sgs, uw_sgs, vw_sgs
scalar SGS  : alpha_t, ut_sgs, vt_sgs, wt_sgs

SIGN CONVENTION: Neko stores <2 nu_t S_ij> and <alpha_t ds/dx_i>. Here they are
converted on reading into the actual SGS fluxes
    uw_sgs = -<2 nu_t S_13>,     wt_sgs = -<alpha_t dtheta/dz>
so that resolved + SGS = total flux directly (e.g. uw + uw_sgs).
(Only the deviatoric part of the SGS stress is available, as in Neko.)

Time weighting
--------------
Each CSV time stamp is the average over the window since the previous write
(the first window starts at the stats `start_time`). The window length is
stored as the coordinate ``window`` (time) and used by `profiles.time_average`.
"""

import glob
import os
import re
import warnings

import numpy as np
import xarray as xr

from . import sem

# ----------------------------------------------------------------------------
# Neko variable lists (order as in the Neko statistics guide)
# ----------------------------------------------------------------------------
FLUID_BASIC = ["p", "u", "v", "w", "pp", "uu", "vv", "ww", "uv", "uw", "vw"]
FLUID_FULL = FLUID_BASIC + [
    "uuu", "vvv", "www", "uuv", "uuw", "uvv", "uvw", "vvw", "uww", "vww",
    "uuuu", "vvvv", "wwww", "ppp", "pppp", "pu", "pv", "pw",
    "pdudx", "pdudy", "pdudz", "pdvdx", "pdvdy", "pdvdz", "pdwdx", "pdwdy", "pdwdz",
    "e11", "e22", "e33", "e12", "e13", "e23",
]
_SCALAR_BASIC = ["s", "us", "vs", "ws", "ss"]
_SCALAR_FULL = _SCALAR_BASIC + [
    "sss", "ssss", "uss", "vss", "wss", "uus", "vvs", "wws", "uvs", "uws", "vws", "ps",
    "pdsdx", "pdsdy", "pdsdz", "udsdx", "udsdy", "udsdz", "vdsdx", "vdsdy", "vdsdz",
    "wdsdx", "wdsdy", "wdsdz", "sdudx", "sdudy", "sdudz", "sdvdx", "sdvdy", "sdvdz",
    "sdwdx", "sdwdy", "sdwdz", "ess", "eus", "evs", "ews",
]
FLUID_SGS = ["nut", "uu_sgs", "vv_sgs", "ww_sgs", "uv_sgs", "uw_sgs", "vw_sgs"]


def scalar_names(sym="t", full=True):
    """Scalar statistics names with the Neko 's' replaced by `sym`."""
    if sym in "uvwpde" or len(sym) != 1:
        raise ValueError("scalar symbol must be a single letter not in 'uvwpde'")
    return [n.replace("s", sym) for n in (_SCALAR_FULL if full else _SCALAR_BASIC)]


def scalar_sgs_names(sym="t"):
    return [f"alpha_{sym}", f"u{sym}_sgs", f"v{sym}_sgs", f"w{sym}_sgs"]


def _sgs_sign(ds):
    """Neko stores +<2 nut Sij> and +<alpha ds/dxi>: flip to physical SGS fluxes."""
    for k in list(ds.data_vars):
        if k.endswith("_sgs"):
            ds[k] = -ds[k]
    return ds


# ----------------------------------------------------------------------------
# CSV reading
# ----------------------------------------------------------------------------
def _read_table(path):
    try:
        import pandas as pd
        return pd.read_csv(path, header=None, comment="#").to_numpy(dtype=float)
    except ImportError:  # pragma: no cover
        return np.loadtxt(path, delimiter=",", ndmin=2)


def _detect_N(z_block):
    """Polynomial order from the duplicated element-interface heights."""
    zr = np.round(z_block, 6)
    zu = np.unique(zr)
    nel = zr.size - zu.size + 1
    if nel < 1 or (zu.size - 1) % nel:
        return None
    N = (zu.size - 1) // nel
    return N if sem.is_gll_axis(zu, N) else None


def _table_to_dataset(arr, names, source=""):
    ncol = arr.shape[1] - 2
    if ncol != len(names):
        raise ValueError(f"{source}: {ncol} data columns, expected {len(names)}")
    t = np.round(arr[:, 0], 6)
    z = np.round(arr[:, 1], 6)
    N = _detect_N(arr[t == t[0], 1])
    tu, it = np.unique(t, return_inverse=True)
    zu, iz = np.unique(z, return_inverse=True)
    data = np.zeros((tu.size, zu.size, ncol))
    cnt = np.zeros((tu.size, zu.size, 1))
    np.add.at(data, (it, iz), arr[:, 2:])        # duplicated interface points
    np.add.at(cnt, (it, iz), 1.0)                # are averaged
    data /= np.maximum(cnt, 1)
    ds = xr.Dataset({n: (("time", "z"), data[:, :, i]) for i, n in enumerate(names)},
                    coords={"time": tu, "z": zu})
    if N is not None:
        ds.z.attrs["N"] = N
    ds.z.attrs["units"] = "m"
    ds.attrs["source"] = source
    return ds


def _read_restarts(path_or_stem, names):
    """Read one csv, or all restart pieces 'stemR.csv' and concatenate in time."""
    if os.path.isfile(path_or_stem):
        files = [path_or_stem]
    else:
        folder, stem = os.path.split(path_or_stem)
        pat = re.compile(re.escape(stem) + r"(\d+)\.csv$")
        files = sorted((f for f in glob.glob(os.path.join(folder or ".", stem + "*.csv"))
                        if pat.match(os.path.basename(f))),
                       key=lambda f: int(pat.match(os.path.basename(f)).group(1)))
    if not files:
        raise FileNotFoundError(path_or_stem)
    dss = [_table_to_dataset(_read_table(f), names, f) for f in files]
    ds = xr.concat(dss, "time", combine_attrs="drop_conflicts") if len(dss) > 1 else dss[0]
    ds = ds.sortby("time")
    _, keep = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=keep)
    ds.attrs["files"] = ", ".join(files)
    return ds


def _guess_set(path, basic_len, full_len):
    f = path if os.path.isfile(path) else sorted(glob.glob(path + "[0-9]*.csv"))[0]
    with open(f) as fh:
        ncol = len(fh.readline().split(",")) - 2
    if ncol == full_len:
        return "full"
    if ncol == basic_len:
        return "basic"
    raise ValueError(f"{f}: {ncol} columns matches neither basic nor full set")


def read_fluid_stats(path):
    """fluid_stats csv (file, or stem like 'output/fluid_stats' for restarts)."""
    kind = _guess_set(path, len(FLUID_BASIC), len(FLUID_FULL))
    ds = _read_restarts(path, FLUID_FULL if kind == "full" else FLUID_BASIC)
    ds.attrs["fluid_set"] = kind
    return ds


def read_scalar_stats(path, sym="t"):
    """scalar_stats csv; Neko 's' is renamed to `sym` (default 't')."""
    kind = _guess_set(path, len(_SCALAR_BASIC), len(_SCALAR_FULL))
    ds = _read_restarts(path, scalar_names(sym, kind == "full"))
    ds.attrs[f"scalar_set_{sym}"] = kind
    return ds


def read_fluid_sgs_stats(path):
    return _sgs_sign(_read_restarts(path, FLUID_SGS))


def read_scalar_sgs_stats(path, sym="t"):
    return _sgs_sign(_read_restarts(path, scalar_sgs_names(sym)))


# ----------------------------------------------------------------------------
# Time windows
# ----------------------------------------------------------------------------
def add_windows(ds, start_time=None):
    """Store the averaging-window length of every time stamp as coord 'window'."""
    t = ds.time.values
    win = np.diff(t, prepend=np.nan)
    if t.size == 1:
        win[0] = (t[0] - start_time) if start_time is not None else 1.0
    elif start_time is not None and start_time < t[0]:
        win[0] = t[0] - start_time
    else:
        win[0] = win[1]
    ds = ds.assign_coords(window=("time", win))
    ds.window.attrs["description"] = "length of the averaging window ending at time"
    return ds


# ----------------------------------------------------------------------------
# One-stop loader
# ----------------------------------------------------------------------------
def _first_existing(folder, stems):
    for stem in stems:
        p = os.path.join(folder, stem)
        if glob.glob(p + "[0-9]*.csv"):
            return p
    return None


def load_stats(folder, scalars=None, start_time=None, case=None, verbose=True):
    """Load every Neko statistics CSV in `folder` into one raw Dataset (time, z).

    Parameters
    ----------
    folder     : Neko output folder.
    scalars    : {neko_field_name: symbol}, default {'temperature': 't'}.
    start_time : statistics start_time (for the first window weight). If None
                 it is taken from `case` (a CaseInfo from read_case) if given.
    """
    scalars = scalars or {"temperature": "t"}
    if start_time is None and case is not None:
        start_time = case.stats_start_time
    parts, found = [], []

    p = _first_existing(folder, ["fluid_stats"])
    if p:
        parts.append(read_fluid_stats(p)); found.append(f"fluid_stats ({parts[-1].attrs['fluid_set']})")
    p = _first_existing(folder, ["fluid_sgs_stats"])
    if p:
        parts.append(read_fluid_sgs_stats(p)); found.append("fluid_sgs_stats")
    for name, sym in scalars.items():
        p = _first_existing(folder, [f"scalar_stats_{name}", "scalar_stats"])
        if p:
            parts.append(read_scalar_stats(p, sym))
            found.append(f"scalar_stats_{name} ({parts[-1].attrs[f'scalar_set_{sym}']}) -> '{sym}'")
        p = _first_existing(folder, [f"scalar_sgs_stats_{name}", "scalar_sgs_stats"])
        if p:
            parts.append(read_scalar_sgs_stats(p, sym)); found.append(f"scalar_sgs_stats ({name})")
    if not parts:
        raise FileNotFoundError(f"No Neko statistics csv found in {folder}")

    ntimes = [d.time.size for d in parts]
    N = next((d.z.attrs["N"] for d in parts if "N" in d.z.attrs), None)
    attrs = {}
    for d in parts:
        attrs.update({k: v for k, v in d.attrs.items() if k not in ("source", "files")})
    ds = xr.merge(parts, join="inner", compat="override", combine_attrs="drop")
    if len(set(ntimes)) > 1:
        warnings.warn(f"Stats files have different time stamps {ntimes}; kept the common ones "
                      f"({ds.time.size}).")
    if N is not None:
        ds.z.attrs["N"] = N
    ds.z.attrs["units"] = "m"
    ds.attrs.update(attrs)
    ds.attrs["start_time"] = start_time if start_time is not None else np.nan
    ds = add_windows(ds, start_time)
    if verbose:
        print(f"[nekabl] loaded {', '.join(found)}")
        print(f"[nekabl] {ds.time.size} windows, t = {ds.time.values[0]:.1f} ... "
              f"{ds.time.values[-1]:.1f}, {ds.z.size} heights, N = {N}")
    return ds


# ----------------------------------------------------------------------------
# Statistics stored as fld files (3D Neko stats, Nek5000)
# ----------------------------------------------------------------------------
_SLOTS = ["p", "u", "v", "w", "t"] + [f"s{i}" for i in range(1, 60)]


def read_stats_fld(path, kind="fluid", sym="t", mesh=None):
    """Horizontally average a 3D statistics fld file -> raw Dataset (time, z).

    kind : 'fluid', 'scalar', 'fluid_sgs' or 'scalar_sgs' (Neko slot order).
    Nek5000 KTH-toolbox 'sts' files follow the same 44-field fluid order
    (assumed; check with your toolbox version).
    """
    from . import fields

    ds3 = fields.read_field(path, mesh=mesh)
    nvar = len(ds3.data_vars)
    if kind == "fluid":
        names = FLUID_FULL if nvar >= len(FLUID_FULL) else FLUID_BASIC
    elif kind == "scalar":
        names = scalar_names(sym, nvar >= len(_SCALAR_FULL))
    elif kind == "fluid_sgs":
        names = FLUID_SGS
    elif kind == "scalar_sgs":
        names = scalar_sgs_names(sym)
    else:
        raise ValueError(kind)
    if nvar != len(names):
        warnings.warn(f"{path}: {nvar} fields in the file, {len(names)} expected for '{kind}'")
    out = xr.Dataset({n: fields.hmean(ds3[s]) for n, s in zip(names, _SLOTS) if s in ds3})
    out = out.drop_vars([c for c in ("wz",) if c in out.coords]).expand_dims(time=[ds3.attrs["time"]])
    out.z.attrs["N"] = ds3.attrs["N"]
    return _sgs_sign(out) if "sgs" in kind else out


def load_stats_fld(files, kind="fluid", sym="t", start_time=None, mesh=None):
    """Series of 3D statistics fld files -> raw Dataset (time, z) with windows."""
    files = sorted(glob.glob(files)) if isinstance(files, str) else list(files)
    ds = xr.concat([read_stats_fld(f, kind, sym, mesh) for f in files], "time")
    return add_windows(ds, start_time)


def read_nek5000_avg(avg, rms, rm2=None, sym="t", mesh=None):
    """Nek5000 avg_all output (avg*, rms*, rm2* files of the same time) -> raw
    basic statistics (u v w p t uu vv ww pp tt [uv vw uw]) as Dataset (time, z).
    Note: avg_all does not provide the heat fluxes <wt>."""
    from . import fields

    a = fields.read_field(avg, mesh=mesh)
    r = fields.read_field(rms, mesh=mesh or fields._default_mesh(avg))
    out = {k: fields.hmean(a[k]) for k in ("u", "v", "w", "p") if k in a}
    if "t" in a:
        out[sym] = fields.hmean(a["t"])
        out[sym + sym] = fields.hmean(r["t"])
    for k, n in (("u", "uu"), ("v", "vv"), ("w", "ww"), ("p", "pp")):
        out[n] = fields.hmean(r[k])
    if rm2 is not None:
        r2 = fields.read_field(rm2, mesh=mesh or fields._default_mesh(avg))
        for k, n in (("u", "uv"), ("v", "vw"), ("w", "uw")):
            out[n] = fields.hmean(r2[k])
    ds = xr.Dataset(out).expand_dims(time=[a.attrs["time"]])
    ds = ds.drop_vars([c for c in ("wz",) if c in ds.coords])
    ds.z.attrs["N"] = a.attrs["N"]
    ds.attrs["fluid_set"] = "basic"
    return add_windows(ds)

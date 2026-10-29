"""
fields.py -- Read Neko / Nek5000 field files (*.f00000, "fld" format) on
Cartesian box meshes and turn them into xarray Datasets with dims (z, y, x).

Use field files sparingly (they are big); for mean quantities prefer the
statistics CSV files (see stats.py). Field files are needed for:
    * visualisation (slices)                       -> visualization.py
    * spectra, autocorrelations, PDFs, quadrants    -> turbulence.py
    * profiles of quantities that are not in the stats (e.g. SGS TKE)

Data layout
-----------
The SEM grid is kept as is: x, y, z are the *unique* GLL coordinates of the
box (shared element faces appear once; their element copies are averaged,
which loses nothing for continuous fields -- checked against pymech). The polynomial order is stored in
``ds.x.attrs['N']`` (same for y, z) and the GLL quadrature weights as
coordinates ``wx``, ``wy``, ``wz``: an exact horizontal average is

    (f * ds.wx * ds.wy).sum(('x','y')) / (ds.wx.sum() * ds.wy.sum())

which is what `hmean` does.

Variable names
--------------
Slots of the fld file are named  u, v, w, p, t (temperature slot), s1, s2, ...
Use `rename` to give the passive scalars meaningful names, e.g. for a Neko run
with the Deardorff model:  rename={'s1': 'e', 's2': 'delta', 's3': 'nut'}.

Nek5000 cases with y vertical: read with vertical='y' and the data are
rotated so that z is vertical everywhere in the package.
"""

import glob
import os
import re

import numpy as np
import xarray as xr

from . import sem

_MESH_CACHE = {}


# ----------------------------------------------------------------------------
# Low-level reading
# ----------------------------------------------------------------------------
def read_header(path):
    """Parse the 132-byte header of a fld file. Returns a dict."""
    with open(path, "rb") as f:
        raw = f.read(132).decode("ascii", errors="ignore")
        tag_bytes = f.read(4)
    tok = raw.split()
    if not tok or tok[0] != "#std":
        raise IOError(f"{path} does not look like a Nek/Neko fld file")
    hdr = dict(
        wdsize=int(tok[1]), lx=int(tok[2]), ly=int(tok[3]), lz=int(tok[4]),
        nel=int(tok[5]), nel_total=int(tok[6]), time=float(tok[7]),
        step=int(tok[8]), fid=int(tok[9]), nfiles=int(tok[10]),
        code=tok[11] if len(tok) > 11 else "",   # Nek5000 adds "0.0E+00 F" after it
    )
    little = np.frombuffer(tag_bytes, "<f4")[0]
    hdr["endian"] = "<" if abs(little - 6.54321) < 1e-4 else ">"
    hdr["gdim"] = 2 if hdr["lz"] == 1 else 3
    hdr["groups"] = _groups(hdr["code"], hdr["gdim"])
    return hdr


def _groups(code, gdim=3):
    """'XUPTS03' -> [('x','y','z'), ('u','v','w'), ('p',), ('t',), ('s1',), ...]
    (vectors have 2 components in 2D files)."""
    groups = []
    if "X" in code:
        groups.append(("x", "y", "z")[:gdim])
    if "U" in code:
        groups.append(("u", "v", "w")[:gdim])
    if "P" in code:
        groups.append(("p",))
    if "T" in code:
        groups.append(("t",))
    m = re.search(r"S(\d+)", code)
    if m:
        groups += [(f"s{i}",) for i in range(1, int(m.group(1)) + 1)]
    return groups


def _read_raw(path, variables=None, hdr=None):
    """Read element data. Returns hdr, element map, {name: (nel, lz, ly, lx)}."""
    hdr = hdr or read_header(path)
    if hdr["nfiles"] > 1:
        raise NotImplementedError("Multi-file fld output (nfiles > 1) is not supported.")
    e, ws = hdr["endian"], hdr["wdsize"]
    dt = np.dtype(f"{e}f{ws}")
    nel, lx, ly, lz = hdr["nel"], hdr["lx"], hdr["ly"], hdr["lz"]
    npts = lx * ly * lz
    emap = np.fromfile(path, dtype=f"{e}i4", count=nel, offset=136)
    offset = 136 + 4 * nel
    out = {}
    for g in hdr["groups"]:
        size = nel * len(g) * npts
        if variables is None or any(v in variables for v in g) or g[0] == "x":
            blk = np.fromfile(path, dtype=dt, count=size, offset=offset)
            blk = blk.reshape(nel, len(g), lz, ly, lx)
            for j, name in enumerate(g):
                if variables is None or name in variables or g[0] == "x":
                    out[name] = blk[:, j].astype(np.float64)
        offset += size * ws
    return hdr, emap, out


def _cluster(c, tol):
    """Integer index of each value in c after merging values closer than tol."""
    order = np.argsort(c)
    cs = c[order]
    new = np.concatenate(([True], np.diff(cs) > tol))
    idx_sorted = np.cumsum(new) - 1
    idx = np.empty_like(idx_sorted)
    idx[order] = idx_sorted
    return idx


def _mesh_structure(mesh_path):
    """Structured box layout from the coordinates in a mesh-bearing fld file.

    Elements are placed on a (nelz, nely, nelx) lattice from their centroids;
    every GLL node gets a linear index in the global array of unique points.
    """
    key = os.path.abspath(mesh_path)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    hdr, emap, d = _read_raw(mesh_path, variables=[])
    if "x" not in d:
        raise IOError(f"{mesh_path} has no coordinates (X); pass mesh=<file with X>")
    order = np.argsort(emap)
    names = ("x", "y", "z")[:hdr["gdim"]]
    C = [d[k][order] for k in names]
    N = hdr["lx"] - 1
    idx = []
    for c in C:
        cen = c.mean(axis=(1, 2, 3))
        ext = (c.max(axis=(1, 2, 3)) - c.min(axis=(1, 2, 3))).min()
        idx.append(_cluster(cen, 0.25 * ext))
    nels = [i.max() + 1 for i in idx]
    if np.prod(nels) != hdr["nel"]:
        raise ValueError("Mesh is not a structured Cartesian box; cannot build regular arrays.")
    n = [ne * N + 1 for ne in nels]                     # unique points per direction
    r = np.arange(N + 1)
    gx = (idx[0][:, None] * N + r)[:, None, None, :]
    gy = (idx[1][:, None] * N + r)[:, None, :, None]
    if hdr["gdim"] == 3:
        gz = (idx[2][:, None] * N + r)[:, :, None, None]
        li = (gz * n[1] + gy) * n[0] + gx
        shape = (n[2], n[1], n[0])
    else:
        li = gy * n[0] + gx + np.zeros((1, 1, 1, 1), int)
        shape = (n[1], n[0])
    li = np.broadcast_to(li, C[0].shape).ravel()
    st = dict(N=N, li=li, shape=shape, count=np.bincount(li, minlength=np.prod(shape)),
              emap=emap, gdim=hdr["gdim"])
    G = [_assemble(c, st)[0] for c in C]
    L = max(np.ptp(g) for g in G)
    for k, (name, g) in enumerate(zip(names, G)):
        ax = len(shape) - 1 - k                         # x is the last axis
        others = tuple(i for i in range(len(shape)) if i != ax)
        if np.ptp(g, axis=others).max() > 1e-4 * L:
            raise ValueError("Mesh elements are not axis-aligned: not a Cartesian box mesh.")
        st[name] = g.mean(axis=others)
    _MESH_CACHE[key] = st
    return st

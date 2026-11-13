"""
run.py -- One object per simulation, the same for Neko and Nek5000.

    a = nk.Run("cases/neutral", label="neutral")          # Neko (has a .case)
    b = nk.Run("cases/gabls_nek", label="Nek5000")        # Nek5000 (has a .par / SIZE)

    prof = {r.label: r.profile(2000, 3000) for r in (a, b)}
    nk.plot_profiles(prof, ["M", "T"], zi=True)            # z/zi of each run

Everything a Run returns is in the package convention (z vertical, Neko
names), so every function of the package works on both codes:

    run.case          physical parameters (CaseInfo; see case.py)
    run.raw           raw statistics (time, z)                    [cached]
    run.profile(t0, t1)   time-averaged derived profiles (z), with the bulk
                      scales (u*, L, zi, w*, ...) in .attrs        [cached]
    run.history()     derived profiles for every statistics window (time, z)
    run.timeseries()  u*, L, zi, w* ... per statistics window
    run.fields(...)   field snapshots (time, z, y, x)
    run.log()         (steps, meta) from the log file, or None

Default averaging window of profile(): the second half of the statistics.
"""

import glob
import os

import numpy as np

from . import case as _case
from . import fields as _fields
from . import plotstyle as _ps
from . import stats as _stats


def detect_kind(path):
    """'neko' if a .case file is found, 'nek5000' if a .par/.rea/.usr/SIZE file is.
    A .case in the folder itself wins, then Nek5000 markers, then a .case in
    the parent (Neko output sub-folders)."""
    if glob.glob(os.path.join(path, "*.case")):
        return "neko"
    for pat in ("*.par", "*.rea", "*.re2", "SIZE", "*.usr", os.path.join("compile", "SIZE"),
                os.path.join("compile", "*.usr")):
        if glob.glob(os.path.join(path, pat)):
            return "nek5000"
    if _case.find_case(path):
        return "neko"
    raise FileNotFoundError(f"{path}: neither a Neko (.case) nor a Nek5000 (.par/.rea/SIZE) case")
